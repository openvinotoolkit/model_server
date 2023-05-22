// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "s3filesystem.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>

#include "logging.hpp"
#include "stringutils.hpp"

namespace ovms {

namespace s3 = Aws::S3;
namespace fs = std::filesystem;

StatusCode S3FileSystem::parsePath(const std::string& path, std::string* bucket, std::string* object) {
    std::smatch sm;

    if (isPathEscaped(path)) {
        SPDLOG_LOGGER_ERROR(s3_logger, "Path {} escape with .. is forbidden.", path);
        return StatusCode::PATH_INVALID;
    }

    if (!std::regex_match(path, sm, s3_regex_)) {
        int bucket_start = path.find(FileSystem::S3_URL_PREFIX) + FileSystem::S3_URL_PREFIX.size();
        int bucket_end = path.find("/", bucket_start);

        if (bucket_end > bucket_start) {
            *bucket = path.substr(bucket_start, bucket_end - bucket_start);
            *object = path.substr(bucket_end + 1);
        } else {
            *bucket = path.substr(bucket_start);
            *object = "";
        }
    } else {
        *bucket = sm[3];
        *object = sm[4];
    }

    if (bucket->empty()) {
        SPDLOG_LOGGER_ERROR(s3_logger, "No bucket name found in path {}", path);

        return StatusCode::S3_BUCKET_NOT_FOUND;
    }

    return StatusCode::OK;
}

S3FileSystem::S3FileSystem(const Aws::SDKOptions& options, const std::string& s3_path) :
    options_(options),
    s3_regex_(FileSystem::S3_URL_PREFIX + "([0-9a-zA-Z-.]+):([0-9]+)/([0-9a-z.-]+)(((/"
                                          "[0-9a-zA-Z.-_]+)*)?)"),
    proxy_regex_("^(https?)://(([^:]{1,128}):([^@]{1,256})@)?([^:/]{1,255})(:([0-9]{1,5}))?/?") {
    Aws::Client::ClientConfiguration config;
    Aws::Auth::AWSCredentials credentials;

    const char* profile_name = std::getenv("AWS_PROFILE");
    const char* secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
    const char* key_id = std::getenv("AWS_ACCESS_KEY_ID");
    const char* region = std::getenv("AWS_REGION");
    const char* session_token = std::getenv("AWS_SESSION_TOKEN");
    const char* s3_endpoint = std::getenv("S3_ENDPOINT");
    const char* http_proxy = std::getenv("http_proxy") != nullptr ? std::getenv("http_proxy") : std::getenv("HTTP_PROXY");
    const char* https_proxy = std::getenv("https_proxy") != nullptr ? std::getenv("https_proxy") : std::getenv("HTTPS_PROXY");
    const std::string default_proxy = https_proxy != nullptr ? std::string(https_proxy) : http_proxy != nullptr ? std::string(http_proxy) : "";

    if ((secret_key != NULL) && (key_id != NULL)) {
        credentials.SetAWSAccessKeyId(key_id);
        credentials.SetAWSSecretKey(secret_key);
        config = Aws::Client::ClientConfiguration();
        if (region != NULL) {
            config.region = region;
        }
        if (session_token != NULL) {
            credentials.SetSessionToken(session_token);
        }
    } else if (profile_name) {
        config = Aws::Client::ClientConfiguration(profile_name);
    } else {
        config = Aws::Client::ClientConfiguration("default");
    }

    std::string host_name, host_port, bucket, object;
    std::smatch sm;
    if (std::regex_match(s3_path, sm, s3_regex_)) {
        host_name = sm[1];
        host_port = sm[2];
        bucket = sm[3];
        object = sm[4];

        config.endpointOverride = Aws::String(host_name + ":" + host_port);
        config.scheme = Aws::Http::Scheme::HTTP;
    }
    if (s3_endpoint != NULL) {
        std::string endpoint(s3_endpoint);
        if (endpoint.rfind("http://") != std::string::npos) {
            endpoint = endpoint.substr(7);
        }
        config.endpointOverride = Aws::String(endpoint.c_str());
        config.scheme = Aws::Http::Scheme::HTTP;
    }

    if (!default_proxy.empty()) {
        if (std::regex_match(default_proxy, sm, proxy_regex_)) {
            config.proxyHost = sm[5].str();
            config.proxyPort = std::stoi(sm[7].str());
            config.proxyScheme = sm[1].str().size() == 4 ? Aws::Http::Scheme::HTTP : Aws::Http::Scheme::HTTPS;
            if (!sm[3].str().empty()) {
                config.proxyUserName = sm[3].str();
            }
            if (!sm[4].str().empty()) {
                config.proxyPassword = sm[4].str();
            }
        } else {
            SPDLOG_LOGGER_ERROR(s3_logger, "Couldn't parse proxy: {}", default_proxy);
        }
    }

    if (profile_name) {
        client_ = s3::S3Client(
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
            false);
    } else if ((secret_key != NULL) && (key_id != NULL)) {
        client_ = s3::S3Client(
            credentials,
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
            false);
    } else {
        std::shared_ptr<Aws::Auth::AnonymousAWSCredentialsProvider> provider = std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>();
        client_ = s3::S3Client(
            provider,
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
            true);
    }
}

S3FileSystem::~S3FileSystem() {
    Aws::ShutdownAPI(options_);
}

StatusCode S3FileSystem::fileExists(const std::string& path, bool* exists) {
    *exists = false;
    std::string bucket, object;

    auto status = parsePath(path, &bucket, &object);
    if (status != StatusCode::OK) {
        return status;
    }

    s3::Model::HeadObjectRequest head_request;
    head_request.SetBucket(bucket.c_str());
    head_request.SetKey(object.c_str());

    auto head_object_outcome = client_.HeadObject(head_request);
    if (head_object_outcome.IsSuccess()) {
        *exists = true;
        return StatusCode::OK;
    }

    bool is_dir;
    status = isDirectory(path, &is_dir);
    if (status != StatusCode::OK) {
        return status;
    }
    *exists = is_dir;

    return StatusCode::OK;
}

StatusCode S3FileSystem::isDirectory(const std::string& path, bool* is_dir) {
    *is_dir = false;
    std::string bucket, object_path;
    auto status = parsePath(path, &bucket, &object_path);
    if (status != StatusCode::OK) {
        return status;
    }

    // Check if the bucket exists
    s3::Model::HeadBucketRequest head_request;
    head_request.WithBucket(bucket.c_str());

    auto head_bucket_outcome = client_.HeadBucket(head_request);
    if (!head_bucket_outcome.IsSuccess()) {
        SPDLOG_LOGGER_ERROR(s3_logger, "Invalid or missing S3 credentials, or bucket does not exist - {}. {}", bucket, head_bucket_outcome.GetError().GetMessage());
        return StatusCode::S3_METADATA_FAIL;
    }

    // Root case - bucket exists and object path is empty
    if (object_path.empty()) {
        *is_dir = true;
        return StatusCode::OK;
    }

    // List the objects in the bucket
    s3::Model::ListObjectsRequest list_objects_request;
    list_objects_request.SetBucket(bucket.c_str());
    list_objects_request.SetPrefix(appendSlash(object_path).c_str());
    auto list_objects_outcome = client_.ListObjects(list_objects_request);

    if (list_objects_outcome.IsSuccess()) {
        *is_dir = !list_objects_outcome.GetResult().GetContents().empty();
    } else {
        SPDLOG_LOGGER_ERROR(s3_logger, "Failed to list objects with prefix {}", path);
        return StatusCode::S3_FAILED_LIST_OBJECTS;
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectoryContents(const std::string& path, std::set<std::string>* contents) {
    // Parse bucket and dir_path
    std::string bucket, dir_path, full_dir;
    auto status = parsePath(path, &bucket, &dir_path);
    if (status != StatusCode::OK) {
        return status;
    }
    std::string true_path = FileSystem::S3_URL_PREFIX + bucket + '/' + dir_path;

    // Capture the full path to facilitate content listing
    full_dir = appendSlash(dir_path);

    // Issue request for objects with prefix
    s3::Model::ListObjectsRequest objects_request;
    objects_request.SetBucket(bucket.c_str());
    objects_request.SetPrefix(full_dir.c_str());
    auto list_objects_outcome = client_.ListObjects(objects_request);

    if (list_objects_outcome.IsSuccess()) {
        Aws::Vector<Aws::S3::Model::Object> object_list = list_objects_outcome.GetResult().GetContents();
        for (auto const& s3_object : object_list) {
            // In the case of empty directories, the directory itself will appear here
            if (s3_object.GetKey().c_str() == full_dir) {
                continue;
            }

            // We have to make sure that subdirectory contents do not appear here
            std::string name(s3_object.GetKey().c_str());
            int item_start = name.find(full_dir) + full_dir.size();
            int item_end = name.find("/", item_start);

            // Let set take care of subdirectory contents
            std::string item = name.substr(item_start, item_end - item_start);
            contents->insert(item);
        }
    } else {
        SPDLOG_LOGGER_ERROR(s3_logger, "Could not list contents of directory {}", true_path);
        return StatusCode::S3_INVALID_ACCESS;
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectorySubdirs(const std::string& path, std::set<std::string>* subdirs) {
    // Parse bucket and dir_path
    std::string bucket, dir_path;
    auto status = parsePath(path, &bucket, &dir_path);
    if (status != StatusCode::OK) {
        return status;
    }
    std::string true_path = FileSystem::S3_URL_PREFIX + bucket + '/' + dir_path;
    status = getDirectoryContents(true_path, subdirs);
    if (status != StatusCode::OK) {
        return status;
    }

    // Erase non-directory entries...
    for (auto iter = subdirs->begin(); iter != subdirs->end();) {
        bool is_dir;
        status = isDirectory(FileSystem::joinPath({true_path, *iter}), &is_dir);
        if (status != StatusCode::OK) {
            return status;
        }
        if (!is_dir) {
            iter = subdirs->erase(iter);
        } else {
            ++iter;
        }
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::getDirectoryFiles(const std::string& path, std::set<std::string>* files) {
    // Parse bucket and dir_path
    std::string bucket, dir_path;
    auto status = parsePath(path, &bucket, &dir_path);
    if (status != StatusCode::OK) {
        return status;
    }

    std::string true_path = FileSystem::S3_URL_PREFIX + bucket + '/' + dir_path;
    status = getDirectoryContents(true_path, files);
    if (status != StatusCode::OK) {
        return status;
    }

    // Erase directory entries...
    for (auto iter = files->begin(); iter != files->end();) {
        bool is_dir;
        status = isDirectory(FileSystem::joinPath({true_path, *iter}), &is_dir);
        if (status != StatusCode::OK) {
            return status;
        }
        if (is_dir) {
            iter = files->erase(iter);
        } else {
            ++iter;
        }
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::readTextFile(const std::string& path, std::string* contents) {
    bool exists;
    auto status = fileExists(path, &exists);
    if (status != StatusCode::OK) {
        return status;
    }

    if (!exists) {
        SPDLOG_LOGGER_ERROR(s3_logger, "File does not exist at {}", path);
        return StatusCode::S3_FILE_NOT_FOUND;
    }

    std::string bucket, object;
    status = parsePath(path, &bucket, &object);
    if (status != StatusCode::OK) {
        return status;
    }

    // Send a request for the objects metadata
    s3::Model::GetObjectRequest object_request;
    object_request.SetBucket(bucket.c_str());
    object_request.SetKey(object.c_str());

    auto get_object_outcome = client_.GetObject(object_request);
    if (get_object_outcome.IsSuccess()) {
        auto& object_result = get_object_outcome.GetResultWithOwnership().GetBody();

        std::string data = "";
        char c = 0;
        while (object_result.get(c)) {
            data += c;
        }

        *contents = data;
    } else {
        SPDLOG_LOGGER_ERROR(s3_logger, "Failed to get object at {}", path);
        return StatusCode::S3_FILE_INVALID;
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::downloadFileFolder(const std::string& path, const std::string& local_path) {
    bool exists;
    auto status = fileExists(path, &exists);
    if (status != StatusCode::OK) {
        return status;
    }
    if (!exists) {
        SPDLOG_LOGGER_ERROR(s3_logger, "File/folder does not exist at {}", path);
        return StatusCode::S3_FILE_NOT_FOUND;
    }

    std::string effective_path, host_name, host_port, bucket, object;

    std::smatch sm;
    if (std::regex_match(path, sm, s3_regex_)) {
        host_name = sm[1];
        host_port = sm[2];
        bucket = sm[3];
        object = sm[4];
        effective_path = FileSystem::S3_URL_PREFIX + bucket + object;
    } else {
        effective_path = path;
    }

    bool is_dir = false;
    std::set<std::string> contents, files;
    status = isDirectory(effective_path, &is_dir);
    if (status != StatusCode::OK) {
        return status;
    }
    if (is_dir) {
        status = getDirectoryContents(effective_path, &contents);
        if (status != StatusCode::OK) {
            return status;
        }

        for (auto iter = contents.begin(); iter != contents.end(); ++iter) {
            bool is_subdir;
            std::string s3_fpath = FileSystem::joinPath({effective_path, *iter});
            std::string local_fpath = FileSystem::joinPath({local_path, *iter});
            status = isDirectory(s3_fpath, &is_subdir);
            if (status != StatusCode::OK) {
                return status;
            }
            if (is_subdir) {
                // Create local mirror of sub-directories
                int status = mkdir(const_cast<char*>(local_fpath.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);
                if (status == -1) {
                    SPDLOG_LOGGER_ERROR(s3_logger, "Failed to create local folder: {} {} ", local_fpath, strerror(errno));
                    return StatusCode::PATH_INVALID;
                }

                // Add with s3 path
                std::set<std::string> subdir_files;
                auto s = getDirectoryFiles(s3_fpath, &subdir_files);
                if (s != StatusCode::OK) {
                    return s;
                }
                for (auto itr = subdir_files.begin(); itr != subdir_files.end(); ++itr) {
                    files.insert(FileSystem::joinPath({s3_fpath, *itr}));
                }
            } else {
                files.insert(s3_fpath);
            }
        }

        for (auto iter = files.begin(); iter != files.end(); ++iter) {
            if (std::any_of(acceptedFiles.begin(), acceptedFiles.end(), [&iter](const std::string& x) {
                    return iter->size() > 0 && endsWith(*iter, x);
                })) {
                std::string bucket, object;
                status = parsePath(*iter, &bucket, &object);
                if (status != StatusCode::OK) {
                    return status;
                }

                // Send a request for the objects metadata
                s3::Model::GetObjectRequest object_request;
                object_request.SetBucket(bucket.c_str());
                object_request.SetKey(object.c_str());

                auto get_object_outcome = client_.GetObject(object_request);
                if (get_object_outcome.IsSuccess()) {
                    auto& retrieved_file =
                        get_object_outcome.GetResultWithOwnership().GetBody();
                    std::string s3_removed_path = (*iter).substr(effective_path.size());
                    std::string local_file_path = FileSystem::joinPath({local_path, s3_removed_path});
                    std::ofstream output_file(local_file_path.c_str(), std::ios::binary);
                    output_file << retrieved_file.rdbuf();
                    output_file.close();
                } else {
                    SPDLOG_LOGGER_ERROR(s3_logger, "Failed to get object at {}", *iter);
                    return StatusCode::S3_FAILED_GET_OBJECT;
                }
            }
        }
    } else {
        std::string bucket, object;
        auto s = parsePath(effective_path, &bucket, &object);
        if (s != StatusCode::OK) {
            return s;
        }

        // Send a request for the objects metadata
        s3::Model::GetObjectRequest object_request;
        object_request.SetBucket(bucket.c_str());
        object_request.SetKey(object.c_str());

        auto get_object_outcome = client_.GetObject(object_request);
        if (get_object_outcome.IsSuccess()) {
            auto& retrieved_file = get_object_outcome.GetResultWithOwnership().GetBody();
            std::ofstream output_file(local_path.c_str(), std::ios::binary);
            output_file << retrieved_file.rdbuf();
            output_file.close();
        } else {
            SPDLOG_LOGGER_ERROR(s3_logger, "Failed to get object at {}", effective_path);
            return StatusCode::S3_FAILED_GET_OBJECT;
        }
    }

    return StatusCode::OK;
}

StatusCode S3FileSystem::downloadModelVersions(const std::string& path,
    std::string* local_path,
    const std::vector<model_version_t>& versions) {
    auto sc = createTempPath(local_path);
    if (sc != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(s3_logger, "Failed to create a temporary path {}", sc);
        return sc;
    }

    StatusCode result = StatusCode::OK;
    for (auto& ver : versions) {
        std::string versionpath = path;
        if (!endsWith(versionpath, "/")) {
            versionpath.append("/");
        }
        versionpath.append(std::to_string(ver));
        std::string lpath = *local_path;
        if (!endsWith(lpath, "/")) {
            lpath.append("/");
        }
        lpath.append(std::to_string(ver));
        fs::create_directory(lpath);
        auto status = downloadFileFolder(versionpath, lpath);
        if (status != StatusCode::OK) {
            result = status;
            SPDLOG_LOGGER_ERROR(s3_logger, "Failed to download model version {}", versionpath);
        }
    }

    return result;
}

StatusCode S3FileSystem::deleteFileFolder(const std::string& path) {
    SPDLOG_LOGGER_DEBUG(s3_logger, "Deleting local file folder {}", path);
    if (::remove(path.c_str()) == 0) {
        return StatusCode::OK;
    } else {
        SPDLOG_LOGGER_ERROR(s3_logger, "Unable to remove local path: {}", path);
        return StatusCode::FILE_INVALID;
    }
}
}  // namespace ovms
