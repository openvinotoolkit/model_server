//*****************************************************************************
// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "openai_completions.hpp"

#include <cmath>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#define STB_IMAGE_IMPLEMENTATION
#include "../../logging.hpp"
#include "../../profiler.hpp"
#pragma warning(push)
#pragma warning(disable : 6262)
#include "stb_image.h"  // NOLINT
#pragma warning(default : 6262)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include <curl/curl.h>
#include <regex>

using namespace rapidjson;

namespace ovms {

constexpr size_t DEFAULT_MAX_STOP_WORDS = 16;  // same as deep-seek

absl::Status OpenAIChatCompletionsHandler::parseCompletionsPart() {
    // prompt: string
    auto it = doc.FindMember("prompt");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsString()) {
            return absl::InvalidArgumentError("prompt is not a string");
        } else {
            request.prompt = it->value.GetString();
        }
    }
    if (!request.prompt.has_value() || !request.prompt.value().size()) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "prompt is missing");
    }
    // logprobs: int; 1 value allowed
    it = doc.FindMember("logprobs");
    if (it != doc.MemberEnd()) {
        if (it->value.IsNull()) {
            request.logprobs = 0;
        } else if (!it->value.IsInt()) {
            return absl::InvalidArgumentError("logprobs accepts integer values");
        } else if (it->value.GetInt() != 1) {
            return absl::InvalidArgumentError("accepted logprobs value is currently 1 only");
        } else {
            request.logprobs = it->value.GetInt();
        }
    }
    if (request.logprobs && request.stream) {
        return absl::InvalidArgumentError("logprobs are not supported in streaming mode.");
    }

    // echo: bool; optional - defaults to false
    it = doc.FindMember("echo");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("echo accepts values true or false");
        request.echo = it->value.GetBool();
    }
    // specific part of max_tokens validation due to echo dependency
    if (request.maxTokens == 0) {
        if (!request.echo)
            return absl::InvalidArgumentError("max_tokens value should be greater than 0 unless echo is set");
    }

    return absl::OkStatus();
}

ov::Tensor loadImageStbi(unsigned char* image, const int x, const int y, const int desiredChannels) {
    if (!image) {
        std::stringstream errorMessage;
        errorMessage << stbi_failure_reason();
        throw std::runtime_error{errorMessage.str()};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t bytes, size_t) {
            if (channels * height * width != bytes) {
                throw std::runtime_error{"Unexpected number of bytes was requested to deallocate."};
            }
            if (image != nullptr) {
                stbi_image_free(image);
                image = nullptr;
            }
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desiredChannels)},
        SharedImageAllocator{image, desiredChannels, y, x});
}

static size_t appendChunkCallback(void* downloadedChunk, size_t size, size_t nmemb,
    void* image) {
    size_t realsize = size * nmemb;
    auto& mem = *static_cast<std::string*>(image);
    mem.append(static_cast<char*>(downloadedChunk), realsize);
    return realsize;
}
#define CURL_SETOPT(setopt)   \
    if (status == CURLE_OK) { \
        status = setopt;      \
    }

static absl::Status downloadImage(const char* url, std::string& image, const int64_t& sizeLimit) {
    CURL* curl_handle = curl_easy_init();
    auto status = curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, appendChunkCallback))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &image))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_MAXFILESIZE, sizeLimit))

    if (status != CURLE_OK) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Setting curl opts failed: {}", curl_easy_strerror(status));
        return absl::InvalidArgumentError("Image downloading failed");
    }

    status = curl_easy_perform(curl_handle);
    if (status != CURLE_OK) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Downloading image failed: {}", curl_easy_strerror(status));
        return absl::InvalidArgumentError("Image downloading failed");
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Downloading image succeeded, {} bytes retrieved", decoded.size());
    }
    curl_easy_cleanup(curl_handle);
    return absl::OkStatus();
}

ov::Tensor loadImageStbiFromMemory(const std::string& imageBytes) {
    int x = 0, y = 0, channelsInFile = 0;
    constexpr int desiredChannels = 3;
    unsigned char* image = stbi_load_from_memory(
        (const unsigned char*)imageBytes.data(), imageBytes.size(),
        &x, &y, &channelsInFile, desiredChannels);
    return loadImageStbi(image, x, y, desiredChannels);
}

ov::Tensor loadImageStbiFromFile(char const* filename) {
    int x = 0, y = 0, channelsInFile = 0;
    constexpr int desiredChannels = 3;
    unsigned char* image = stbi_load(
        filename,
        &x, &y, &channelsInFile, desiredChannels);
    return loadImageStbi(image, x, y, desiredChannels);
}

absl::Status OpenAIChatCompletionsHandler::parseMessages(std::optional<std::string> allowedLocalMediaPath) {
    auto it = doc.FindMember("messages");
    if (it == doc.MemberEnd())
        return absl::InvalidArgumentError("Messages missing in request");
    if (!it->value.IsArray())
        return absl::InvalidArgumentError("Messages are not an array");
    if (it->value.GetArray().Size() == 0)
        return absl::InvalidArgumentError("Messages array cannot be empty");
    bool jsonChanged = false;
    for (size_t i = 0; i < it->value.GetArray().Size(); i++) {
        auto& obj = it->value.GetArray()[i];
        if (!obj.IsObject())
            return absl::InvalidArgumentError("Message is not a JSON object");
        // Add new message to chat history
        request.chatHistory.push_back({});
        for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
            if (!member->name.IsString())
                return absl::InvalidArgumentError("Invalid message structure");
            if (member->value.IsString() && (member->name.GetString() == std::string("role") || member->name.GetString() == std::string("content"))) {
                // Add new field to the last message in history
                // tools handing to be done later
                request.chatHistory.back().insert({member->name.GetString(), member->value.GetString()});
                continue;
            } else {
                if (member->name.GetString() == std::string("content") && member->value.IsArray()) {
                    // Adjust content field format when it is passed as an array of objects (typically with images)
                    if (member->value.GetArray().Size() == 0) {
                        return absl::InvalidArgumentError("Invalid message structure - content array is empty");
                    }
                    jsonChanged = true;
                    Value contentText(rapidjson::kStringType);
                    contentText.SetString("", doc.GetAllocator());
                    for (auto& v : member->value.GetArray()) {
                        if (!v.IsObject()) {
                            return absl::InvalidArgumentError("Invalid message structure - content array should contain objects");
                        }
                        auto entry = v.GetObject();
                        if (!entry.HasMember("type") || !entry["type"].IsString()) {
                            return absl::InvalidArgumentError("Invalid message structure - content object type missing");
                        }
                        auto entryType = entry["type"].GetString();
                        if (entryType == std::string("text")) {
                            if (!entry.HasMember("text") || !entry["text"].IsString()) {
                                return absl::InvalidArgumentError("Invalid message structure - content text missing");
                            }
                            contentText = entry["text"];
                            continue;
                        } else if (entryType == std::string("image_url")) {
                            if (!entry.HasMember("image_url") || !entry["image_url"].IsObject()) {
                                return absl::InvalidArgumentError("Invalid message structure - content image_url missing");
                            }
                            auto imageUrl = entry["image_url"].GetObject();
                            if (!imageUrl.HasMember("url") || !imageUrl["url"].IsString()) {
                                return absl::InvalidArgumentError("Invalid message structure - image_url does not have url field");
                            }
                            std::string url = imageUrl["url"].GetString();
                            std::string pattern = "base64,";
                            std::size_t pos = url.find(pattern);
                            std::string decoded;
                            ov::Tensor tensor;
                            if (pos != std::string::npos) {
                                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
                                size_t offset = pos + pattern.length();
                                if (!absl::Base64Unescape(std::string_view(url.data() + offset, url.size() - offset), &decoded)) {
                                    return absl::InvalidArgumentError("Invalid base64 string in request");
                                }
                                try {
                                    tensor = loadImageStbiFromMemory(decoded);
                                } catch (std::runtime_error& e) {
                                    std::stringstream ss;
                                    ss << "Image parsing failed: " << e.what();
                                    SPDLOG_LOGGER_ERROR(llm_calculator_logger, ss.str());
                                    return absl::InvalidArgumentError(ss.str());
                                }
                            } else if (std::regex_match(url.c_str(), std::regex("^(http|https|ftp|sftp|)://(.*)"))) {
                                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
                                int64_t sizeLimit = 20000000;  // restrict single image size to 20MB
                                auto status = downloadImage(url.c_str(), decoded, sizeLimit);
                                if (status != absl::OkStatus()) {
                                    return status;
                                }
                                curl_global_cleanup();
                                try {
                                    tensor = loadImageStbiFromMemory(decoded);
                                } catch (std::runtime_error& e) {
                                    std::stringstream ss;
                                    ss << "Image parsing failed: " << e.what();
                                    SPDLOG_LOGGER_ERROR(llm_calculator_logger, ss.str());
                                    return absl::InvalidArgumentError("Image parsing failed");
                                }

                            } else {
                                if (!allowedLocalMediaPath.has_value()) {
                                    return absl::InvalidArgumentError("Loading images from local filesystem is disabled.");
                                }
                                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from local filesystem");
                                const auto firstMissmatch = std::mismatch(url.begin(), url.end(), allowedLocalMediaPath.value().begin(), allowedLocalMediaPath.value().end());
                                if (firstMissmatch.second != allowedLocalMediaPath.value().end()) {
                                    return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
                                }
                                try {
                                    tensor = loadImageStbiFromFile(url.c_str());
                                } catch (std::runtime_error& e) {
                                    std::stringstream ss;
                                    ss << "Image file " << url.c_str() << " parsing failed: " << e.what();
                                    SPDLOG_LOGGER_ERROR(llm_calculator_logger, ss.str());
                                    return absl::InvalidArgumentError(ss.str());
                                }
                            }
                            request.imageHistory.push_back({i, tensor});
                        } else {
                            return absl::InvalidArgumentError("Unsupported content type");
                        }
                    }
                    // Pulling out text from nested structure to the "content" field for text and replace whole "content" value for image data
                    // with empty string, since images are stored separately in request.images
                    member->value = contentText;
                    // Add new field to the last message in history if content is text
                    if (member->value.IsString()) {
                        request.chatHistory.back().insert({member->name.GetString(), member->value.GetString()});
                    }
                }
            }
        }
        const auto& lastMessage = request.chatHistory.back();
        if (lastMessage.find("content") == lastMessage.end() || lastMessage.find("role") == lastMessage.end()) {
            return absl::InvalidArgumentError("Every message must have both 'content' and 'role' fields");
        }
    }
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseTools() {
    auto tool_choice_it = doc.FindMember("tool_choice");
    std::string tool_choice{"auto"};
    if (tool_choice_it != doc.MemberEnd()) {
        if (tool_choice_it->value.IsString()) {
            tool_choice = tool_choice_it->value.GetString();
            if (tool_choice != "none" && tool_choice != "auto")
                return absl::InvalidArgumentError("tool_choice should be either 'none' or 'auto'");
        } else if (tool_choice_it->value.IsObject()) {
            auto tool_choice_functionIt = tool_choice_it->value.GetObject().FindMember("function");
            if (tool_choice_functionIt != tool_choice_it->value.GetObject().MemberEnd() && tool_choice_functionIt->value.IsObject()) {
                auto nameIt = tool_choice_functionIt->value.GetObject().FindMember("name");
                if (nameIt != tool_choice_functionIt->value.GetObject().MemberEnd() && nameIt->value.IsString()) {
                    tool_choice = nameIt->value.GetString();
                } else {
                    return absl::InvalidArgumentError("tool_choice.function.name is not a valid string");
                }
            } else {
                return absl::InvalidArgumentError("tool_choice.function is not a valid JSON object");
            }
        } else {
            return absl::InvalidArgumentError("tool_choice is not a valid JSON object or string");
        }
    }

    bool jsonChanged = false;
    if (tool_choice == "auto")  // for now, with auto choice we don't need to do anything
        return absl::OkStatus();
    if (tool_choice == "none") {
        // remove tools from the request
        doc.RemoveMember("tools");
        jsonChanged = true;
    }
    auto it = doc.FindMember("tools");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsArray())
            return absl::InvalidArgumentError("Tools are not an array");
        for (size_t i = 0; i < it->value.GetArray().Size();) {
            auto& obj = it->value.GetArray()[i];
            if (!obj.IsObject())
                return absl::InvalidArgumentError("Tool is not a JSON object");
            auto functionIt = obj.FindMember("function");
            if (functionIt != obj.MemberEnd() && functionIt->value.IsObject()) {
                auto nameIt = functionIt->value.GetObject().FindMember("name");
                if (nameIt != functionIt->value.GetObject().MemberEnd() && nameIt->value.IsString()) {
                    std::string functionName = nameIt->value.GetString();
                    if (tool_choice != functionName) {
                        it->value.Erase(&obj);
                        jsonChanged = true;
                    } else {
                        i++;
                    }
                } else {
                    return absl::InvalidArgumentError("Function object does not contain a valid name field");
                }
            } else {
                return absl::InvalidArgumentError("Function is not a valid JSON object");
            }
            // Add new tool to tools list - TBD
        }
    }
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    return absl::OkStatus();
}

const std::string& OpenAIChatCompletionsHandler::getProcessedJson() const {
    return request.processedJson;
}
const ImageHistory& OpenAIChatCompletionsHandler::getImageHistory() const {
    return request.imageHistory;
}

ov::genai::ChatHistory& OpenAIChatCompletionsHandler::getChatHistory() {
    return request.chatHistory;
}

std::optional<int> OpenAIChatCompletionsHandler::getMaxTokens() const {
    return request.maxTokens;
}

absl::Status OpenAIChatCompletionsHandler::parseChatCompletionsPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath) {
    // messages: [{role: content}, {role: content}, ...]; required
    auto status = parseMessages(allowedLocalMediaPath);
    if (status != absl::OkStatus()) {
        return status;
    }
    status = parseTools();
    if (status != absl::OkStatus()) {
        return status;
    }
    // logprobs: bool; optional - defaults to false
    auto it = doc.FindMember("logprobs");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("logprobs accepts values true or false");
        request.logprobschat = it->value.GetBool();
    }
    if (request.logprobschat && request.stream) {
        return absl::InvalidArgumentError("logprobs are not supported in streaming mode.");
    }
    // max_completion_tokens: uint; optional
    it = doc.FindMember("max_completion_tokens");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_completion_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_completion_tokens is not an unsigned integer");
        }
        if (maxTokensLimit.has_value() && it->value.GetUint() > maxTokensLimit.value())
            return absl::InvalidArgumentError(absl::StrCat("max_completion_tokens exceeds limit provided in graph config: ", maxTokensLimit.value()));
        request.maxTokens = it->value.GetUint();
    }
    // specific part of max_tokens validation due to echo dependency
    if (request.maxTokens == 0) {
        return absl::InvalidArgumentError("max_tokens value should be greater than 0");
    }

    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseCommonPart(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength) {
    OVMS_PROFILE_FUNCTION();
    // stream: bool; optional
    if (!doc.IsObject())
        return absl::InvalidArgumentError("Received json is not an object");
    auto it = doc.FindMember("stream");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("Stream is not bool");
        request.stream = it->value.GetBool();
    }

    it = doc.FindMember("stream_options");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!request.stream)
            return absl::InvalidArgumentError("stream_options provided, but stream not set to true");
        if (!it->value.IsObject())
            return absl::InvalidArgumentError("stream_options is not an object");
        auto streamOptionsObj = it->value.GetObject();

        size_t streamOptionsFound = 0;
        it = streamOptionsObj.FindMember("include_usage");
        if (it != streamOptionsObj.MemberEnd()) {
            if (!it->value.IsBool())
                return absl::InvalidArgumentError("stream_options.include_usage is not a boolean");
            request.streamOptions.includeUsage = it->value.GetBool();
            streamOptionsFound++;
        }

        if (streamOptionsObj.MemberCount() > streamOptionsFound) {
            return absl::InvalidArgumentError("Found unexpected stream options. Properties accepted in stream_options: include_usage");
        }
    }

    // model: string; required
    it = doc.FindMember("model");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsString())
            return absl::InvalidArgumentError("model is not a string");
        request.model = it->value.GetString();
    } else {
        return absl::InvalidArgumentError("model missing in request");
    }

    // ignore_eos: bool; optional - defaults to false
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("ignore_eos");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("ignore_eos accepts values true or false");
        request.ignoreEOS = it->value.GetBool();
    }

    // max_tokens: uint; optional
    // Common part checked here, specific parts are checked in parseCompletionsPart and parseChatCompletionsPart
    // TODO: Deprecated - this will need to be removed in the future
    it = doc.FindMember("max_tokens");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
        }
        if (maxTokensLimit.has_value() && !(it->value.GetUint() < maxTokensLimit.value()))
            return absl::InvalidArgumentError(absl::StrCat("max_tokens exceeds limit provided in graph config: ", maxTokensLimit.value()));
        request.maxTokens = it->value.GetUint();
    } else {
        if (maxTokensLimit.has_value()) {
            request.maxTokens = maxTokensLimit.value();
        }
    }

    // frequency_penalty: float; optional - defaults to 0
    it = doc.FindMember("frequency_penalty");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("frequency_penalty is not a valid number");
        request.frequencyPenalty = it->value.GetDouble();
        if (request.frequencyPenalty < -2.0f || request.frequencyPenalty > 2.0f)
            return absl::InvalidArgumentError("frequency_penalty out of range(-2.0, 2.0)");
    }

    // presence_penalty: float; optional - defaults to 0
    it = doc.FindMember("presence_penalty");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("presence_penalty is not a valid number");
        request.presencePenalty = it->value.GetDouble();
        if (request.presencePenalty < -2.0f || request.presencePenalty > 2.0f)
            return absl::InvalidArgumentError("presence_penalty out of range(-2.0, 2.0)");
    }

    // repetition_penalty: float; optional - defaults to 1.0
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("repetition_penalty");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("repetition_penalty is not a valid number");
        request.repetitionPenalty = it->value.GetDouble();
    }

    // length_penalty: float; optional - defaults to 1.0
    // Extension, unsupported by OpenAI API however supported by vLLM and CB lib
    it = doc.FindMember("length_penalty");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("length_penalty is not a valid number");
        request.lengthPenalty = it->value.GetDouble();
    }

    // temperature: float; optional - defaults to 1.0
    it = doc.FindMember("temperature");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("temperature is not a valid number");
        request.temperature = it->value.GetDouble();
        if (request.temperature < 0.0f || request.temperature > 2.0f)
            return absl::InvalidArgumentError("temperature out of range(0.0, 2.0)");
    }

    // top_p: float; optional - defaults to 1
    it = doc.FindMember("top_p");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("top_p is not a valid number");
        request.topP = it->value.GetDouble();
        if (request.topP < 0.0f || request.topP > 1.0f)
            return absl::InvalidArgumentError("top_p out of range(0.0, 1.0)");
    }

    // top_k: int; optional - defaults to 0
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("top_k");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsInt())
            return absl::InvalidArgumentError("top_k is not an integer");
        request.topK = it->value.GetInt();
    }

    // seed: int; optional - defaults to 0 (not set)
    it = doc.FindMember("seed");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("seed is not an unsigned integer");
        request.seed = it->value.GetUint();
    }

    // stop: string or array; optional - defaults to null (not set)
    it = doc.FindMember("stop");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (it->value.IsString()) {
            request.stop = std::set<std::string>{it->value.GetString()};
        } else if (it->value.IsArray()) {
            auto stopArray = it->value.GetArray();
            if (stopArray.Size() > DEFAULT_MAX_STOP_WORDS) {
                std::stringstream ss;
                ss << "stop array must have no more than " << DEFAULT_MAX_STOP_WORDS << " strings";
                return absl::InvalidArgumentError(ss.str());
            }
            if (!stopArray.Empty()) {
                request.stop = std::set<std::string>{};
                for (size_t i = 0; i < stopArray.Size(); i++) {
                    const auto& element = stopArray[i];
                    if (!element.IsString())
                        return absl::InvalidArgumentError("stop array contains non string element");
                    request.stop->insert(element.GetString());
                }
            }
        } else {
            return absl::InvalidArgumentError("stop is not a string or array of strings");
        }
    }

    // include_stop_str_in_output: bool; optional - defaults to false
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib

    // If stream is true, then include stop string in output by default
    if (request.stream) {
        request.includeStopStrInOutput = true;
    }

    it = doc.FindMember("include_stop_str_in_output");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("include_stop_str_in_output accepts values true or false");
        if (!it->value.GetBool() && request.stream)
            return absl::InvalidArgumentError("include_stop_str_in_output cannot be set to false if streaming is used");
        request.includeStopStrInOutput = it->value.GetBool();
    }

    // best_of: int; optional - defaults to 1
    // Extension, unsupported by OpenAI API, however supported by vLLM, supported in CB lib by mapping to group_size param
    it = doc.FindMember("best_of");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("best_of is not an unsigned integer");
        if (it->value.GetUint() == 0)
            return absl::InvalidArgumentError("best_of value should be greater than 0");
        if (!(it->value.GetUint() < bestOfLimit))
            return absl::InvalidArgumentError(absl::StrCat("best_of exceeds limit provided in graph config: ", bestOfLimit));
        if (request.stream)
            return absl::InvalidArgumentError("best_of cannot be used in streaming mode");
        request.bestOf = it->value.GetUint();
    }

    // n: int; optional - defaults to 1
    // Supported by OpenAI API and vLLM, supported in CB lib by mapping to num_return_sequences param
    it = doc.FindMember("n");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("n is not an unsigned integer");
        if (it->value.GetUint() == 0)
            return absl::InvalidArgumentError("n value should be greater than 0");
        size_t bestOf = request.bestOf.has_value() ? request.bestOf.value() : 1;  // 1 is default best_of value
        if (bestOf < it->value.GetUint()) {
            return absl::InvalidArgumentError("n value cannot be greater than best_of");
        }
        request.numReturnSequences = it->value.GetUint();
    }

    // Assisted decoding specific parameters

    auto numAssistantTokensIt = doc.FindMember("num_assistant_tokens");
    auto assistantConfidenceThresholdIt = doc.FindMember("assistant_confidence_threshold");
    auto maxNgramSizeIt = doc.FindMember("max_ngram_size");

    bool numAssistantTokensItHasValue = (numAssistantTokensIt != doc.MemberEnd() && !numAssistantTokensIt->value.IsNull());
    bool assistantConfidenceThresholdItHasValue = (assistantConfidenceThresholdIt != doc.MemberEnd() && !assistantConfidenceThresholdIt->value.IsNull());
    bool maxNgramSizeItHasValue = (maxNgramSizeIt != doc.MemberEnd() && !maxNgramSizeIt->value.IsNull());

    if (numAssistantTokensItHasValue) {
        request.numAssistantTokens = numAssistantTokensIt->value.GetUint();
    }
    if (assistantConfidenceThresholdItHasValue) {
        request.assistantConfidenceThreshold = assistantConfidenceThresholdIt->value.GetDouble();
    }
    if (maxNgramSizeItHasValue) {
        request.maxNgramSize = maxNgramSizeIt->value.GetUint();
    }
    request.maxModelLength = maxModelLength;

    // logit_bias TODO
    // top_logprobs TODO
    // response_format TODO
    // tools TODO
    // tool_choice TODO
    // user TODO
    // function_call TODO (deprecated)
    // functions TODO (deprecated)
    return absl::OkStatus();
}

std::optional<std::string> OpenAIChatCompletionsHandler::getPrompt() const { return request.prompt; }
std::optional<int> OpenAIChatCompletionsHandler::getNumReturnSequences() const { return request.numReturnSequences; }
StreamOptions OpenAIChatCompletionsHandler::getStreamOptions() const { return request.streamOptions; }

bool OpenAIChatCompletionsHandler::isStream() const { return request.stream; }
std::string OpenAIChatCompletionsHandler::getModel() const { return request.model; }

void OpenAIChatCompletionsHandler::setPromptTokensUsage(size_t promptTokens) {
    usage.promptTokens = promptTokens;
}

void OpenAIChatCompletionsHandler::incrementProcessedTokens(size_t numTokens) {
    processedTokens += numTokens;
    if (!request.echo || processedTokens > usage.promptTokens)
        usage.completionTokens += numTokens;
}

ov::genai::GenerationConfig OpenAIChatCompletionsHandler::createGenerationConfig() const {
    return request.createGenerationConfig();
}

absl::Status OpenAIChatCompletionsHandler::parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength, std::optional<std::string> allowedLocalMediaPath) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);

    if (status != absl::OkStatus())
        return status;

    if (endpoint == Endpoint::COMPLETIONS)
        status = parseCompletionsPart();
    else
        status = parseChatCompletionsPart(maxTokensLimit, allowedLocalMediaPath);

    return status;
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs, int64_t botTokenId) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int index = 0;
    usage.completionTokens = 0;
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generationOutput.generated_ids);
        usage.completionTokens += generationOutput.generated_ids.size();
        if (request.echo)
            usage.completionTokens -= usage.promptTokens;
        // Find botTokenId in generated_ids
        auto botTokenIt = std::find(generationOutput.generated_ids.begin(), generationOutput.generated_ids.end(), botTokenId);
        std::string completeResponse, toolsResponse;
        if (botTokenIt != generationOutput.generated_ids.end()) {
            // Tokens before botTokenId (not including botTokenId)
            completeResponse = tokenizer.decode(std::vector<int64_t>(generationOutput.generated_ids.begin(), botTokenIt));
            // Tokens after botTokenId
            auto afterBotTokenIt = botTokenIt + 1;
            if (afterBotTokenIt != generationOutput.generated_ids.end()) {
            toolsResponse = tokenizer.decode(std::vector<int64_t>(afterBotTokenIt, generationOutput.generated_ids.end()));
            }
        } else {
            // If botTokenId not found, decode all tokens as completeResponse
            completeResponse = tokenizer.decode(generationOutput.generated_ids);
        }

        std::cout << "Complete response: " << completeResponse << std::endl;
        std::cout << "Tools response: " << toolsResponse << std::endl;
        
        writer.StartObject();  // {
        // finish_reason: string;
        // "stop" => natural stop point due to stopping criteria
        // "length" => due to reaching max_tokens parameter
        writer.String("finish_reason");
        switch (generationOutput.finish_reason) {
        case ov::genai::GenerationFinishReason::STOP:
            writer.String("stop");
            break;
        case ov::genai::GenerationFinishReason::LENGTH:
            writer.String("length");
            break;
        default:
            writer.Null();
        }
        // index: integer; Choice index, only n=1 supported anyway
        writer.String("index");
        writer.Int(index++);
        // logprobs: object/null; Log probability information for the choice. TODO
        writer.String("logprobs");
        if (this->request.logprobschat || this->request.logprobs) {
            if (endpoint == Endpoint::CHAT_COMPLETIONS) {
                writer.StartObject();  // {
                writer.String("content");
                writer.StartArray();  // [

                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    writer.StartObject();  // {

                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String("token");
                    writer.String(token.c_str());

                    float logprob = generationOutput.generated_log_probs[i];
                    writer.String("logprob");
                    writeLogprob(writer, logprob);
                    writer.String("bytes");
                    writer.StartArray();  // [
                    // Assuming tokenizer returned UTF-8 encoded string
                    const unsigned char* tokenBytes = reinterpret_cast<const unsigned char*>(token.c_str());
                    for (int j = 0; tokenBytes[j] != 0; j++)
                        writer.Int(tokenBytes[j]);
                    writer.EndArray();  // ]

                    // top_logprobs are currently hardcoded to return empty array to comply with the API
                    // for full support significant changes on GenAI side are required
                    writer.String("top_logprobs");
                    writer.StartArray();  // [
                                          /*                  
                    Commented out due to supported only top_logprobs 1
                    writer.StartObject();  // {

                    writer.String("token");
                    writer.String(token.c_str());

                    writer.String("logprob");
                    writeLogprob(writer, logprob);
                    writer.String("bytes");
                    writer.StartArray();  // [
                    for (int j = 0; tokenBytes[j] != 0; j++)
                        writer.Int(tokenBytes[j]);
                    writer.EndArray();  // ]

                    writer.EndObject();  // } */
                    writer.EndArray();    // ]

                    writer.EndObject();  // }
                }
                writer.EndArray();   // ]
                writer.EndObject();  // }
            }
            if (endpoint == Endpoint::COMPLETIONS) {
                writer.StartObject();  // {
                writer.String("tokens");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String(token.c_str());
                }
                writer.EndArray();  // ]

                writer.String("token_logprobs");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    float logprob = generationOutput.generated_log_probs[i];
                    writeLogprob(writer, logprob);
                }
                writer.EndArray();  // ]

                writer.String("top_logprobs");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    writer.StartObject();  // {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String(token.c_str());
                    float logprob = generationOutput.generated_log_probs[i];
                    writeLogprob(writer, logprob);
                    writer.EndObject();  // }
                }
                writer.EndArray();  // ]

                writer.String("text_offset");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    if (i == 0) {
                        writer.Int(0);
                    } else {
                        std::string text_before_token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids.begin(), generationOutput.generated_ids.begin() + i}));
                        writer.Uint(text_before_token.size());
                    }
                }
                writer.EndArray();   // ]
                writer.EndObject();  // }
            }
        } else {
            writer.Null();
        }
        // message: object
        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            writer.String("message");
            writer.StartObject();  // {
            // content: string; Actual content of the text produced
            writer.String("content");
            writer.String(completeResponse.c_str());
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String("role");
            writer.String("assistant");  // TODO - hardcoded

            // Tools output serialization
            if (!toolsResponse.empty()) {
                // Assuming there's only one tool call in the response
                rapidjson::Document toolsDoc;
                toolsDoc.Parse(toolsResponse.c_str());
                if (toolsDoc.HasParseError()) {
                    SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to parse toolsResponse as JSON");
                    return "";
                }

                std::string toolName, toolParameters;

                if (toolsDoc.HasMember("name") && toolsDoc["name"].IsString()) {
                    toolName = toolsDoc["name"].GetString();
                }

                if (toolsDoc.HasMember("parameters") && toolsDoc["parameters"].IsObject()) {
                    rapidjson::StringBuffer sb;
                    rapidjson::Writer<rapidjson::StringBuffer> toolWriter(sb);
                    toolsDoc["parameters"].Accept(toolWriter);
                    toolParameters = sb.GetString();
                } else {
                    SPDLOG_LOGGER_ERROR(llm_calculator_logger, "toolsResponse does not contain valid parameters object");
                    return "";
                }

                writer.String("tool_calls");
                writer.StartArray();  // [
                writer.StartObject();  // {
                writer.String("id");
                writer.String("tool_call_0");  // TODO: generate unique ID for each tool call

                writer.String("type");
                writer.String("function");

                writer.String("function");
                writer.StartObject();  // {
                writer.String("name");
                writer.String(toolName.c_str());  // TODO: replace with actual tool name from toolsDoc
                writer.String("arguments");
                writer.String(toolParameters.c_str());  // Assuming toolsResponse is a valid JSON string
                writer.EndObject();  // }
                writer.EndObject();  // }
                writer.EndArray();   // ]
            }
            writer.EndObject();  // }
        } else if (endpoint == Endpoint::COMPLETIONS) {
            writer.String("text");
            writer.String(completeResponse.c_str());
        }

        writer.EndObject();  // }
    }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.model.c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion");
    }

    writer.String("usage");
    writer.StartObject();  // {
    writer.String("prompt_tokens");
    writer.Int(usage.promptTokens);
    writer.String("completion_tokens");
    writer.Int(usage.completionTokens);
    writer.String("total_tokens");
    writer.Int(usage.calculateTotalTokens());
    writer.EndObject();  // }

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const ov::genai::EncodedResults& results) {  // TODO separate common part with function implemented above
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int index = 0;
    usage.completionTokens = 0;
    for (int i = 0; i < results.tokens.size(); i++) {
        const std::vector<int64_t>& tokens = results.tokens[i];
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
        usage.completionTokens += tokens.size();
        if (request.echo)
            usage.completionTokens -= usage.promptTokens;
        std::string completeResponse = tokenizer.decode(tokens);
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Decoded response: {}", completeResponse);
        writer.StartObject();  // {
        writer.String("finish_reason");
        writer.String("stop");
        // index: integer; Choice index, only n=1 supported anyway
        writer.String("index");
        writer.Int(index++);
        // logprobs: object/null; Log probability information for the choice. TODO
        // message: object
        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            writer.String("message");
            writer.StartObject();  // {
            // content: string; Actual content of the text produced
            writer.String("content");
            writer.String(completeResponse.c_str());
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String("role");
            writer.String("assistant");  // TODO - hardcoded
            // TODO: tools_call
            // TODO: function_call (deprecated)
            writer.EndObject();  // }
        } else if (endpoint == Endpoint::COMPLETIONS) {
            writer.String("text");
            writer.String(completeResponse.c_str());
        }

        writer.EndObject();  // }
    }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.model.c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion");
    }

    writer.String("usage");
    writer.StartObject();  // {
    writer.String("prompt_tokens");
    writer.Int(usage.promptTokens);
    writer.String("completion_tokens");
    writer.Int(usage.completionTokens);
    writer.String("total_tokens");
    writer.Int(usage.calculateTotalTokens());
    writer.EndObject();  // }

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const ov::genai::VLMDecodedResults& results, size_t completionTokens) {  // TODO separate common part with function implemented above
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int index = 0;
    usage.completionTokens = completionTokens;
    for (int i = 0; i < results.texts.size(); i++) {
        const std::string& texts = results.texts[i];
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
        writer.StartObject();  // {
        writer.String("finish_reason");
        writer.String("stop");
        // index: integer; Choice index, only n=1 supported anyway
        writer.String("index");
        writer.Int(index++);
        // logprobs: object/null; Log probability information for the choice. TODO
        // message: object
        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            writer.String("message");
            writer.StartObject();  // {
            // content: string; Actual content of the text produced
            writer.String("content");
            writer.String(texts.c_str());
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String("role");
            writer.String("assistant");  // TODO - hardcoded
            // TODO: tools_call
            // TODO: function_call (deprecated)
            writer.EndObject();  // }
        } else if (endpoint == Endpoint::COMPLETIONS) {
            writer.String("text");
            writer.String(texts.c_str());
        }

        writer.EndObject();  // }
    }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.model.c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion");
    }

    writer.String("usage");
    writer.StartObject();  // {
    writer.String("prompt_tokens");
    writer.Int(usage.promptTokens);
    writer.String("completion_tokens");
    writer.Int(usage.completionTokens);
    writer.String("total_tokens");
    writer.Int(usage.calculateTotalTokens());
    writer.EndObject();  // }

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();   // [
    writer.StartObject();  // {
    // finish_reason: string or null; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)/null
    // "stop" => natural stop point due to stopping criteria
    // "length" => due to reaching max_tokens parameter
    // "content_filter" => when produced restricted output (not supported)
    // "tool_calls" => generation stopped and waiting for tool output (not supported)
    // "function_call" => deprecated
    // null - natural scenario when the generation has not completed yet
    writer.String("finish_reason");
    switch (finishReason) {
    case ov::genai::GenerationFinishReason::STOP:
        writer.String("stop");
        break;
    case ov::genai::GenerationFinishReason::LENGTH:
        writer.String("length");
        break;
    default:
        writer.Null();
    }
    // index: integer; Choice index, only n=1 supported anyway
    writer.String("index");
    writer.Int(0);
    // logprobs: object/null; Log probability information for the choice. TODO
    writer.String("logprobs");
    writer.Null();
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("delta");
        writer.StartObject();  // {
        writer.String("content");
        // writer.String("role");
        // writer.String("assistant");
        // role: string; Role of the text producer
        writer.String(chunkResponse.c_str());
        writer.EndObject();  // }
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("text");
        writer.String(chunkResponse.c_str());
    }
    // TODO: tools_call
    // TODO: function_call (deprecated)
    writer.EndObject();  // }
    writer.EndArray();   // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.model.c_str());

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion.chunk");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion.chunk");
    }

    if (request.streamOptions.includeUsage) {
        writer.String("usage");
        writer.Null();
    }

    // TODO
    // id: string; A unique identifier for the chat completion. Each chunk has the same ID.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeStreamingUsageChunk() {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    writer.String("choices");
    writer.StartArray();  // [
    writer.EndArray();    // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.model.c_str());

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion.chunk");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion.chunk");
    }

    writer.String("usage");
    writer.StartObject();  // {
    writer.String("prompt_tokens");
    writer.Int(usage.promptTokens);
    writer.String("completion_tokens");
    writer.Int(usage.completionTokens);
    writer.String("total_tokens");
    writer.Int(usage.calculateTotalTokens());
    writer.EndObject();  // }

    writer.EndObject();  // }
    return buffer.GetString();
}

void OpenAIChatCompletionsHandler::writeLogprob(Writer<StringBuffer>& writer, float logprob) {
    // genai returns logaritm of probability per token which should be in the range of -inf-0
    // other values could be potentially invalid and should be treated as such
    if (logprob <= 0.0)
        writer.Double(logprob);
    else
        writer.Null();
}
}  // namespace ovms
