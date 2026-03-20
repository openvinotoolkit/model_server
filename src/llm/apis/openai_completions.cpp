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
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include <set>
#include <string.h>

#include <fmt/ranges.h>

#include "openai_json_response.hpp"

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "src/filesystem/filesystem.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include <curl/curl.h>
#include <regex>
#include "../../image_conversion.hpp"  // TODO: Rename to stbi_conversions?

using namespace rapidjson;

namespace ovms {

constexpr size_t DEFAULT_MAX_STOP_WORDS = 16;  // same as deep-seek

namespace {

ov::genai::JsonContainer rapidJsonValueToJsonContainer(const rapidjson::Value& value) {
    if (value.IsNull()) {
        return ov::genai::JsonContainer(nullptr);
    }
    if (value.IsBool()) {
        return ov::genai::JsonContainer(value.GetBool());
    }
    if (value.IsInt()) {
        return ov::genai::JsonContainer(value.GetInt());
    }
    if (value.IsUint()) {
        return ov::genai::JsonContainer(static_cast<int64_t>(value.GetUint()));
    }
    if (value.IsInt64()) {
        return ov::genai::JsonContainer(value.GetInt64());
    }
    if (value.IsUint64()) {
        auto uintValue = value.GetUint64();
        if (uintValue <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return ov::genai::JsonContainer(static_cast<int64_t>(uintValue));
        }
        return ov::genai::JsonContainer(static_cast<double>(uintValue));
    }
    if (value.IsDouble()) {
        return ov::genai::JsonContainer(value.GetDouble());
    }
    if (value.IsString()) {
        return ov::genai::JsonContainer(std::string(value.GetString(), value.GetStringLength()));
    }
    if (value.IsArray()) {
        ov::genai::JsonContainer arrayContainer = ov::genai::JsonContainer::array();
        for (const auto& item : value.GetArray()) {
            arrayContainer.push_back(rapidJsonValueToJsonContainer(item));
        }
        return arrayContainer;
    }
    if (value.IsObject()) {
        ov::genai::JsonContainer objectContainer = ov::genai::JsonContainer::object();
        for (auto member = value.MemberBegin(); member != value.MemberEnd(); ++member) {
            const std::string key(member->name.GetString(), member->name.GetStringLength());
            objectContainer[key] = rapidJsonValueToJsonContainer(member->value);
        }
        return objectContainer;
    }
    throw std::invalid_argument("Unsupported JSON value type");
}

std::string serializeResponsesEvent(const std::function<void(Writer<StringBuffer>&)>& eventSerializer) {
    StringBuffer eventBuffer;
    Writer<StringBuffer> eventWriter(eventBuffer);
    eventSerializer(eventWriter);
    return std::string(eventBuffer.GetString());
}

void serializeNotSupportedNullField(Writer<StringBuffer>& writer, const char* fieldName) {
    writer.String(fieldName);
    writer.Null();
}

void serializeNotSupportedZeroField(Writer<StringBuffer>& writer, const char* fieldName) {
    writer.String(fieldName);
    writer.Uint64(0);
}

void serializeNotSupportedEmptyArrayField(Writer<StringBuffer>& writer, const char* fieldName) {
    writer.String(fieldName);
    writer.StartArray();
    writer.EndArray();
}

}  // namespace

void OpenAIChatCompletionsHandler::serializeResponsesToolChoice(Writer<StringBuffer>& writer) const {
    writer.String("tool_choice");
    if (request.toolChoice.empty()) {
        writer.String("auto");
    } else if (request.toolChoice == "auto" || request.toolChoice == "none" || request.toolChoice == "required") {
        writer.String(request.toolChoice.c_str());
    } else {
        writer.StartObject();
        writer.String("type");
        writer.String("function");
        writer.String("name");
        writer.String(request.toolChoice.c_str());
        writer.EndObject();
    }
}

void OpenAIChatCompletionsHandler::serializeResponsesTools(Writer<StringBuffer>& writer) const {
    writer.String("tools");
    writer.StartArray();
    for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap) {
        writer.StartObject();
        writer.String("type");
        writer.String("function");
        writer.String("name");
        writer.String(toolName.c_str());
        writer.String("parameters");
        writer.RawValue(toolSchemaWrapper.stringRepr.c_str(), toolSchemaWrapper.stringRepr.size(), rapidjson::kObjectType);
        writer.EndObject();
    }
    writer.EndArray();
}

void OpenAIChatCompletionsHandler::serializeResponsesResponseObject(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
    const char* status, const std::string& fullOutputText, bool includeUsage,
    const char* incompleteReason, const char* errorMessage, const char* errorCode) const {
    writer.StartObject();
    writer.String("id");
    writer.String(responseId.c_str());
    writer.String("object");
    writer.String("response");
    writer.String("created_at");
    writer.Int64(createdAt);
    if (std::string(status) == "completed") {
        const auto completedAt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        writer.String("completed_at");
        writer.Int64(completedAt);
    }
    if (incompleteReason != nullptr) {
        writer.String("incomplete_details");
        writer.StartObject();
        writer.String("reason");
        writer.String(incompleteReason);
        writer.EndObject();
    }
    writer.String("error");
    if (errorMessage != nullptr) {
        writer.StartObject();
        writer.String("code");
        writer.String(errorCode != nullptr ? errorCode : "server_error");
        writer.String("message");
        writer.String(errorMessage);
        writer.EndObject();
    } else {
        writer.Null();
    }
    writer.String("model");
    writer.String(request.model.c_str());
    writer.String("status");
    writer.String(status);

    writer.String("parallel_tool_calls");
    writer.Bool(false);
    serializeNotSupportedNullField(writer, "previous_response_id");
    serializeNotSupportedNullField(writer, "reasoning");
    writer.String("store");
    writer.Bool(true);
    writer.String("temperature");
    if (request.temperature.has_value()) {
        writer.Double(static_cast<double>(request.temperature.value()));
    } else {
        writer.Double(1.0);
    }
    writer.String("text");
    writer.StartObject();
    writer.String("format");
    writer.StartObject();
    writer.String("type");
    writer.String("text");
    writer.EndObject();
    writer.EndObject();
    serializeResponsesToolChoice(writer);
    serializeResponsesTools(writer);
    writer.String("top_p");
    if (request.topP.has_value()) {
        writer.Double(static_cast<double>(request.topP.value()));
    } else {
        writer.Double(1.0);
    }
    writer.String("truncation");
    writer.String("disabled");
    serializeNotSupportedNullField(writer, "user");
    writer.String("metadata");
    writer.StartObject();
    writer.EndObject();

    if (request.maxTokens.has_value()) {
        writer.String("max_output_tokens");
        writer.Uint64(static_cast<uint64_t>(request.maxTokens.value()));
    }

    writer.String("output");
    writer.StartArray();
    if (!fullOutputText.empty()) {
        writer.StartObject();
        writer.String("id");
        writer.String("msg-0");
        writer.String("type");
        writer.String("message");
        writer.String("role");
        writer.String("assistant");
        writer.String("status");
        if (std::string(status) == "completed") {
            writer.String("completed");
        } else if (std::string(status) == "incomplete") {
            writer.String("incomplete");
        } else {
            writer.String("in_progress");
        }
        writer.String("content");
        writer.StartArray();
        serializeResponsesPart(writer, fullOutputText);
        writer.EndArray();
        writer.EndObject();
    }
    writer.EndArray();

    if (includeUsage) {
        writer.String("usage");
        writer.StartObject();
        writer.String("input_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.promptTokens));
        writer.String("input_tokens_details");
        writer.StartObject();
        serializeNotSupportedZeroField(writer, "cached_tokens");
        writer.EndObject();
        writer.String("output_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.completionTokens));
        writer.String("output_tokens_details");
        writer.StartObject();
        serializeNotSupportedZeroField(writer, "reasoning_tokens");
        writer.EndObject();
        writer.String("total_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.calculateTotalTokens()));
        writer.EndObject();
    }

    writer.EndObject();
}

void OpenAIChatCompletionsHandler::serializeResponsesOutputItem(Writer<StringBuffer>& writer, const std::string& outputItemId,
    const std::string& text, const char* status, bool withContent) {
    writer.StartObject();
    writer.String("id");
    writer.String(outputItemId.c_str());
    writer.String("type");
    writer.String("message");
    writer.String("role");
    writer.String("assistant");
    writer.String("status");
    writer.String(status);
    writer.String("content");
    writer.StartArray();
    if (withContent) {
        serializeResponsesPart(writer, text);
    }
    writer.EndArray();
    writer.EndObject();
}

void OpenAIChatCompletionsHandler::serializeResponsesPart(Writer<StringBuffer>& writer, const std::string& text) {
    writer.StartObject();
    writer.String("type");
    writer.String("output_text");
    writer.String("text");
    writer.String(text.c_str());
    writer.String("annotations");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
}

std::string OpenAIChatCompletionsHandler::serializeResponsesUnaryResponse(const std::vector<ParsedOutput>& parsedOutputs,
    ov::genai::GenerationFinishReason finishReason) const {
    const bool isIncomplete = (finishReason == ov::genai::GenerationFinishReason::LENGTH);
    const char* responseStatus = isIncomplete ? "incomplete" : "completed";
    const auto createdAt = std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count();
    const auto completedAt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();
    writer.String("id");
    writer.String(responseId.c_str());
    writer.String("object");
    writer.String("response");
    writer.String("created_at");
    writer.Int64(createdAt);
    if (!isIncomplete) {
        writer.String("completed_at");
        writer.Int64(completedAt);
    }
    if (isIncomplete) {
        writer.String("incomplete_details");
        writer.StartObject();
        writer.String("reason");
        writer.String("max_tokens");
        writer.EndObject();
    }
    serializeNotSupportedNullField(writer, "error");
    writer.String("model");
    writer.String(request.model.c_str());
    writer.String("status");
    writer.String(responseStatus);

    writer.String("parallel_tool_calls");
    writer.Bool(false);
    serializeNotSupportedNullField(writer, "previous_response_id");
    serializeNotSupportedNullField(writer, "reasoning");
    writer.String("store");
    writer.Bool(true);
    writer.String("temperature");
    if (request.temperature.has_value()) {
        writer.Double(static_cast<double>(request.temperature.value()));
    } else {
        writer.Double(1.0);
    }
    writer.String("text");
    writer.StartObject();
    writer.String("format");
    writer.StartObject();
    writer.String("type");
    writer.String("text");
    writer.EndObject();
    writer.EndObject();
    serializeResponsesToolChoice(writer);
    serializeResponsesTools(writer);
    writer.String("top_p");
    if (request.topP.has_value()) {
        writer.Double(static_cast<double>(request.topP.value()));
    } else {
        writer.Double(1.0);
    }
    writer.String("truncation");
    writer.String("disabled");
    serializeNotSupportedNullField(writer, "user");
    writer.String("metadata");
    writer.StartObject();
    writer.EndObject();

    if (request.maxTokens.has_value()) {
        writer.String("max_output_tokens");
        writer.Uint64(static_cast<uint64_t>(request.maxTokens.value()));
    }

    writer.String("output");
    writer.StartArray();
    int outputIndex = 0;
    for (const auto& parsedOutput : parsedOutputs) {
        const std::string outputId = "msg-" + std::to_string(outputIndex++);

        writer.StartObject();
        writer.String("id");
        writer.String(outputId.c_str());
        writer.String("type");
        writer.String("message");
        writer.String("role");
        writer.String("assistant");
        writer.String("status");
        writer.String(responseStatus);
        writer.String("content");
        writer.StartArray();
        serializeResponsesPart(writer, parsedOutput.content);
        writer.EndArray();
        writer.EndObject();
    }
    writer.EndArray();

    writer.String("usage");
    writer.StartObject();
    writer.String("input_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.promptTokens));
    writer.String("input_tokens_details");
    writer.StartObject();
    serializeNotSupportedZeroField(writer, "cached_tokens");
    writer.EndObject();
    writer.String("output_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.completionTokens));
    writer.String("output_tokens_details");
    writer.StartObject();
    serializeNotSupportedZeroField(writer, "reasoning_tokens");
    writer.EndObject();
    writer.String("total_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.calculateTotalTokens()));
    writer.EndObject();

    writer.EndObject();

    return buffer.GetString();
}

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
    if (!curl_handle) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to initialize curl handle");
        return absl::InternalError("Image downloading failed");
    }
    auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curl_handle, curl_easy_cleanup);

    auto status = curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, appendChunkCallback))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &image))
    CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA))
    const char* envAllowRedirects = std::getenv("OVMS_MEDIA_URL_ALLOW_REDIRECTS");
    if (envAllowRedirects != nullptr && (std::strcmp(envAllowRedirects, "1") == 0)) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "URL redirects allowed");
        CURL_SETOPT(curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L))
    }
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
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Downloading image succeeded, {} bytes retrieved", image.size());
    }
    return absl::OkStatus();
}

static bool isDomainAllowed(const std::vector<std::string>& allowedDomains, const char* url) {
    if (allowedDomains.size() == 1 && allowedDomains[0] == "all") {
        return true;
    }
    CURLUcode rc;
    CURLU* parsedUrl = curl_url();
    rc = curl_url_set(parsedUrl, CURLUPART_URL, url, 0);
    if (rc) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing url {} failed", url);
        curl_url_cleanup(parsedUrl);
        return false;
    }
    char* host;
    rc = curl_url_get(parsedUrl, CURLUPART_HOST, &host, 0);
    if (rc) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing url {} hostname failed", url);
        curl_url_cleanup(parsedUrl);
        return false;
    }
    bool allowed = false;
    for (const auto& allowedDomain : allowedDomains) {
        if (allowedDomain.compare(host) == 0) {
            allowed = true;
            break;
        }
    }
    curl_free(host);
    curl_url_cleanup(parsedUrl);
    return allowed;
}

absl::Status OpenAIChatCompletionsHandler::ensureArgumentsInToolCalls(Value& messageObj, bool& jsonChanged) {
    auto& allocator = doc.GetAllocator();
    auto toolCallsIt = messageObj.FindMember("tool_calls");
    if (toolCallsIt != messageObj.MemberEnd() && toolCallsIt->value.IsArray()) {
        const auto& toolCallsArray = toolCallsIt->value.GetArray();
        for (rapidjson::SizeType j = 0; j < toolCallsArray.Size(); ++j) {
            auto& toolCall = toolCallsArray[j];
            if (!toolCall.IsObject()) {
                return absl::InvalidArgumentError("Each tool_call must be an object");
            }
            auto functionIt = toolCall.FindMember("function");
            if (functionIt == toolCall.MemberEnd() || !functionIt->value.IsObject()) {
                return absl::InvalidArgumentError("Each tool_call must have a 'function' object");
            }
            const auto& functionObj = functionIt->value.GetObject();
            if (functionObj.FindMember("arguments") == functionObj.MemberEnd()) {
                // Add "arguments": "{}"
                rapidjson::Value argumentsKey("arguments", allocator);
                rapidjson::Value argumentsValue;
                argumentsValue.SetString("{}", allocator);
                functionIt->value.GetObject().AddMember(argumentsKey, argumentsValue, allocator);
                jsonChanged = true;
            }
        }
    }
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseResponsesInput(std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    auto inputIt = doc.FindMember("input");
    if (inputIt == doc.MemberEnd()) {
        return absl::InvalidArgumentError("input missing in request");
    }

    if (inputIt->value.IsString()) {
        request.prompt = inputIt->value.GetString();
        if (request.prompt.value().empty()) {
            return absl::InvalidArgumentError("input cannot be empty");
        }

        request.chatHistory.push_back({});
        request.chatHistory.last()["role"] = "user";
        request.chatHistory.last()["content"] = request.prompt.value();
    } else if (inputIt->value.IsArray()) {
        if (inputIt->value.GetArray().Size() == 0) {
            return absl::InvalidArgumentError("Messages array cannot be empty");
        }

        for (size_t i = 0; i < inputIt->value.GetArray().Size(); ++i) {
            auto& item = inputIt->value.GetArray()[i];
            if (!item.IsObject()) {
                return absl::InvalidArgumentError("input array items must be objects");
            }

            auto itemObj = item.GetObject();
            auto roleIt = itemObj.FindMember("role");
            if (roleIt == itemObj.MemberEnd() || !roleIt->value.IsString()) {
                return absl::InvalidArgumentError("input item role is missing or invalid");
            }

            request.chatHistory.push_back({});
            request.chatHistory.last()["role"] = roleIt->value.GetString();

            auto contentIt = itemObj.FindMember("content");
            if (contentIt == itemObj.MemberEnd()) {
                return absl::InvalidArgumentError("input item content is missing");
            }

            if (contentIt->value.IsString()) {
                request.chatHistory.last()["content"] = contentIt->value.GetString();
                continue;
            }

            if (!contentIt->value.IsArray()) {
                return absl::InvalidArgumentError("input item content must be a string or array");
            }
            if (contentIt->value.GetArray().Size() == 0) {
                return absl::InvalidArgumentError("Invalid message structure - content array is empty");
            }

            std::string contentText;
            for (auto& contentItem : contentIt->value.GetArray()) {
                if (!contentItem.IsObject()) {
                    return absl::InvalidArgumentError("input content items must be objects");
                }
                auto contentObj = contentItem.GetObject();
                auto typeIt = contentObj.FindMember("type");
                if (typeIt == contentObj.MemberEnd() || !typeIt->value.IsString()) {
                    return absl::InvalidArgumentError("input content item type is missing or invalid");
                }

                const std::string type = typeIt->value.GetString();
                if (type == "input_text") {
                    auto textIt = contentObj.FindMember("text");
                    if (textIt == contentObj.MemberEnd() || !textIt->value.IsString()) {
                        return absl::InvalidArgumentError("input_text requires a valid text field");
                    }
                    contentText = textIt->value.GetString();
                } else if (type == "input_image") {
                    std::string imageUrl;
                    auto imageUrlIt = contentObj.FindMember("image_url");
                    if (imageUrlIt == contentObj.MemberEnd()) {
                        return absl::InvalidArgumentError("input_image requires image_url field");
                    }
                    if (imageUrlIt->value.IsString()) {
                        imageUrl = imageUrlIt->value.GetString();
                    } else if (imageUrlIt->value.IsObject()) {
                        auto imageUrlObj = imageUrlIt->value.GetObject();
                        auto urlIt = imageUrlObj.FindMember("url");
                        if (urlIt == imageUrlObj.MemberEnd() || !urlIt->value.IsString()) {
                            return absl::InvalidArgumentError("input_image.image_url.url is missing or invalid");
                        }
                        imageUrl = urlIt->value.GetString();
                    } else {
                        return absl::InvalidArgumentError("input_image.image_url must be a string or object");
                    }

                    std::string pattern = "base64,";
                    std::size_t pos = imageUrl.find(pattern);
                    std::string decoded;
                    ov::Tensor tensor;
                    if (pos != std::string::npos) {
                        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
                        size_t offset = pos + pattern.length();
                        if (!absl::Base64Unescape(std::string_view(imageUrl.data() + offset, imageUrl.size() - offset), &decoded)) {
                            return absl::InvalidArgumentError("Invalid base64 string in request");
                        }
                        try {
                            tensor = loadImageStbiFromMemory(decoded);
                        } catch (std::runtime_error& e) {
                            std::stringstream ss;
                            ss << "Image parsing failed: " << e.what();
                            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                            return absl::InvalidArgumentError(ss.str());
                        }
                    } else if (std::regex_match(imageUrl.c_str(), std::regex("^(http|https|ftp|sftp|)://(.*)"))) {
                        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
                        int64_t sizeLimit = 20000000;  // restrict single image size to 20MB
                        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), imageUrl.c_str())) {
                            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
                        }
                        auto status = downloadImage(imageUrl.c_str(), decoded, sizeLimit);
                        if (status != absl::OkStatus()) {
                            return status;
                        }
                        try {
                            tensor = loadImageStbiFromMemory(decoded);
                        } catch (std::runtime_error& e) {
                            std::stringstream ss;
                            ss << "Image parsing failed: " << e.what();
                            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                            return absl::InvalidArgumentError("Image parsing failed");
                        }
                    } else {
                        if (!allowedLocalMediaPath.has_value()) {
                            return absl::InvalidArgumentError("Loading images from local filesystem is disabled.");
                        }
                        if (FileSystem::isPathEscaped(imageUrl)) {
                            std::stringstream ss;
                            ss << "Path " << imageUrl.c_str() << " escape with .. is forbidden.";
                            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                            return absl::InvalidArgumentError(ss.str());
                        }
                        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from local filesystem");
                        const auto firstMissmatch = std::mismatch(imageUrl.begin(), imageUrl.end(), allowedLocalMediaPath.value().begin(), allowedLocalMediaPath.value().end());
                        if (firstMissmatch.second != allowedLocalMediaPath.value().end()) {
                            return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
                        }
                        try {
                            tensor = loadImageStbiFromFile(imageUrl.c_str());
                        } catch (std::runtime_error& e) {
                            std::stringstream ss;
                            ss << "Image file " << imageUrl.c_str() << " parsing failed: " << e.what();
                            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                            return absl::InvalidArgumentError(ss.str());
                        }
                    }
                    request.imageHistory.push_back({i, tensor});
                } else {
                    return absl::InvalidArgumentError("Unsupported content type. Supported types are input_text and input_image.");
                }
            }

            request.chatHistory.last()["content"] = contentText;
        }
    } else {
        return absl::InvalidArgumentError("input is not a string or array");
    }

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed responses input directly to chat history without mutating request JSON");
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseMessages(std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
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
        // Add new message to chat history with role, content, and tool-related fields (tool_calls, tool_call_id, name)
        // Other values are not stored in chat history, but are still present in the request object
        request.chatHistory.push_back({});
        for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
            if (!member->name.IsString())
                return absl::InvalidArgumentError("Invalid message structure");
            std::string memberName = member->name.GetString();
            if (member->value.IsString() && (memberName == "role" || memberName == "content")) {
                // Add new field to the last message in history
                request.chatHistory.last()[memberName] = member->value.GetString();
                continue;
            }
            // Handle tool_call_id and name string fields for tool calling
            if (member->value.IsString() && (memberName == "tool_call_id" || memberName == "name")) {
                request.chatHistory.last()[memberName] = member->value.GetString();
                continue;
            }
            // Handle null content (common in assistant messages with tool_calls)
            if (memberName == "content" && member->value.IsNull()) {
                request.chatHistory.last()[memberName] = ov::genai::JsonContainer(nullptr);
                continue;
            }
            // Handle tool_calls array for function calling
            if (memberName == "tool_calls" && member->value.IsArray()) {
                request.chatHistory.last()[memberName] = rapidJsonValueToJsonContainer(member->value);
                continue;
            }
            if (memberName == "content" && member->value.IsArray()) {
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
                                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                                return absl::InvalidArgumentError(ss.str());
                            }
                        } else if (std::regex_match(url.c_str(), std::regex("^(http|https|ftp|sftp|)://(.*)"))) {
                            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
                            int64_t sizeLimit = 20000000;  // restrict single image size to 20MB
                            if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), url.c_str())) {
                                return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
                            }
                            auto status = downloadImage(url.c_str(), decoded, sizeLimit);
                            if (status != absl::OkStatus()) {
                                return status;
                            }
                            try {
                                tensor = loadImageStbiFromMemory(decoded);
                            } catch (std::runtime_error& e) {
                                std::stringstream ss;
                                ss << "Image parsing failed: " << e.what();
                                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                                return absl::InvalidArgumentError("Image parsing failed");
                            }

                        } else {
                            if (!allowedLocalMediaPath.has_value()) {
                                return absl::InvalidArgumentError("Loading images from local filesystem is disabled.");
                            }
                            if (FileSystem::isPathEscaped(url)) {
                                std::stringstream ss;
                                ss << "Path " << url.c_str() << " escape with .. is forbidden.";
                                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                                return absl::InvalidArgumentError(ss.str());
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
                                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
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
                    request.chatHistory.last()[member->name.GetString()] = member->value.GetString();
                }
            }
        }
        auto lastMessage = request.chatHistory.last();
        if (!lastMessage.contains("role")) {
            return absl::InvalidArgumentError("Every message must have 'role' field");
        }
        if (!lastMessage.contains("content")) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Message does not have content field which might be an issue for some chat templates. Adding empty content.");
            lastMessage["content"] = "";
            obj.AddMember("content", Value().SetString("", doc.GetAllocator()), doc.GetAllocator());
            jsonChanged = true;
        }
        // If message has tool calls, make sure each tool call has "arguments" field
        auto status = ensureArgumentsInToolCalls(obj, jsonChanged);
        if (status != absl::OkStatus()) {
            return status;
        }
    }
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed messages successfully");
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseTools() {
    auto tool_choice_it = doc.FindMember("tool_choice");
    std::string tool_choice{"auto"};
    if (tool_choice_it != doc.MemberEnd() && !tool_choice_it->value.IsNull()) {
        if (tool_choice_it->value.IsString()) {
            tool_choice = tool_choice_it->value.GetString();
            if (tool_choice != "none" && tool_choice != "auto" && tool_choice != "required")
                return absl::InvalidArgumentError("tool_choice should be either 'none' or 'auto' or 'required'");
        } else if (tool_choice_it->value.IsObject()) {
            auto toolChoiceObj = tool_choice_it->value.GetObject();
            auto tool_choice_functionIt = toolChoiceObj.FindMember("function");
            if (tool_choice_functionIt != toolChoiceObj.MemberEnd() && tool_choice_functionIt->value.IsObject()) {
                auto nameIt = tool_choice_functionIt->value.GetObject().FindMember("name");
                if (nameIt != tool_choice_functionIt->value.GetObject().MemberEnd() && nameIt->value.IsString()) {
                    tool_choice = nameIt->value.GetString();
                } else {
                    return absl::InvalidArgumentError("tool_choice.function.name is not a valid string");
                }
            } else {
                auto typeIt = toolChoiceObj.FindMember("type");
                auto nameIt = toolChoiceObj.FindMember("name");
                if (typeIt != toolChoiceObj.MemberEnd() && typeIt->value.IsString() && std::string(typeIt->value.GetString()) == "function") {
                    if (nameIt == toolChoiceObj.MemberEnd() || !nameIt->value.IsString()) {
                        return absl::InvalidArgumentError("tool_choice.name is not a valid string");
                    }
                    tool_choice = nameIt->value.GetString();
                } else {
                    return absl::InvalidArgumentError("tool_choice.function is not a valid JSON object");
                }
            }
        } else {
            return absl::InvalidArgumentError("tool_choice is not a valid JSON object or string");
        }
    }
    bool jsonChanged = false;
    if (tool_choice == "none") {
        // remove tools from the request
        doc.RemoveMember("tools");
        jsonChanged = true;
    }
    auto it = doc.FindMember("tools");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsArray())
            return absl::InvalidArgumentError("Tools are not an array");
        for (size_t i = 0; i < it->value.GetArray().Size();) {
            auto& obj = it->value.GetArray()[i];
            if (!obj.IsObject())
                return absl::InvalidArgumentError("Tool is not a JSON object");
            rapidjson::Value* functionObj = nullptr;
            rapidjson::Value* parametersValue = nullptr;
            const char* functionNameCStr = nullptr;

            auto functionIt = obj.FindMember("function");
            if (functionIt != obj.MemberEnd()) {
                if (!functionIt->value.IsObject()) {
                    return absl::InvalidArgumentError("Function is not a valid JSON object");
                }
                functionObj = &functionIt->value;
                auto nameIt = functionObj->GetObject().FindMember("name");
                if (nameIt == functionObj->GetObject().MemberEnd() || !nameIt->value.IsString()) {
                    return absl::InvalidArgumentError("Function object does not contain a valid name field");
                }
                functionNameCStr = nameIt->value.GetString();
                auto parametersIt = functionObj->GetObject().FindMember("parameters");
                if (parametersIt != functionObj->GetObject().MemberEnd()) {
                    parametersValue = &parametersIt->value;
                }
            } else {
                auto typeIt = obj.FindMember("type");
                if (typeIt == obj.MemberEnd() || !typeIt->value.IsString()) {
                    return absl::InvalidArgumentError("Tool type is missing or invalid");
                }
                if (std::string(typeIt->value.GetString()) != "function") {
                    return absl::InvalidArgumentError("Only function tools are supported");
                }

                auto nameIt = obj.FindMember("name");
                if (nameIt == obj.MemberEnd() || !nameIt->value.IsString()) {
                    return absl::InvalidArgumentError("Function object does not contain a valid name field");
                }
                functionNameCStr = nameIt->value.GetString();

                auto parametersIt = obj.FindMember("parameters");
                if (parametersIt != obj.MemberEnd()) {
                    parametersValue = &parametersIt->value;
                }
            }

            std::string functionName = functionNameCStr;
            // If tool_choice is set to "auto", we keep all tools
            // If tool_choice is set to a specific function name, we keep only that tool
            if (tool_choice != "auto" && tool_choice != "required" && tool_choice != functionName) {
                it->value.Erase(&obj);
                jsonChanged = true;
                continue;
            }

            i++;
            // If we keep the tool, add tool name and schema to the request
            if (parametersValue != nullptr) {
                if (!parametersValue->IsObject()) {
                    return absl::InvalidArgumentError("Function parameters are not a valid JSON object");
                }
                // now we want to insert to a mapping of
                // tool name -> tool schema representations struct
                // Dump parameters object to string since this is the schema format expected by GenAI
                // Keep the rapidjson::Value object as well to avoid re-parsing in outputParsers
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                parametersValue->Accept(writer);
                std::string parametersStr = buffer.GetString();
                ToolSchemaWrapper schemaReprs{parametersValue, std::move(parametersStr)};
                request.toolNameSchemaMap[functionName] = std::move(schemaReprs);
            }
        }
    } else {
        tool_choice = "none";  // If tools are not provided, set tool_choice to "none"
    }

    request.toolChoice = tool_choice;
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    return absl::OkStatus();
}

absl::StatusOr<std::optional<ov::genai::JsonContainer>> OpenAIChatCompletionsHandler::parseToolsToJsonContainer() {
    auto it = doc.FindMember("tools");
    if (it == doc.MemberEnd() || it->value.IsNull()) {
        return std::nullopt;
    }
    try {
        return rapidJsonValueToJsonContainer(it->value);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Direct tools conversion to JsonContainer failed: {}. Falling back to JSON string conversion.", e.what());
        try {
            rapidjson::StringBuffer toolsBuffer;
            rapidjson::Writer<rapidjson::StringBuffer> toolsWriter(toolsBuffer);
            it->value.Accept(toolsWriter);
            return ov::genai::JsonContainer::from_json_string(toolsBuffer.GetString());
        } catch (const std::exception& fallbackEx) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Fallback tools conversion failed: {}", fallbackEx.what());
            return absl::InvalidArgumentError(absl::StrCat("Invalid tools payload: ", fallbackEx.what()));
        }
    }
}

absl::StatusOr<std::optional<ov::genai::JsonContainer>> OpenAIChatCompletionsHandler::parseChatTemplateKwargsToJsonContainer() {
    auto it = doc.FindMember("chat_template_kwargs");
    if (it == doc.MemberEnd() || it->value.IsNull()) {
        return std::nullopt;
    }
    if (!it->value.IsObject()) {
        return absl::InvalidArgumentError("chat_template_kwargs must be an object");
    }
    try {
        return rapidJsonValueToJsonContainer(it->value);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Direct chat_template_kwargs conversion to JsonContainer failed: {}. Falling back to JSON string conversion.", e.what());
        try {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            it->value.Accept(writer);
            return ov::genai::JsonContainer::from_json_string(buffer.GetString());
        } catch (const std::exception& fallbackEx) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Fallback chat_template_kwargs conversion failed: {}", fallbackEx.what());
            return absl::InvalidArgumentError(absl::StrCat("Invalid chat_template_kwargs payload: ", fallbackEx.what()));
        }
    }
}

const bool OpenAIChatCompletionsHandler::areToolsAvailable() const {
    return !request.toolNameSchemaMap.empty();
}

const OpenAIChatCompletionsRequest& OpenAIChatCompletionsHandler::getRequest() const {
    return request;
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

std::optional<std::string> OpenAIChatCompletionsHandler::getResponseFormat() const {
    return request.responseFormat;
}

std::string convertOpenAIResponseFormatToStructuralTagStringFormat(const rapidjson::Value& openAIFormat) {
    // Build the new object: {"type": "structural_tag", "format": <openAIFormat>}
    // If response_format has {"json_schema": {"schema": {...}}}, flatten it to {"json_schema": {...}}
    rapidjson::Document flatFormatDoc;
    flatFormatDoc.CopyFrom(openAIFormat, flatFormatDoc.GetAllocator());

    if (flatFormatDoc.HasMember("json_schema") && flatFormatDoc["json_schema"].IsObject()) {
        auto& jsonSchema = flatFormatDoc["json_schema"];
        if (jsonSchema.HasMember("schema") && jsonSchema["schema"].IsObject()) {
            // Move all members from "schema" to "json_schema"
            rapidjson::Value schemaObjCopy;
            schemaObjCopy.CopyFrom(jsonSchema["schema"], flatFormatDoc.GetAllocator());  // Make a copy as we will modify jsonSchema
            for (auto itr = schemaObjCopy.MemberBegin(); itr != schemaObjCopy.MemberEnd(); ++itr) {
                rapidjson::Value key;
                key.CopyFrom(itr->name, flatFormatDoc.GetAllocator());
                rapidjson::Value value;
                value.CopyFrom(itr->value, flatFormatDoc.GetAllocator());
                jsonSchema.AddMember(key, value, flatFormatDoc.GetAllocator());
            }
            // Remove the "schema" member
            jsonSchema.RemoveMember("schema");
        }
    }

    // Serialize the flattened response_format object
    rapidjson::StringBuffer formatBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> formatWriter(formatBuffer);
    flatFormatDoc.Accept(formatWriter);

    // Build the new object: {"type": "structural_tag", "format": <flattened>}
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("type");
    writer.String("structural_tag");
    writer.Key("format");
    writer.RawValue(formatBuffer.GetString(), formatBuffer.GetSize(), rapidjson::kObjectType);
    writer.EndObject();
    return buffer.GetString();
}

absl::Status OpenAIChatCompletionsHandler::parseChatCompletionsPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    // messages: [{role: content}, {role: content}, ...]; required
    auto status = parseMessages(allowedLocalMediaPath, allowedMediaDomains);
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

    // parse response_format
    it = doc.FindMember("response_format");
    if (it != doc.MemberEnd()) {
        if (it->value.IsNull())
            return absl::OkStatus();
        if (!it->value.IsObject())
            return absl::InvalidArgumentError("response_format is not an object");
        const rapidjson::Value& responseFormat = it->value;
        request.responseFormat = convertOpenAIResponseFormatToStructuralTagStringFormat(responseFormat);
    }

    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseResponsesPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    // input: string; required
    auto it = doc.FindMember("input");
    if (it == doc.MemberEnd()) {
        return absl::InvalidArgumentError("input missing in request");
    }

    auto messagesStatus = parseResponsesInput(allowedLocalMediaPath, allowedMediaDomains);
    if (!messagesStatus.ok()) {
        return messagesStatus;
    }

#if (PYTHON_DISABLE == 0)
    // Build processedJson with "messages" array from chatHistory so that
    // the Python chat template path (which reads request_json["messages"])
    // can consume Responses API input without a separate code path.
    {
        Document processedDoc;
        processedDoc.SetObject();
        auto& alloc = processedDoc.GetAllocator();

        Value messagesArray(kArrayType);
        for (size_t i = 0; i < request.chatHistory.size(); ++i) {
            Value msgObj(kObjectType);
            auto role = request.chatHistory[i]["role"].as_string();
            if (role.has_value()) {
                msgObj.AddMember("role", Value(role.value().c_str(), alloc), alloc);
            }
            auto content = request.chatHistory[i]["content"].as_string();
            if (content.has_value()) {
                msgObj.AddMember("content", Value(content.value().c_str(), alloc), alloc);
            }
            messagesArray.PushBack(msgObj, alloc);
        }
        processedDoc.AddMember("messages", messagesArray, alloc);

        // Copy tools from original doc if present
        auto toolsIt = doc.FindMember("tools");
        if (toolsIt != doc.MemberEnd() && !toolsIt->value.IsNull()) {
            Value toolsCopy(toolsIt->value, alloc);
            processedDoc.AddMember("tools", toolsCopy, alloc);
        }

        // Copy chat_template_kwargs from original doc if present
        auto kwargsIt = doc.FindMember("chat_template_kwargs");
        if (kwargsIt != doc.MemberEnd() && !kwargsIt->value.IsNull()) {
            Value kwargsCopy(kwargsIt->value, alloc);
            processedDoc.AddMember("chat_template_kwargs", kwargsCopy, alloc);
        }

        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        processedDoc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
#endif
    // logprobs: bool; optional - defaults to false
    it = doc.FindMember("logprobs");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("logprobs accepts values true or false");
        request.logprobschat = it->value.GetBool();
    }
    if (request.logprobschat && request.stream) {
        return absl::InvalidArgumentError("logprobs are not supported in streaming mode.");
    }

    auto toolsStatus = parseTools();
    if (!toolsStatus.ok()) {
        return toolsStatus;
    }

    // max_output_tokens: uint; optional
    // OpenAI Responses API uses this field for output token limit.
    it = doc.FindMember("max_output_tokens");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_output_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_output_tokens is not an unsigned integer");
        }
        if (maxTokensLimit.has_value() && it->value.GetUint() > maxTokensLimit.value())
            return absl::InvalidArgumentError(absl::StrCat("max_output_tokens exceeds limit provided in graph config: ", maxTokensLimit.value()));
        request.maxTokens = it->value.GetUint();
    }

    // specific part of max_output_tokens validation
    if (request.maxTokens == 0) {
        return absl::InvalidArgumentError("max_output_tokens value should be greater than 0");
    }

    // parse response_format
    it = doc.FindMember("response_format");
    if (it != doc.MemberEnd()) {
        if (it->value.IsNull())
            return absl::OkStatus();
        if (!it->value.IsObject())
            return absl::InvalidArgumentError("response_format is not an object");
        const rapidjson::Value& responseFormat = it->value;
        request.responseFormat = convertOpenAIResponseFormatToStructuralTagStringFormat(responseFormat);
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
    // Not applicable for RESPONSES endpoint which uses max_output_tokens instead
    if (endpoint != Endpoint::RESPONSES) {
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
    // Not supported for RESPONSES streaming - output_index is hardcoded to 0
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
    // Not supported for RESPONSES streaming - output_index is hardcoded to 0
    it = doc.FindMember("n");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("n is not an unsigned integer");
        if (it->value.GetUint() == 0)
            return absl::InvalidArgumentError("n value should be greater than 0");
        if (endpoint == Endpoint::RESPONSES && request.stream && it->value.GetUint() > 1)
            return absl::InvalidArgumentError("n greater than 1 is not supported for Responses API streaming");
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
Endpoint OpenAIChatCompletionsHandler::getEndpoint() const { return endpoint; }
std::string OpenAIChatCompletionsHandler::getModel() const { return request.model; }
std::string OpenAIChatCompletionsHandler::getToolChoice() const { return request.toolChoice; }
const std::unique_ptr<OutputParser>& OpenAIChatCompletionsHandler::getOutputParser() const { return outputParser; }

void OpenAIChatCompletionsHandler::setPromptTokensUsage(size_t promptTokens) {
    usage.promptTokens = promptTokens;
}

void OpenAIChatCompletionsHandler::setCompletionTokensUsage(size_t completionTokens) {
    usage.completionTokens = completionTokens;
}

void OpenAIChatCompletionsHandler::incrementProcessedTokens(size_t numTokens) {
    processedTokens += numTokens;
    if (!request.echo || processedTokens > usage.promptTokens)
        usage.completionTokens += numTokens;
}

absl::Status OpenAIChatCompletionsHandler::parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (status != absl::OkStatus())
        return status;
    if (endpoint == Endpoint::COMPLETIONS)
        status = parseCompletionsPart();
    else if (endpoint == Endpoint::RESPONSES)
        status = parseResponsesPart(maxTokensLimit, allowedLocalMediaPath, allowedMediaDomains);
    else
        status = parseChatCompletionsPart(maxTokensLimit, allowedLocalMediaPath, allowedMediaDomains);

    return status;
}

void updateUsage(CompletionUsageStatistics& usage, const std::vector<int64_t>& generatedIds, bool echoPrompt) {
    OVMS_PROFILE_FUNCTION();
    usage.completionTokens += generatedIds.size();
    if (echoPrompt)
        usage.completionTokens -= usage.promptTokens;
}

static std::optional<std::string> mapFinishReason(ov::genai::GenerationFinishReason finishReason, bool hasToolCalls) {
    // GenerationFinishReason::TOOL_CALLS is not available in GenAI yet.
    // Use feature detection based on presence of tool calls as a workaround until GenAI exposes TOOL_CALLS.
    if (hasToolCalls && finishReason == ov::genai::GenerationFinishReason::STOP) {
        return "tool_calls";
    }
    switch (finishReason) {
    case ov::genai::GenerationFinishReason::STOP:
        return "stop";
    case ov::genai::GenerationFinishReason::LENGTH:
        return "length";
    default:
        return std::nullopt;
    }
}

static bool hasToolCallsInStreamingDelta(const rapidjson::Document& delta) {
    if (!delta.HasMember("delta") || !delta["delta"].IsObject()) {
        return false;
    }
    const auto& deltaObj = delta["delta"];
    return deltaObj.HasMember("tool_calls") && deltaObj["tool_calls"].IsArray();
}

ParsedOutput OpenAIChatCompletionsHandler::parseOutputIfNeeded(const std::vector<int64_t>& generatedIds) {
    OVMS_PROFILE_FUNCTION();
    ParsedOutput parsedOutput;
    if (endpoint != Endpoint::CHAT_COMPLETIONS || outputParser == nullptr) {
        parsedOutput.content = this->tokenizer.decode(generatedIds);
    } else {
        parsedOutput = outputParser->parse(generatedIds, this->areToolsAvailable());
    }
    return parsedOutput;
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) {
    OVMS_PROFILE_FUNCTION();
    if (endpoint == Endpoint::RESPONSES) {
        std::vector<ParsedOutput> parsedOutputs;
        usage.completionTokens = 0;
        ov::genai::GenerationFinishReason responsesFinishReason = ov::genai::GenerationFinishReason::STOP;
        for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
            updateUsage(usage, generationOutput.generated_ids, request.echo);
            parsedOutputs.push_back(parseOutputIfNeeded(generationOutput.generated_ids));
            if (generationOutput.finish_reason == ov::genai::GenerationFinishReason::LENGTH) {
                responsesFinishReason = ov::genai::GenerationFinishReason::LENGTH;
            }
        }
        return serializeResponsesUnaryResponse(parsedOutputs, responsesFinishReason);
    }

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    // choices: array of size N, where N is related to n request parameter
    jsonResponse.StartArray("choices");
    int index = 0;
    // Manual usage setup for CB pipelines. For legacy we rely on PerfMetrics object from GenAI `generate` results
    usage.completionTokens = 0;
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generationOutput.generated_ids);

        updateUsage(usage, generationOutput.generated_ids, request.echo);
        ParsedOutput parsedOutput = parseOutputIfNeeded(generationOutput.generated_ids);

        jsonResponse.StartObject();
        // finish_reason: string;
        // "stop" => natural stop point due to stopping criteria
        // "length" => due to reaching max_tokens parameter
        // "tool_calls" => generation stopped due to generated tool calls

        std::optional<std::string> finishReason = mapFinishReason(generationOutput.finish_reason, !parsedOutput.toolCalls.empty());
        if (!finishReason.has_value()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unknown finish reason: {}", static_cast<int>(generationOutput.finish_reason));
        }
        jsonResponse.FinishReason(finishReason.value_or("unknown"));
        // index: integer; Choice index, only n=1 supported anyway
        jsonResponse.Index(index++);

        // logprobs: object/null; Log probability information for the choice. TODO
        if (this->request.logprobschat || this->request.logprobs) {
            jsonResponse.StartObject("logprobs");
            if (endpoint == Endpoint::CHAT_COMPLETIONS) {
                jsonResponse.StartArray("content");

                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    float logprob = generationOutput.generated_log_probs[i];
                    jsonResponse.LogprobObject(token, logprob);
                }
                jsonResponse.EndArray();
            }
            if (endpoint == Endpoint::COMPLETIONS) {
                jsonResponse.StartArray("tokens");
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    jsonResponse.String(token);
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("token_logprobs");
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    float logprob = generationOutput.generated_log_probs[i];
                    jsonResponse.LogprobValue(logprob);
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("top_logprobs");
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    jsonResponse.StartObject();
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    float logprob = generationOutput.generated_log_probs[i];
                    jsonResponse.Logprob(token, logprob);
                    jsonResponse.EndObject();
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("text_offset");
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    if (i == 0) {
                        jsonResponse.TextOffsetValue(0);
                    } else {
                        std::string text_before_token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids.begin(), generationOutput.generated_ids.begin() + i}));
                        jsonResponse.TextOffsetValue(text_before_token.size());
                    }
                }
                jsonResponse.EndArray();
            }
            jsonResponse.EndObject();
        } else {
            jsonResponse.Null("logprobs");  // "logprobs": null
        }

        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            jsonResponse.MessageObject(parsedOutput);
        } else if (endpoint == Endpoint::COMPLETIONS) {
            jsonResponse.Text(parsedOutput);
        }

        // finish message object
        jsonResponse.EndObject();
    }
    // finish choices array
    jsonResponse.EndArray();

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    jsonResponse.Int("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    jsonResponse.String("model", request.model);

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.String("object", "chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.String("object", "text_completion");
    }

    jsonResponse.UsageObject(usage);

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(ov::genai::EncodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();
    if (endpoint == Endpoint::RESPONSES) {
        std::vector<ParsedOutput> parsedOutputs;
        for (const auto& tokens : results.tokens) {
            parsedOutputs.push_back(parseOutputIfNeeded(tokens));
        }
        return serializeResponsesUnaryResponse(parsedOutputs);
    }

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    // choices: array of size N, where N is related to n request parameter
    jsonResponse.StartArray("choices");
    int index = 0;
    for (int i = 0; i < results.tokens.size(); i++) {
        const std::vector<int64_t>& tokens = results.tokens[i];
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
        ParsedOutput parsedOutput = parseOutputIfNeeded(tokens);
        jsonResponse.StartObject();
        // finish_reason: "stop" in regular scenario, "tool_calls" if output contains tool calls
        auto finishReason = mapFinishReason(ov::genai::GenerationFinishReason::STOP, !parsedOutput.toolCalls.empty());
        jsonResponse.FinishReason(finishReason.value_or("unknown"));
        // index: integer; Choice index, only n=1 supported anyway
        jsonResponse.Index(index++);

        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            jsonResponse.MessageObject(parsedOutput);
        } else if (endpoint == Endpoint::COMPLETIONS) {
            jsonResponse.Text(parsedOutput);
        }

        // finish message object
        jsonResponse.EndObject();
    }
    // finish choices array
    jsonResponse.EndArray();

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    jsonResponse.Int("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    jsonResponse.String("model", request.model);

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.String("object", "chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.String("object", "text_completion");
    }

    jsonResponse.UsageObject(usage);

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(ov::genai::VLMDecodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();
    if (endpoint == Endpoint::RESPONSES) {
        std::vector<ParsedOutput> parsedOutputs;
        for (const std::string& text : results.texts) {
            auto result = tokenizer.encode(text);
            auto& input_ids = result.input_ids;
            if (input_ids.get_shape().size() != 2)
                throw std::runtime_error("input_ids should have 2 dimensions");
            if (input_ids.get_shape()[0] != 1)
                throw std::runtime_error("input_ids should have 1 batch size");
            if (input_ids.get_element_type() != ov::element::i64)
                throw std::runtime_error("input_ids should have i64 element type");

            int64_t* input_ids_data = reinterpret_cast<int64_t*>(input_ids.data());
            std::vector<int64_t> generatedTokens(input_ids_data, input_ids_data + input_ids.get_shape()[1]);
            updateUsage(usage, generatedTokens, request.echo);
            parsedOutputs.push_back(parseOutputIfNeeded(generatedTokens));
        }
        return serializeResponsesUnaryResponse(parsedOutputs);
    }

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    // choices: array of size N, where N is related to n request parameter
    jsonResponse.StartArray("choices");
    int index = 0;

    for (int i = 0; i < results.texts.size(); i++) {
        const std::string& text = results.texts[i];
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated text: {}", text);

        // Workaround to use OVMS unary parsers: get tokens from string
        // This way we have detokenized text from GenAI and calculate tokens, to further convert back to text again, in parseOutputIfNeeded...
        auto result = tokenizer.encode(text);
        auto& input_ids = result.input_ids;
        if (input_ids.get_shape().size() != 2)
            throw std::runtime_error("input_ids should have 2 dimensions");
        if (input_ids.get_shape()[0] != 1)
            throw std::runtime_error("input_ids should have 1 batch size");
        if (input_ids.get_element_type() != ov::element::i64)
            throw std::runtime_error("input_ids should have i64 element type");

        int64_t* input_ids_data = reinterpret_cast<int64_t*>(input_ids.data());
        std::vector<int64_t> generatedTokens(input_ids_data, input_ids_data + input_ids.get_shape()[1]);

        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generatedTokens);
        ParsedOutput parsedOutput = parseOutputIfNeeded(generatedTokens);
        jsonResponse.StartObject();
        // finish_reason: "stop" in regular scenario, "tool_calls" if output contains tool calls
        auto finishReason = mapFinishReason(ov::genai::GenerationFinishReason::STOP, !parsedOutput.toolCalls.empty());
        jsonResponse.FinishReason(finishReason.value_or("unknown"));
        // index: integer; Choice index, only n=1 supported anyway
        jsonResponse.Index(index++);
        // logprobs: object/null; Log probability information for the choice. TODO

        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            jsonResponse.MessageObject(parsedOutput);
        } else if (endpoint == Endpoint::COMPLETIONS) {
            jsonResponse.Text(parsedOutput);
        }

        // finish message object
        jsonResponse.EndObject();
    }
    // finish choices array
    jsonResponse.EndArray();

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    jsonResponse.Int("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    jsonResponse.String("model", request.model);

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.String("object", "chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.String("object", "text_completion");
    }

    jsonResponse.UsageObject(usage);

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeResponsesStreamingInitEvents() {
    const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    const std::string outputItemId = "msg-0";

    std::vector<std::string> events;

    events.emplace_back(serializeResponsesEvent([this, &responseId, createdAt](Writer<StringBuffer>& writer) {
        writer.StartObject();
        writer.String("type");
        writer.String("response.created");
        writer.String("sequence_number");
        writer.Uint64(responsesStreamingSequenceNumber++);
        writer.String("response");
        serializeResponsesResponseObject(writer, responseId, createdAt, "in_progress", "", false);
        writer.EndObject();
    }));

    events.emplace_back(serializeResponsesEvent([this, &responseId, createdAt](Writer<StringBuffer>& writer) {
        writer.StartObject();
        writer.String("type");
        writer.String("response.in_progress");
        writer.String("sequence_number");
        writer.Uint64(responsesStreamingSequenceNumber++);
        writer.String("response");
        serializeResponsesResponseObject(writer, responseId, createdAt, "in_progress", "", false);
        writer.EndObject();
    }));

    events.emplace_back(serializeResponsesEvent([this, &outputItemId](Writer<StringBuffer>& writer) {
        writer.StartObject();
        writer.String("type");
        writer.String("response.output_item.added");
        writer.String("sequence_number");
        writer.Uint64(responsesStreamingSequenceNumber++);
        writer.String("output_index");
        writer.Uint64(0);
        writer.String("item");
        serializeResponsesOutputItem(writer, outputItemId, "", "in_progress", false);
        writer.EndObject();
    }));

    events.emplace_back(serializeResponsesEvent([this, &outputItemId](Writer<StringBuffer>& writer) {
        writer.StartObject();
        writer.String("type");
        writer.String("response.content_part.added");
        writer.String("sequence_number");
        writer.Uint64(responsesStreamingSequenceNumber++);
        writer.String("output_index");
        writer.Uint64(0);
        writer.String("content_index");
        writer.Uint64(0);
        writer.String("item_id");
        writer.String(outputItemId.c_str());
        writer.String("part");
        serializeResponsesPart(writer, "");
        writer.EndObject();
    }));

    responsesStreamingInitialized = true;

    std::stringstream ss;
    ss << events.front();
    for (size_t i = 1; i < events.size(); ++i) {
        ss << "\n\ndata: " << events[i];
    }
    return ss.str();
}

std::string OpenAIChatCompletionsHandler::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();
    if (endpoint == Endpoint::RESPONSES) {
        const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
        const std::string responseId = "resp-" + std::to_string(createdAt);
        const std::string outputItemId = "msg-0";

        std::vector<std::string> events;
        if (!responsesStreamingInitialized) {
            // Fallback: if init events were not sent earlier, emit them now
            std::string initEvents = serializeResponsesStreamingInitEvents();
            if (!initEvents.empty()) {
                events.emplace_back(std::move(initEvents));
            }
        }

        if (!chunkResponse.empty()) {
            responsesStreamingOutputText += chunkResponse;
            events.emplace_back(serializeResponsesEvent([this, &chunkResponse, &outputItemId](Writer<StringBuffer>& writer) {
                writer.StartObject();
                writer.String("type");
                writer.String("response.output_text.delta");
                writer.String("sequence_number");
                writer.Uint64(responsesStreamingSequenceNumber++);
                writer.String("output_index");
                writer.Uint64(0);
                writer.String("content_index");
                writer.Uint64(0);
                writer.String("item_id");
                writer.String(outputItemId.c_str());
                writer.String("delta");
                writer.String(chunkResponse.c_str());
                serializeNotSupportedEmptyArrayField(writer, "logprobs");
                writer.EndObject();
            }));
        }

        if (finishReason != ov::genai::GenerationFinishReason::NONE) {
            events.emplace_back(serializeResponsesEvent([this, &outputItemId](Writer<StringBuffer>& writer) {
                writer.StartObject();
                writer.String("type");
                writer.String("response.output_text.done");
                writer.String("sequence_number");
                writer.Uint64(responsesStreamingSequenceNumber++);
                writer.String("output_index");
                writer.Uint64(0);
                writer.String("content_index");
                writer.Uint64(0);
                writer.String("item_id");
                writer.String(outputItemId.c_str());
                writer.String("text");
                writer.String(responsesStreamingOutputText.c_str());
                serializeNotSupportedEmptyArrayField(writer, "logprobs");
                writer.EndObject();
            }));

            events.emplace_back(serializeResponsesEvent([this, &outputItemId](Writer<StringBuffer>& writer) {
                writer.StartObject();
                writer.String("type");
                writer.String("response.content_part.done");
                writer.String("sequence_number");
                writer.Uint64(responsesStreamingSequenceNumber++);
                writer.String("output_index");
                writer.Uint64(0);
                writer.String("content_index");
                writer.Uint64(0);
                writer.String("item_id");
                writer.String(outputItemId.c_str());
                writer.String("part");
                serializeResponsesPart(writer, responsesStreamingOutputText);
                writer.EndObject();
            }));

            events.emplace_back(serializeResponsesEvent([this, &outputItemId, finishReason](Writer<StringBuffer>& writer) {
                const char* itemStatus = (finishReason == ov::genai::GenerationFinishReason::LENGTH) ? "incomplete" : "completed";
                writer.StartObject();
                writer.String("type");
                writer.String("response.output_item.done");
                writer.String("sequence_number");
                writer.Uint64(responsesStreamingSequenceNumber++);
                writer.String("output_index");
                writer.Uint64(0);
                writer.String("item");
                serializeResponsesOutputItem(writer, outputItemId, responsesStreamingOutputText, itemStatus, true);
                writer.EndObject();
            }));

            events.emplace_back(serializeResponsesEvent([this, &responseId, createdAt, finishReason](Writer<StringBuffer>& writer) {
                const bool isIncomplete = (finishReason == ov::genai::GenerationFinishReason::LENGTH);
                const char* responseStatus = isIncomplete ? "incomplete" : "completed";
                const char* eventType = isIncomplete ? "response.incomplete" : "response.completed";
                const char* incompleteReason = isIncomplete ? "max_tokens" : nullptr;
                writer.StartObject();
                writer.String("type");
                writer.String(eventType);
                writer.String("sequence_number");
                writer.Uint64(responsesStreamingSequenceNumber++);
                writer.String("response");
                serializeResponsesResponseObject(writer, responseId, createdAt, responseStatus, responsesStreamingOutputText, true, incompleteReason);
                writer.EndObject();
            }));
        }

        if (events.empty()) {
            return "";
        }

        std::stringstream ss;
        ss << events.front();
        for (size_t i = 1; i < events.size(); ++i) {
            ss << "\n\ndata: " << events[i];
        }
        return ss.str();
    }

    Document doc;
    doc.SetObject();
    Document::AllocatorType& allocator = doc.GetAllocator();

    Value choices(kArrayType);
    Value choice(kObjectType);
    bool hasToolCalls = false;

    // choices: array of size N, where N is related to n request parameter
    choices.SetArray();
    choice.SetObject();
    // finish_reason: string or null; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)/null
    // "stop" => natural stop point due to stopping criteria
    // "length" => due to reaching max_tokens parameter
    // "content_filter" => when produced restricted output (not supported)
    // "tool_calls" => generation stopped and waiting for tool output
    // "function_call" => deprecated
    // null - natural scenario when the generation has not completed yet
    // index: integer; Choice index, only n=1 supported anyway
    choice.AddMember("index", 0, allocator);
    // logprobs: object/null; Log probability information for the choice. TODO
    choice.AddMember("logprobs", Value(), allocator);
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        if (outputParser != nullptr) {
            std::optional<Document> delta = outputParser->parseChunk(chunkResponse, areToolsAvailable(), finishReason);
            if (!delta.has_value()) {
                // If the generation is still ongoing, there is nothing to emit yet
                if (finishReason == ov::genai::GenerationFinishReason::NONE) {
                    return "";
                }
                // Generation finished but parser returned no delta (e.g. empty chunk after tool call).
                // We still need to emit a chunk with the appropriate finish_reason.
            }
            if (delta.has_value() && delta->HasMember("delta")) {
                // Deep copy the "delta" member value into the choice object
                choice.AddMember("delta", Value((*delta)["delta"], allocator), allocator);
                hasToolCalls = hasToolCallsInStreamingDelta(*delta);
                if (hasToolCalls) {
                    toolCallsDetectedInStream = true;
                }
            }

        } else {
            Value delta(kObjectType);
            delta.SetObject();
            delta.AddMember("content", Value(chunkResponse.c_str(), allocator), allocator);
            choice.AddMember("delta", delta, allocator);
        }
    } else if (endpoint == Endpoint::COMPLETIONS) {
        choice.AddMember("text", Value(chunkResponse.c_str(), allocator), allocator);
    }

    auto serializedFinishReason = mapFinishReason(finishReason, hasToolCalls || toolCallsDetectedInStream);
    if (serializedFinishReason.has_value()) {
        choice.AddMember("finish_reason", Value(serializedFinishReason.value().c_str(), allocator), allocator);
    } else {
        choice.AddMember("finish_reason", Value(rapidjson::kNullType), allocator);
    }

    choices.PushBack(choice, allocator);
    doc.AddMember("choices", choices, allocator);

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    doc.AddMember("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count(), allocator);

    // model: string; copied from the request
    doc.AddMember("model", Value(request.model.c_str(), allocator), allocator);

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        doc.AddMember("object", Value("chat.completion.chunk", allocator), allocator);
    } else if (endpoint == Endpoint::COMPLETIONS) {
        doc.AddMember("object", Value("text_completion.chunk", allocator), allocator);
    }

    if (request.streamOptions.includeUsage) {
        doc.AddMember("usage", Value(), allocator);
    }

    // TODO
    // id: string; A unique identifier for the chat completion. Each chunk has the same ID.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeResponsesFailedEvent(const std::string& errorMessage, const char* errorCode) {
    const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);

    std::vector<std::string> events;
    if (!responsesStreamingInitialized) {
        std::string initEvents = serializeResponsesStreamingInitEvents();
        if (!initEvents.empty()) {
            events.emplace_back(std::move(initEvents));
        }
    }

    events.emplace_back(serializeResponsesEvent([this, &responseId, createdAt, &errorMessage, errorCode](Writer<StringBuffer>& writer) {
        writer.StartObject();
        writer.String("type");
        writer.String("response.failed");
        writer.String("sequence_number");
        writer.Uint64(responsesStreamingSequenceNumber++);
        writer.String("response");
        serializeResponsesResponseObject(writer, responseId, createdAt, "failed", responsesStreamingOutputText, false,
            nullptr, errorMessage.c_str(), errorCode);
        writer.EndObject();
    }));

    std::stringstream ss;
    ss << events.front();
    for (size_t i = 1; i < events.size(); ++i) {
        ss << "\n\ndata: " << events[i];
    }
    return ss.str();
}

std::string OpenAIChatCompletionsHandler::serializeStreamingUsageChunk() {
    OVMS_PROFILE_FUNCTION();
    if (endpoint == Endpoint::RESPONSES) {
        return "";
    }
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    writer.String("choices");
    writer.StartArray();  // [
    writer.EndArray();    // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int64(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

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
    writer.Uint64(static_cast<uint64_t>(usage.promptTokens));
    writer.String("completion_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.completionTokens));
    writer.String("total_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.calculateTotalTokens()));
    writer.EndObject();  // }

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string OpenAIChatCompletionsHandler::serializeStreamingHandshakeChunk() {
    OVMS_PROFILE_FUNCTION();
    Document doc;
    doc.SetObject();
    Document::AllocatorType& allocator = doc.GetAllocator();

    Value choices(kArrayType);
    Value choice(kObjectType);

    // choices: array of size N, where N is related to n request parameter
    choices.SetArray();
    choice.SetObject();

    choice.AddMember("index", 0, allocator);
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        Value delta(kObjectType);
        delta.SetObject();
        delta.AddMember("role", Value("assistant", allocator), allocator);
        delta.AddMember("content", Value(rapidjson::kNullType), allocator);
        choice.AddMember("delta", delta, allocator);
    } else if (endpoint == Endpoint::COMPLETIONS) {
        choice.AddMember("text", Value(rapidjson::kNullType), allocator);
    }

    choice.AddMember("finish_reason", Value(rapidjson::kNullType), allocator);
    choices.PushBack(choice, allocator);

    doc.AddMember("choices", choices, allocator);

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    doc.AddMember("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count(), allocator);

    // model: string; copied from the request
    doc.AddMember("model", Value(request.model.c_str(), allocator), allocator);

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        doc.AddMember("object", Value("chat.completion.chunk", allocator), allocator);
    } else if (endpoint == Endpoint::COMPLETIONS) {
        doc.AddMember("object", Value("text_completion.chunk", allocator), allocator);
    }

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}
}  // namespace ovms
