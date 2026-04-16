//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "openai_api_handler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include <set>
#include <string.h>

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "../../filesystem/filesystem.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include <curl/curl.h>
#include <regex>
#include "../../image_conversion.hpp"

using namespace rapidjson;

namespace ovms {

constexpr size_t DEFAULT_MAX_STOP_WORDS = 16;  // same as deep-seek

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

// Default no-op implementations for streaming lifecycle events
std::string OpenAIApiHandler::serializeStreamingCreatedEvent() {
    return "";
}

std::string OpenAIApiHandler::serializeStreamingInProgressEvent() {
    return "";
}

std::string OpenAIApiHandler::serializeFailedEvent(const std::string& errorMessage, ResponsesErrorCode errorCode) {
    return "";
}

// --- Image download utilities ---

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

absl::Status downloadImage(const char* url, std::string& image, const int64_t& sizeLimit) {
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

bool isDomainAllowed(const std::vector<std::string>& allowedDomains, const char* url) {
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

absl::StatusOr<ov::Tensor> loadImage(const std::string& imageSource,
    const std::optional<std::string>& allowedLocalMediaPath,
    const std::optional<std::vector<std::string>>& allowedMediaDomains) {
    std::size_t pos = imageSource.find(BASE64_PREFIX);
    std::string decoded;
    ov::Tensor tensor;
    if (pos != std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
        size_t offset = pos + BASE64_PREFIX.length();
        if (!absl::Base64Unescape(std::string_view(imageSource.data() + offset, imageSource.size() - offset), &decoded)) {
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
    } else if (std::regex_match(imageSource.c_str(), std::regex("^(http|https|ftp|sftp|)://(.*)"))) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image using curl");
        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), imageSource.c_str())) {
            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
        }
        auto status = downloadImage(imageSource.c_str(), decoded, MAX_IMAGE_SIZE_BYTES);
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
        if (FileSystem::isPathEscaped(imageSource)) {
            std::stringstream ss;
            ss << "Path " << imageSource.c_str() << " escape with .. is forbidden.";
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
            return absl::InvalidArgumentError(ss.str());
        }
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from local filesystem");
        const auto firstMissmatch = std::mismatch(imageSource.begin(), imageSource.end(), allowedLocalMediaPath.value().begin(), allowedLocalMediaPath.value().end());
        if (firstMissmatch.second != allowedLocalMediaPath.value().end()) {
            return absl::InvalidArgumentError("Given filepath is not subpath of allowed_local_media_path");
        }
        try {
            tensor = loadImageStbiFromFile(imageSource.c_str());
        } catch (std::runtime_error& e) {
            std::stringstream ss;
            ss << "Image file " << imageSource.c_str() << " parsing failed: " << e.what();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
            return absl::InvalidArgumentError(ss.str());
        }
    }
    return tensor;
}

std::vector<int64_t> OpenAIApiHandler::encodeTextToTokens(const std::string& text) {
    auto result = tokenizer.encode(text);
    auto& input_ids = result.input_ids;
    if (input_ids.get_shape().size() != 2)
        throw std::runtime_error("input_ids should have 2 dimensions");
    if (input_ids.get_shape()[0] != 1)
        throw std::runtime_error("input_ids should have 1 batch size");
    if (input_ids.get_element_type() != ov::element::i64)
        throw std::runtime_error("input_ids should have i64 element type");
    int64_t* data = reinterpret_cast<int64_t*>(input_ids.data());
    return std::vector<int64_t>(data, data + input_ids.get_shape()[1]);
}

absl::Status OpenAIApiHandler::parseResponseFormat() {
    auto it = doc.FindMember("response_format");
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

// --- Shared parsing methods ---

absl::Status OpenAIApiHandler::ensureArgumentsInToolCalls(Value& messageObj, bool& jsonChanged) {
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

absl::Status OpenAIApiHandler::parseTools() {
    auto toolChoiceIt = doc.FindMember("tool_choice");
    std::string toolChoice{"auto"};
    if (toolChoiceIt != doc.MemberEnd() && !toolChoiceIt->value.IsNull()) {
        if (toolChoiceIt->value.IsString()) {
            toolChoice = toolChoiceIt->value.GetString();
            if (toolChoice != "none" && toolChoice != "auto" && toolChoice != "required")
                return absl::InvalidArgumentError("tool_choice should be either 'none' or 'auto' or 'required'");
        } else if (toolChoiceIt->value.IsObject()) {
            auto toolChoiceObj = toolChoiceIt->value.GetObject();
            auto toolChoiceFunctionIt = toolChoiceObj.FindMember("function");
            if (toolChoiceFunctionIt != toolChoiceObj.MemberEnd() && toolChoiceFunctionIt->value.IsObject()) {
                auto nameIt = toolChoiceFunctionIt->value.GetObject().FindMember("name");
                if (nameIt != toolChoiceFunctionIt->value.GetObject().MemberEnd() && nameIt->value.IsString()) {
                    toolChoice = nameIt->value.GetString();
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
                    toolChoice = nameIt->value.GetString();
                } else {
                    return absl::InvalidArgumentError("tool_choice.function is not a valid JSON object");
                }
            }
        } else {
            return absl::InvalidArgumentError("tool_choice is not a valid JSON object or string");
        }
    }
    bool jsonChanged = false;
    if (toolChoice == "none") {
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
            rapidjson::Value* parametersValue = nullptr;
            std::string functionName;

            auto functionIt = obj.FindMember("function");
            if (functionIt != obj.MemberEnd()) {
                auto typeIt = obj.FindMember("type");
                if (typeIt != obj.MemberEnd() && typeIt->value.IsString() && std::string(typeIt->value.GetString()) != "function") {
                    SPDLOG_WARN("Skipping unsupported tool type: {}", typeIt->value.GetString());
                    it->value.Erase(&obj);
                    jsonChanged = true;
                    continue;
                }
                if (!functionIt->value.IsObject()) {
                    return absl::InvalidArgumentError("Function is not a valid JSON object");
                }
                auto& functionObj = functionIt->value;
                auto nameIt = functionObj.GetObject().FindMember("name");
                if (nameIt == functionObj.GetObject().MemberEnd() || !nameIt->value.IsString()) {
                    return absl::InvalidArgumentError("Function object does not contain a valid name field");
                }
                functionName = nameIt->value.GetString();
                auto parametersIt = functionObj.GetObject().FindMember("parameters");
                if (parametersIt != functionObj.GetObject().MemberEnd()) {
                    parametersValue = &parametersIt->value;
                }
            } else {
                auto typeIt = obj.FindMember("type");
                if (typeIt == obj.MemberEnd() || !typeIt->value.IsString()) {
                    return absl::InvalidArgumentError("Tool type is missing or invalid");
                }
                if (std::string(typeIt->value.GetString()) != "function") {
                    SPDLOG_WARN("Skipping unsupported tool type: {}", typeIt->value.GetString());
                    it->value.Erase(&obj);
                    jsonChanged = true;
                    continue;
                }

                auto nameIt = obj.FindMember("name");
                if (nameIt == obj.MemberEnd() || !nameIt->value.IsString()) {
                    return absl::InvalidArgumentError("Function object does not contain a valid name field");
                }
                functionName = nameIt->value.GetString();

                auto parametersIt = obj.FindMember("parameters");
                if (parametersIt != obj.MemberEnd()) {
                    parametersValue = &parametersIt->value;
                }
            }

            // If toolChoice is set to "auto", we keep all tools
            // If toolChoice is set to a specific function name, we keep only that tool
            if (toolChoice != "auto" && toolChoice != "required" && toolChoice != functionName) {
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
                // Dump parameters object to string since this is the schema format expected by GenAI
                // Keep the rapidjson::Value pointer as well to avoid re-parsing in outputParsers
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                parametersValue->Accept(writer);
                std::string parametersStr = buffer.GetString();
                ToolSchemaWrapper schemaReprs{parametersValue, std::move(parametersStr)};
                request.toolNameSchemaMap[functionName] = std::move(schemaReprs);
            }
        }
    } else {
        toolChoice = "none";  // If tools are not provided, set toolChoice to "none"
    }

    request.toolChoice = toolChoice;
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    return absl::OkStatus();
}

absl::StatusOr<std::optional<ov::genai::JsonContainer>> OpenAIApiHandler::parseToolsToJsonContainer() {
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

absl::StatusOr<std::optional<ov::genai::JsonContainer>> OpenAIApiHandler::parseChatTemplateKwargsToJsonContainer() {
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

const bool OpenAIApiHandler::areToolsAvailable() const {
    return !request.toolNameSchemaMap.empty();
}

const OpenAIRequest& OpenAIApiHandler::getRequest() const {
    return request;
}

const std::string& OpenAIApiHandler::getProcessedJson() const {
    return request.processedJson;
}

const ImageHistory& OpenAIApiHandler::getImageHistory() const {
    return request.imageHistory;
}

ov::genai::ChatHistory& OpenAIApiHandler::getChatHistory() {
    return request.chatHistory;
}

std::optional<int> OpenAIApiHandler::getMaxTokens() const {
    return request.maxTokens;
}

std::optional<std::string> OpenAIApiHandler::getResponseFormat() const {
    return request.responseFormat;
}

std::optional<std::string> OpenAIApiHandler::getPrompt() const { return request.prompt; }
std::optional<int> OpenAIApiHandler::getNumReturnSequences() const { return request.numReturnSequences; }
StreamOptions OpenAIApiHandler::getStreamOptions() const { return request.streamOptions; }

bool OpenAIApiHandler::isStream() const { return request.stream; }
Endpoint OpenAIApiHandler::getEndpoint() const { return endpoint; }
std::string OpenAIApiHandler::getModel() const { return request.model; }
std::string OpenAIApiHandler::getToolChoice() const { return request.toolChoice; }
const std::unique_ptr<OutputParser>& OpenAIApiHandler::getOutputParser() const { return outputParser; }

void OpenAIApiHandler::setPromptTokensUsage(size_t promptTokens) {
    usage.promptTokens = promptTokens;
}

void OpenAIApiHandler::setCompletionTokensUsage(size_t completionTokens) {
    usage.completionTokens = completionTokens;
}

void OpenAIApiHandler::incrementProcessedTokens(size_t numTokens) {
    usage.completionTokens += numTokens;
}

ParsedOutput OpenAIApiHandler::parseOutputIfNeeded(const std::vector<int64_t>& generatedIds) {
    OVMS_PROFILE_FUNCTION();
    ParsedOutput parsedOutput;
    if ((endpoint != Endpoint::CHAT_COMPLETIONS && endpoint != Endpoint::RESPONSES) || outputParser == nullptr) {
        parsedOutput.content = this->tokenizer.decode(generatedIds);
    } else {
        parsedOutput = outputParser->parse(generatedIds, this->areToolsAvailable());
    }
    return parsedOutput;
}

// --- Free functions ---

void updateUsage(CompletionUsageStatistics& usage, const std::vector<int64_t>& generatedIds, bool echoPrompt) {
    OVMS_PROFILE_FUNCTION();
    usage.completionTokens += generatedIds.size();
    if (echoPrompt)
        usage.completionTokens -= usage.promptTokens;
}

std::optional<std::string> mapFinishReason(ov::genai::GenerationFinishReason finishReason, bool hasToolCalls) {
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

absl::Status OpenAIApiHandler::parseCommonPart(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength) {
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

    // TODO: logit_bias
    // TODO: top_logprobs
    // TODO: response_format
    // TODO: tools
    // TODO: tool_choice
    // TODO: user
    // TODO: function_call (deprecated)
    // TODO: functions (deprecated)
    return absl::OkStatus();
}

}  // namespace ovms
