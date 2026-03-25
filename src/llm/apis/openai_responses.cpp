//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "openai_responses.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include <set>
#include <string.h>

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "../../filesystem.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include <regex>
#include "../../image_conversion.hpp"

using namespace rapidjson;

namespace ovms {

constexpr std::string_view RESPONSES_BASE64_PREFIX = "base64,";
constexpr int64_t RESPONSES_MAX_IMAGE_SIZE_BYTES = 20000000;  // 20MB

// --- Request parsing ---

absl::Status OpenAIResponsesHandler::parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (status != absl::OkStatus())
        return status;
    status = parsePart(maxTokensLimit, allowedLocalMediaPath, allowedMediaDomains);
    return status;
}

absl::Status OpenAIResponsesHandler::parseInput(std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
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

            std::string contentText = "";
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

                    std::size_t pos = imageUrl.find(RESPONSES_BASE64_PREFIX);
                    std::string decoded;
                    ov::Tensor tensor;
                    if (pos != std::string::npos) {
                        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Loading image from base64 string");
                        size_t offset = pos + RESPONSES_BASE64_PREFIX.length();
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
                        if (!allowedMediaDomains.has_value() || !isDomainAllowed(allowedMediaDomains.value(), imageUrl.c_str())) {
                            return absl::InvalidArgumentError("Given url does not match any allowed domain from allowed_media_domains");
                        }
                        auto status = downloadImage(imageUrl.c_str(), decoded, RESPONSES_MAX_IMAGE_SIZE_BYTES);
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

absl::Status OpenAIResponsesHandler::parsePart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    // input: string; required
    auto it = doc.FindMember("input");
    if (it == doc.MemberEnd()) {
        return absl::InvalidArgumentError("input missing in request");
    }

    auto messagesStatus = parseInput(allowedLocalMediaPath, allowedMediaDomains);
    if (!messagesStatus.ok()) {
        return messagesStatus;
    }

    // reasoning: object; optional
    // OpenAI Responses API reasoning parameter. Any effort value enables thinking mode.
    it = doc.FindMember("reasoning");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsObject()) {
            return absl::InvalidArgumentError("reasoning is not an object");
        }
        const auto& reasoningObj = it->value;
        auto effortIt = reasoningObj.FindMember("effort");
        if (effortIt != reasoningObj.MemberEnd() && !effortIt->value.IsNull()) {
            if (!effortIt->value.IsString()) {
                return absl::InvalidArgumentError("reasoning.effort is not a string");
            }
            const std::string effort = effortIt->value.GetString();
            if (effort != "low" && effort != "medium" && effort != "high") {
                return absl::InvalidArgumentError("reasoning.effort must be one of: low, medium, high");
            }
            // Inject enable_thinking: true into chat_template_kwargs if not already explicitly set
            auto kwargsIt = doc.FindMember("chat_template_kwargs");
            if (kwargsIt == doc.MemberEnd()) {
                rapidjson::Value kwargs(rapidjson::kObjectType);
                kwargs.AddMember("enable_thinking", true, doc.GetAllocator());
                doc.AddMember("chat_template_kwargs", kwargs, doc.GetAllocator());
            } else if (kwargsIt->value.IsObject()) {
                auto enableThinkingIt = kwargsIt->value.FindMember("enable_thinking");
                if (enableThinkingIt == kwargsIt->value.MemberEnd()) {
                    kwargsIt->value.AddMember("enable_thinking", true, doc.GetAllocator());
                }
                // If enable_thinking is already set explicitly, the user's value takes precedence
            }
        }
        // summary field is accepted but ignored
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

// --- Serialization helpers ---

void OpenAIResponsesHandler::serializeToolChoice(Writer<StringBuffer>& writer) const {
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

void OpenAIResponsesHandler::serializeTools(Writer<StringBuffer>& writer) const {
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

void OpenAIResponsesHandler::serializeResponseObject(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
    const std::string& status, const std::string& fullOutputText, bool includeUsage,
    const std::optional<std::string>& incompleteReason, const std::optional<std::string>& errorMessage, ResponsesErrorCode errorCode) const {
    writer.StartObject();
    writer.String("id");
    writer.String(responseId.c_str());
    writer.String("object");
    writer.String("response");
    writer.String("created_at");
    writer.Int64(createdAt);
    if (status == "completed") {
        const auto completedAt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        writer.String("completed_at");
        writer.Int64(completedAt);
    }
    if (incompleteReason.has_value()) {
        writer.String("incomplete_details");
        writer.StartObject();
        writer.String("reason");
        writer.String(incompleteReason.value().c_str());
        writer.EndObject();
    }
    writer.String("error");
    if (errorMessage.has_value()) {
        writer.StartObject();
        writer.String("code");
        writer.String(responsesErrorCodeToString(errorCode));
        writer.String("message");
        writer.String(errorMessage.value().c_str());
        writer.EndObject();
    } else {
        writer.Null();
    }
    writer.String("model");
    writer.String(request.model.c_str());
    writer.String("status");
    writer.String(status.c_str());

    writer.String("parallel_tool_calls");
    writer.Bool(true);
    // TODO: previous_response_id not supported
    writer.String("store");
    writer.Bool(true);
    // TODO: temperature are only included when explicitly provided in the request
    if (request.temperature.has_value()) {
        writer.String("temperature");
        writer.Double(static_cast<double>(request.temperature.value()));
    }
    writer.String("text");
    writer.StartObject();
    writer.String("format");
    writer.StartObject();
    writer.String("type");
    writer.String("text");
    writer.EndObject();
    writer.EndObject();
    serializeToolChoice(writer);
    serializeTools(writer);
    // TODO: top_p are only included when explicitly provided in the request
    if (request.topP.has_value()) {
        writer.String("top_p");
        writer.Double(static_cast<double>(request.topP.value()));
    }
    writer.String("truncation");
    writer.String("disabled");
    // TODO: user not supported
    writer.String("metadata");
    writer.StartObject();
    writer.EndObject();

    if (request.maxTokens.has_value()) {
        writer.String("max_output_tokens");
        writer.Uint64(static_cast<uint64_t>(request.maxTokens.value()));
    }

    writer.String("output");
    writer.StartArray();
    // Include reasoning output item if reasoning was produced during streaming
    if (!responsesState.reasoningText.empty()) {
        writer.StartObject();
        writer.String("id");
        writer.String("rs-0");
        writer.String("type");
        writer.String("reasoning");
        writer.String("summary");
        writer.StartArray();
        writer.StartObject();
        writer.String("type");
        writer.String("summary_text");
        writer.String("text");
        writer.String(responsesState.reasoningText.c_str());
        writer.EndObject();
        writer.EndArray();
        writer.EndObject();
    }
    // Include function_call output items if tool calls were produced during streaming
    for (const auto& toolCall : responsesState.toolCalls) {
        writer.StartObject();
        writer.String("id");
        writer.String(toolCall.id.c_str());
        writer.String("type");
        writer.String("function_call");
        writer.String("status");
        writer.String(status.c_str());
        writer.String("call_id");
        writer.String(toolCall.id.c_str());
        writer.String("name");
        writer.String(toolCall.name.c_str());
        writer.String("arguments");
        writer.String(toolCall.arguments.c_str());
        writer.EndObject();
    }
    {
        writer.StartObject();
        writer.String("id");
        writer.String("msg-0");
        writer.String("type");
        writer.String("message");
        writer.String("role");
        writer.String("assistant");
        writer.String("status");
        writer.String(status.c_str());
        writer.String("content");
        writer.StartArray();
        serializeTextPart(writer, fullOutputText);
        writer.EndArray();
        writer.EndObject();
    }
    writer.EndArray();

    if (includeUsage) {
        writer.String("usage");
        writer.StartObject();
        writer.String("input_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.promptTokens));
        // TODO: input_tokens_details.cached_tokens not supported
        writer.String("output_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.completionTokens));
        // TODO: output_tokens_details.reasoning_tokens not supported
        writer.String("total_tokens");
        writer.Uint64(static_cast<uint64_t>(usage.calculateTotalTokens()));
        writer.EndObject();
    }

    writer.EndObject();
}

void OpenAIResponsesHandler::serializeOutputItem(Writer<StringBuffer>& writer, const std::string& outputItemId,
    const std::string& text, const std::string& status) {
    writer.StartObject();
    writer.String("id");
    writer.String(outputItemId.c_str());
    writer.String("type");
    writer.String("message");
    writer.String("role");
    writer.String("assistant");
    writer.String("status");
    writer.String(status.c_str());
    writer.String("content");
    writer.StartArray();
    if (status != "in_progress") {
        serializeTextPart(writer, text);
    }
    writer.EndArray();
    writer.EndObject();
}

void OpenAIResponsesHandler::serializeTextPart(Writer<StringBuffer>& writer, const std::string& text) {
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

std::string OpenAIResponsesHandler::serializeUnaryResponseImpl(const std::vector<ParsedOutput>& parsedOutputs,
    ov::genai::GenerationFinishReason finishReason) const {
    const bool isIncomplete = (finishReason == ov::genai::GenerationFinishReason::LENGTH);
    const std::string responseStatus = isIncomplete ? "incomplete" : "completed";
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
    // TODO: error not supported in unary response
    writer.String("model");
    writer.String(request.model.c_str());
    writer.String("status");
    writer.String(responseStatus.c_str());

    writer.String("parallel_tool_calls");
    writer.Bool(true);
    // TODO: previous_response_id not supported
    writer.String("store");
    writer.Bool(true);
    // TODO: temperature/top_p are only included when explicitly provided in the request
    if (request.temperature.has_value()) {
        writer.String("temperature");
        writer.Double(static_cast<double>(request.temperature.value()));
    }
    writer.String("text");
    writer.StartObject();
    writer.String("format");
    writer.StartObject();
    writer.String("type");
    writer.String("text");
    writer.EndObject();
    writer.EndObject();
    serializeToolChoice(writer);
    serializeTools(writer);
    if (request.topP.has_value()) {
        writer.String("top_p");
        writer.Double(static_cast<double>(request.topP.value()));
    }
    writer.String("truncation");
    writer.String("disabled");
    // TODO: user not supported
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
        // Emit reasoning output item if reasoning is available
        if (!parsedOutput.reasoning.empty()) {
            const std::string reasoningId = "rs-" + std::to_string(outputIndex);
            writer.StartObject();
            writer.String("id");
            writer.String(reasoningId.c_str());
            writer.String("type");
            writer.String("reasoning");
            writer.String("summary");
            writer.StartArray();
            writer.StartObject();
            writer.String("type");
            writer.String("summary_text");
            writer.String("text");
            writer.String(parsedOutput.reasoning.c_str());
            writer.EndObject();
            writer.EndArray();
            writer.EndObject();
        }

        if (!parsedOutput.toolCalls.empty()) {
            // Emit function_call output items for each tool call
            for (const auto& toolCall : parsedOutput.toolCalls) {
                writer.StartObject();
                writer.String("id");
                writer.String(toolCall.id.c_str());
                writer.String("type");
                writer.String("function_call");
                writer.String("status");
                writer.String(responseStatus.c_str());
                writer.String("call_id");
                writer.String(toolCall.id.c_str());
                writer.String("name");
                writer.String(toolCall.name.c_str());
                writer.String("arguments");
                writer.String(toolCall.arguments.c_str());
                writer.EndObject();
            }
        }

        // Always emit message output item
        {
            const std::string outputId = "msg-" + std::to_string(outputIndex);

            writer.StartObject();
            writer.String("id");
            writer.String(outputId.c_str());
            writer.String("type");
            writer.String("message");
            writer.String("role");
            writer.String("assistant");
            writer.String("status");
            writer.String(responseStatus.c_str());
            writer.String("content");
            writer.StartArray();
            serializeTextPart(writer, parsedOutput.content);
            writer.EndArray();
            writer.EndObject();
        }

        outputIndex++;
    }
    writer.EndArray();

    writer.String("usage");
    writer.StartObject();
    writer.String("input_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.promptTokens));
    // TODO: input_tokens_details.cached_tokens not supported
    writer.String("output_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.completionTokens));
    // TODO: output_tokens_details.reasoning_tokens not supported
    writer.String("total_tokens");
    writer.Uint64(static_cast<uint64_t>(usage.calculateTotalTokens()));
    writer.EndObject();

    writer.EndObject();

    return buffer.GetString();
}

// --- Unary response serialization ---

std::string OpenAIResponsesHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) {
    OVMS_PROFILE_FUNCTION();
    std::vector<ParsedOutput> parsedOutputs;
    usage.completionTokens = 0;
    constexpr bool echo = false;  // echo is not supported in Responses API
    ov::genai::GenerationFinishReason responsesFinishReason = ov::genai::GenerationFinishReason::STOP;
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        updateUsage(usage, generationOutput.generated_ids, echo);
        parsedOutputs.push_back(parseOutputIfNeeded(generationOutput.generated_ids));
        if (generationOutput.finish_reason == ov::genai::GenerationFinishReason::LENGTH) {
            responsesFinishReason = ov::genai::GenerationFinishReason::LENGTH;
        }
    }
    return serializeUnaryResponseImpl(parsedOutputs, responsesFinishReason);
}

std::string OpenAIResponsesHandler::serializeUnaryResponse(ov::genai::EncodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();
    std::vector<ParsedOutput> parsedOutputs;
    for (const auto& tokens : results.tokens) {
        parsedOutputs.push_back(parseOutputIfNeeded(tokens));
    }
    return serializeUnaryResponseImpl(parsedOutputs);
}

std::string OpenAIResponsesHandler::serializeUnaryResponse(ov::genai::VLMDecodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();
    // Usage is already correctly set from perf_metrics above — no need for updateUsage.
    std::vector<ParsedOutput> parsedOutputs;
    for (const std::string& text : results.texts) {
        if (outputParser != nullptr) {
            // Same workaround as in chat completions
            auto result = tokenizer.encode(text);
            auto& input_ids = result.input_ids;
            if (input_ids.get_shape().size() != 2)
                throw std::runtime_error("input_ids should have 2 dimensions");
            if (input_ids.get_shape()[0] != 1)
                throw std::runtime_error("input_ids should have 1 batch size");
            if (input_ids.get_element_type() != ov::element::i64)
                throw std::runtime_error("input_ids should have i64 element type");

            int64_t* inputIdsData = reinterpret_cast<int64_t*>(input_ids.data());
            std::vector<int64_t> generatedTokens(inputIdsData, inputIdsData + input_ids.get_shape()[1]);
            parsedOutputs.push_back(parseOutputIfNeeded(generatedTokens));
        } else {
            // Fast path: no output parser, use decoded text directly.
            ParsedOutput output;
            output.content = text;
            parsedOutputs.push_back(std::move(output));
        }
    }
    return serializeUnaryResponseImpl(parsedOutputs);
}

// --- Streaming event building blocks ---

void OpenAIResponsesHandler::writeEventHeader(Writer<StringBuffer>& writer, const char* eventType) {
    writer.StartObject();
    writer.String("type");
    writer.String(eventType);
    writer.String("sequence_number");
    writer.Uint64(responsesState.sequenceNumber++);
}

void OpenAIResponsesHandler::writeContentLocation(Writer<StringBuffer>& writer, const std::string& itemId, uint64_t outputIndex) {
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("content_index");
    writer.Uint64(0);
    writer.String("item_id");
    writer.String(itemId.c_str());
}

void OpenAIResponsesHandler::writeReasoningLocation(Writer<StringBuffer>& writer, const std::string& itemId) {
    writer.String("output_index");
    writer.Uint64(0);
    writer.String("summary_index");
    writer.Uint64(0);
    writer.String("item_id");
    writer.String(itemId.c_str());
}

// --- Individual streaming event serializers ---

std::string OpenAIResponsesHandler::serializeCreatedEvent(const std::string& responseId, int64_t createdAt) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.created");
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, "in_progress", "", false);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeInProgressEvent(const std::string& responseId, int64_t createdAt) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.in_progress");
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, "in_progress", "", false);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeOutputItemAddedEvent(const std::string& outputItemId, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.added");
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("item");
    serializeOutputItem(writer, outputItemId, "", "in_progress");
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeContentPartAddedEvent(const std::string& outputItemId, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.content_part.added");
    writeContentLocation(writer, outputItemId, outputIndex);
    writer.String("part");
    serializeTextPart(writer, "");
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeOutputTextDeltaEvent(const std::string& outputItemId, const std::string& delta, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_text.delta");
    writeContentLocation(writer, outputItemId, outputIndex);
    writer.String("delta");
    writer.String(delta.c_str());
    // TODO: logprobs not supported
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeOutputTextDoneEvent(const std::string& outputItemId, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_text.done");
    writeContentLocation(writer, outputItemId, outputIndex);
    writer.String("text");
    writer.String(responsesState.outputText.c_str());
    // TODO: logprobs not supported
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeContentPartDoneEvent(const std::string& outputItemId, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.content_part.done");
    writeContentLocation(writer, outputItemId, outputIndex);
    writer.String("part");
    serializeTextPart(writer, responsesState.outputText);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeOutputItemDoneEvent(const std::string& outputItemId, ov::genai::GenerationFinishReason finishReason, uint64_t outputIndex) {
    const std::string itemStatus = (finishReason == ov::genai::GenerationFinishReason::LENGTH) ? "incomplete" : "completed";
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.done");
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("item");
    serializeOutputItem(writer, outputItemId, responsesState.outputText, itemStatus);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeCompletedEvent(const std::string& responseId, int64_t createdAt, ov::genai::GenerationFinishReason finishReason) {
    const bool isIncomplete = (finishReason == ov::genai::GenerationFinishReason::LENGTH);
    const std::string responseStatus = isIncomplete ? "incomplete" : "completed";
    const char* eventType = isIncomplete ? "response.incomplete" : "response.completed";
    std::optional<std::string> incompleteReason = isIncomplete ? std::optional<std::string>("max_tokens") : std::nullopt;
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, eventType);
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, responseStatus, responsesState.outputText, true, incompleteReason);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeFailedEventBody(const std::string& responseId, int64_t createdAt, const std::string& errorMessage, ResponsesErrorCode errorCode) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.failed");
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, "failed", responsesState.outputText, false,
        std::nullopt, errorMessage, errorCode);
    writer.EndObject();
    return buffer.GetString();
}

// --- Reasoning streaming event serializers ---

std::string OpenAIResponsesHandler::serializeReasoningOutputItemAddedEvent(const std::string& reasoningItemId) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.added");
    writer.String("output_index");
    writer.Uint64(0);
    writer.String("item");
    writer.StartObject();
    writer.String("id");
    writer.String(reasoningItemId.c_str());
    writer.String("type");
    writer.String("reasoning");
    writer.String("summary");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeReasoningSummaryPartAddedEvent(const std::string& reasoningItemId) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.reasoning_summary_part.added");
    writeReasoningLocation(writer, reasoningItemId);
    writer.String("part");
    writer.StartObject();
    writer.String("type");
    writer.String("summary_text");
    writer.String("text");
    writer.String("");
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeReasoningSummaryTextDeltaEvent(const std::string& reasoningItemId, const std::string& delta) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.reasoning_summary_text.delta");
    writeReasoningLocation(writer, reasoningItemId);
    writer.String("delta");
    writer.String(delta.c_str());
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeReasoningSummaryTextDoneEvent(const std::string& reasoningItemId) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.reasoning_summary_text.done");
    writeReasoningLocation(writer, reasoningItemId);
    writer.String("text");
    writer.String(responsesState.reasoningText.c_str());
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeReasoningSummaryPartDoneEvent(const std::string& reasoningItemId) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.reasoning_summary_part.done");
    writeReasoningLocation(writer, reasoningItemId);
    writer.String("part");
    writer.StartObject();
    writer.String("type");
    writer.String("summary_text");
    writer.String("text");
    writer.String(responsesState.reasoningText.c_str());
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeReasoningOutputItemDoneEvent(const std::string& reasoningItemId) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.done");
    writer.String("output_index");
    writer.Uint64(0);
    writer.String("item");
    writer.StartObject();
    writer.String("id");
    writer.String(reasoningItemId.c_str());
    writer.String("type");
    writer.String("reasoning");
    writer.String("summary");
    writer.StartArray();
    writer.StartObject();
    writer.String("type");
    writer.String("summary_text");
    writer.String("text");
    writer.String(responsesState.reasoningText.c_str());
    writer.EndObject();
    writer.EndArray();
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

// --- Function call streaming event serializers ---

std::string OpenAIResponsesHandler::serializeFunctionCallOutputItemAddedEvent(const ToolCall& toolCall, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.added");
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("item");
    writer.StartObject();
    writer.String("id");
    writer.String(toolCall.id.c_str());
    writer.String("type");
    writer.String("function_call");
    writer.String("status");
    writer.String("in_progress");
    writer.String("call_id");
    writer.String(toolCall.id.c_str());
    writer.String("name");
    writer.String(toolCall.name.c_str());
    writer.String("arguments");
    writer.String("");
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeFunctionCallArgumentsDeltaEvent(const std::string& callId, const std::string& delta, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.function_call_arguments.delta");
    writer.String("item_id");
    writer.String(callId.c_str());
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("call_id");
    writer.String(callId.c_str());
    writer.String("delta");
    writer.String(delta.c_str());
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeFunctionCallArgumentsDoneEvent(const ToolCall& toolCall, uint64_t outputIndex) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.function_call_arguments.done");
    writer.String("item_id");
    writer.String(toolCall.id.c_str());
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("call_id");
    writer.String(toolCall.id.c_str());
    writer.String("arguments");
    writer.String(toolCall.arguments.c_str());
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeFunctionCallOutputItemDoneEvent(const ToolCall& toolCall, ov::genai::GenerationFinishReason finishReason, uint64_t outputIndex) {
    const std::string itemStatus = (finishReason == ov::genai::GenerationFinishReason::LENGTH) ? "incomplete" : "completed";
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.output_item.done");
    writer.String("output_index");
    writer.Uint64(outputIndex);
    writer.String("item");
    writer.StartObject();
    writer.String("id");
    writer.String(toolCall.id.c_str());
    writer.String("type");
    writer.String("function_call");
    writer.String("status");
    writer.String(itemStatus.c_str());
    writer.String("call_id");
    writer.String(toolCall.id.c_str());
    writer.String("name");
    writer.String(toolCall.name.c_str());
    writer.String("arguments");
    writer.String(toolCall.arguments.c_str());
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

// --- Top-level streaming methods ---

std::string OpenAIResponsesHandler::serializeStreamingInitEvents() {
    const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    const std::string outputItemId = "msg-0";

    std::vector<std::string> events;

    events.emplace_back(serializeCreatedEvent(responseId, createdAt));
    events.emplace_back(serializeInProgressEvent(responseId, createdAt));

    // When outputParser is present, defer output item events until first chunk
    // because reasoning items need to come before message items
    if (outputParser == nullptr) {
        events.emplace_back(serializeOutputItemAddedEvent(outputItemId));
        events.emplace_back(serializeContentPartAddedEvent(outputItemId));
        responsesState.messageInitialized = true;
    }

    responsesState.initialized = true;

    std::stringstream ss;
    ss << events.front();
    for (size_t i = 1; i < events.size(); ++i) {
        ss << "\n\ndata: " << events[i];
    }
    return ss.str();
}

std::string OpenAIResponsesHandler::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();
    const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    const std::string outputItemId = "msg-0";
    const std::string reasoningItemId = "rs-0";

    std::vector<std::string> events;
    if (!responsesState.initialized) {
        // Fallback: if init events were not sent earlier, emit them now
        std::string initEvents = serializeStreamingInitEvents();
        if (!initEvents.empty()) {
            events.emplace_back(std::move(initEvents));
        }
    }

    if (outputParser != nullptr) {
        // Use output parser to separate reasoning from content
        std::optional<Document> delta = outputParser->parseChunk(chunkResponse, areToolsAvailable(), finishReason);

        if (delta.has_value() && delta->HasMember("delta") && (*delta)["delta"].IsObject()) {
            const auto& deltaObj = (*delta)["delta"];
            if (deltaObj.HasMember("reasoning_content") && deltaObj["reasoning_content"].IsString()) {
                // Reasoning chunk
                if (!responsesState.reasoningInitialized) {
                    events.emplace_back(serializeReasoningOutputItemAddedEvent(reasoningItemId));
                    events.emplace_back(serializeReasoningSummaryPartAddedEvent(reasoningItemId));
                    responsesState.reasoningInitialized = true;
                }
                const std::string reasoningText = deltaObj["reasoning_content"].GetString();
                responsesState.reasoningText += reasoningText;
                events.emplace_back(serializeReasoningSummaryTextDeltaEvent(reasoningItemId, reasoningText));
            } else if (deltaObj.HasMember("content") && deltaObj["content"].IsString()) {
                // Content chunk - close reasoning if it was active, init message if needed
                if (responsesState.reasoningInitialized && !responsesState.reasoningCompleted) {
                    events.emplace_back(serializeReasoningSummaryTextDoneEvent(reasoningItemId));
                    events.emplace_back(serializeReasoningSummaryPartDoneEvent(reasoningItemId));
                    events.emplace_back(serializeReasoningOutputItemDoneEvent(reasoningItemId));
                    responsesState.reasoningCompleted = true;
                }
                const uint64_t msgIdx = responsesState.reasoningInitialized ? 1 : 0;
                if (!responsesState.messageInitialized) {
                    events.emplace_back(serializeOutputItemAddedEvent(outputItemId, msgIdx));
                    events.emplace_back(serializeContentPartAddedEvent(outputItemId, msgIdx));
                    responsesState.messageInitialized = true;
                }
                const std::string contentText = deltaObj["content"].GetString();
                responsesState.outputText += contentText;
                events.emplace_back(serializeOutputTextDeltaEvent(outputItemId, contentText, msgIdx));
            } else if (deltaObj.HasMember("tool_calls") && deltaObj["tool_calls"].IsArray()) {
                // Tool call chunk - close reasoning if active
                if (responsesState.reasoningInitialized && !responsesState.reasoningCompleted) {
                    events.emplace_back(serializeReasoningSummaryTextDoneEvent(reasoningItemId));
                    events.emplace_back(serializeReasoningSummaryPartDoneEvent(reasoningItemId));
                    events.emplace_back(serializeReasoningOutputItemDoneEvent(reasoningItemId));
                    responsesState.reasoningCompleted = true;
                }
                const auto& toolCallsArr = deltaObj["tool_calls"];
                for (rapidjson::SizeType i = 0; i < toolCallsArr.Size(); ++i) {
                    const auto& tc = toolCallsArr[i];
                    int tcIndex = tc.HasMember("index") ? tc["index"].GetInt() : 0;
                    // Determine the output index for this tool call
                    const uint64_t baseIdx = responsesState.reasoningInitialized ? 1 : 0;
                    const uint64_t tcOutputIdx = baseIdx + static_cast<uint64_t>(tcIndex);
                    // Determine if this is a new tool call (has function name)
                    bool isNewToolCall = false;
                    std::string funcName;
                    std::string tcId;
                    std::string argDelta;
                    if (tc.HasMember("function") && tc["function"].IsObject()) {
                        const auto& funcObj = tc["function"];
                        if (funcObj.HasMember("name") && funcObj["name"].IsString()) {
                            funcName = funcObj["name"].GetString();
                            isNewToolCall = true;
                        }
                        if (funcObj.HasMember("arguments") && funcObj["arguments"].IsString()) {
                            argDelta = funcObj["arguments"].GetString();
                        }
                    }
                    if (tc.HasMember("id") && tc["id"].IsString()) {
                        tcId = tc["id"].GetString();
                    }
                    if (isNewToolCall) {
                        // Ensure we have enough entries in our tracking vector
                        while (static_cast<int>(responsesState.toolCalls.size()) <= tcIndex) {
                            responsesState.toolCalls.push_back(ToolCall{});
                        }
                        responsesState.toolCalls[tcIndex].id = tcId;
                        responsesState.toolCalls[tcIndex].name = funcName;
                        responsesState.toolCalls[tcIndex].arguments = "";
                        events.emplace_back(serializeFunctionCallOutputItemAddedEvent(responsesState.toolCalls[tcIndex], tcOutputIdx));
                    }
                    if (!argDelta.empty() && static_cast<int>(responsesState.toolCalls.size()) > tcIndex) {
                        responsesState.toolCalls[tcIndex].arguments += argDelta;
                        events.emplace_back(serializeFunctionCallArgumentsDeltaEvent(responsesState.toolCalls[tcIndex].id, argDelta, tcOutputIdx));
                    }
                }
            }
        }
        // If delta is nullopt, the parser is accumulating tag tokens - skip
    } else {
        // No parser - pass through raw text
        if (!chunkResponse.empty()) {
            responsesState.outputText += chunkResponse;
            events.emplace_back(serializeOutputTextDeltaEvent(outputItemId, chunkResponse));
        }
    }

    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        // Close any open reasoning that wasn't closed by content transition
        if (responsesState.reasoningInitialized && !responsesState.reasoningCompleted) {
            events.emplace_back(serializeReasoningSummaryTextDoneEvent(reasoningItemId));
            events.emplace_back(serializeReasoningSummaryPartDoneEvent(reasoningItemId));
            events.emplace_back(serializeReasoningOutputItemDoneEvent(reasoningItemId));
            responsesState.reasoningCompleted = true;
        }
        // Emit done events for any streaming tool calls
        if (!responsesState.toolCalls.empty()) {
            const uint64_t baseIdx = responsesState.reasoningInitialized ? 1 : 0;
            for (size_t i = 0; i < responsesState.toolCalls.size(); ++i) {
                const uint64_t tcOutputIdx = baseIdx + static_cast<uint64_t>(i);
                events.emplace_back(serializeFunctionCallArgumentsDoneEvent(responsesState.toolCalls[i], tcOutputIdx));
                events.emplace_back(serializeFunctionCallOutputItemDoneEvent(responsesState.toolCalls[i], finishReason, tcOutputIdx));
            }
        }
        // Only emit message item if content was produced or no tool calls were generated
        if (!responsesState.outputText.empty() || responsesState.toolCalls.empty()) {
            const uint64_t msgIdx = (responsesState.reasoningInitialized ? 1 : 0) + responsesState.toolCalls.size();
            if (!responsesState.messageInitialized) {
                events.emplace_back(serializeOutputItemAddedEvent(outputItemId, msgIdx));
                events.emplace_back(serializeContentPartAddedEvent(outputItemId, msgIdx));
                responsesState.messageInitialized = true;
            }
            events.emplace_back(serializeOutputTextDoneEvent(outputItemId, msgIdx));
            events.emplace_back(serializeContentPartDoneEvent(outputItemId, msgIdx));
            events.emplace_back(serializeOutputItemDoneEvent(outputItemId, finishReason, msgIdx));
        }
        events.emplace_back(serializeCompletedEvent(responseId, createdAt, finishReason));
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

std::string OpenAIResponsesHandler::serializeFailedEvent(const std::string& errorMessage, ResponsesErrorCode errorCode) {
    const auto createdAt = std::chrono::duration_cast<std::chrono::microseconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);

    std::vector<std::string> events;
    if (!responsesState.initialized) {
        std::string initEvents = serializeStreamingInitEvents();
        if (!initEvents.empty()) {
            events.emplace_back(std::move(initEvents));
        }
    }

    events.emplace_back(serializeFailedEventBody(responseId, createdAt, errorMessage, errorCode));

    std::stringstream ss;
    ss << events.front();
    for (size_t i = 1; i < events.size(); ++i) {
        ss << "\n\ndata: " << events[i];
    }
    return ss.str();
}

std::string OpenAIResponsesHandler::serializeStreamingUsageChunk() {
    return "";
}

std::string OpenAIResponsesHandler::serializeStreamingHandshakeChunk() {
    return "";
}

}  // namespace ovms
