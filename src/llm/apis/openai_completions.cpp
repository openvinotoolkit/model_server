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
#include <limits>
#include <memory>
#include <stdexcept>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include <set>
#include <string>
#include <string.h>
#include <vector>

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "openai_json_response.hpp"

#include "../../logging.hpp"
#include "../../profiler.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/str_cat.h"
#pragma warning(pop)

using namespace rapidjson;

namespace ovms {

static bool hasToolCallsInStreamingDelta(const rapidjson::Document& delta) {
    if (!delta.HasMember("delta") || !delta["delta"].IsObject()) {
        return false;
    }
    const auto& deltaObj = delta["delta"];
    return deltaObj.HasMember("tool_calls") && deltaObj["tool_calls"].IsArray();
}

// --- Request parsing ---

absl::Status OpenAIChatCompletionsHandler::parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (status != absl::OkStatus())
        return status;
    if (endpoint == Endpoint::COMPLETIONS)
        status = parseCompletionsPart();
    else
        status = parseChatCompletionsPart(maxTokensLimit, allowedLocalMediaPath, allowedMediaDomains);

    return status;
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

    return parseResponseFormat();
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
                        auto tensorResult = loadImage(url, allowedLocalMediaPath, allowedMediaDomains);
                        if (!tensorResult.ok()) {
                            return tensorResult.status();
                        }
                        request.imageHistory.push_back({i, tensorResult.value()});
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

// --- Unary response serialization ---

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) {
    OVMS_PROFILE_FUNCTION();

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

        // TODO: logprobs: object/null; Log probability information for the choice.
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
                        std::string textBeforeToken = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids.begin(), generationOutput.generated_ids.begin() + i}));
                        jsonResponse.TextOffsetValue(textBeforeToken.size());
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

    // TODO: id: string; A unique identifier for the chat completion.

    // TODO: system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(ov::genai::EncodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();

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

    // TODO: id: string; A unique identifier for the chat completion.

    // TODO: system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(ov::genai::VLMDecodedResults& results) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();

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
        auto generatedTokens = encodeTextToTokens(text);

        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generatedTokens);
        ParsedOutput parsedOutput = parseOutputIfNeeded(generatedTokens);
        jsonResponse.StartObject();
        // finish_reason: "stop" in regular scenario, "tool_calls" if output contains tool calls
        auto finishReason = mapFinishReason(ov::genai::GenerationFinishReason::STOP, !parsedOutput.toolCalls.empty());
        jsonResponse.FinishReason(finishReason.value_or("unknown"));
        // index: integer; Choice index, only n=1 supported anyway
        jsonResponse.Index(index++);
        // TODO: logprobs: object/null; Log probability information for the choice.

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

    // TODO: id: string; A unique identifier for the chat completion.

    // TODO: system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // finish response object
    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

// --- Streaming serialization ---

std::string OpenAIChatCompletionsHandler::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();

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
    // TODO: logprobs: object/null; Log probability information for the choice.
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

    // TODO: id: string; A unique identifier for the chat completion. Each chunk has the same ID.

    // TODO: system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
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

void OpenAIChatCompletionsHandler::incrementProcessedTokens(size_t numTokens) {
    processedTokens += numTokens;
    if (!request.echo || processedTokens > usage.promptTokens)
        usage.completionTokens += numTokens;
}
}  // namespace ovms
