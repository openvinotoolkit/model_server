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

#include <fmt/ranges.h>

#include "openai_json_response.hpp"

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "src/filesystem/filesystem.hpp"
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
            // Handle tool_call_id and name string fields for tool calling, and reasoning_content for assistant messages
            if (member->value.IsString() && (memberName == "tool_call_id" || memberName == "name" || memberName == "reasoning_content")) {
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
                // Empty content arrays are accepted and preserved as-is. The
                // EmptyContentArrayNormalizationProcessor converts them to null before
                // downstream processing.
                for (const auto& v : member->value.GetArray()) {
                    if (!v.IsObject()) {
                        return absl::InvalidArgumentError("Invalid message structure - content array should contain objects");
                    }
                    const auto entry = v.GetObject();
                    if (!entry.HasMember("type") || !entry["type"].IsString()) {
                        return absl::InvalidArgumentError("Invalid message structure - content object type missing");
                    }
                    const std::string entryType = entry["type"].GetString();
                    if (entryType == "text") {
                        if (!entry.HasMember("text") || !entry["text"].IsString()) {
                            return absl::InvalidArgumentError("Invalid message structure - content text missing");
                        }
                    } else if (entryType == "image_url") {
                        if (!entry.HasMember("image_url") || !entry["image_url"].IsObject()) {
                            return absl::InvalidArgumentError("Invalid message structure - content image_url missing");
                        }
                        const auto imageUrl = entry["image_url"].GetObject();
                        if (!imageUrl.HasMember("url") || !imageUrl["url"].IsString()) {
                            return absl::InvalidArgumentError("Invalid message structure - image_url does not have url field");
                        }
                    } else if (entryType == "input_audio") {
                        if (!entry.HasMember("input_audio") || !entry["input_audio"].IsObject()) {
                            return absl::InvalidArgumentError("Invalid message structure - input_audio object missing");
                        }
                        const auto inputAudio = entry["input_audio"].GetObject();
                        if (!inputAudio.HasMember("data") || !inputAudio["data"].IsString()) {
                            return absl::InvalidArgumentError("Invalid message structure - input_audio does not have a valid data field");
                        }
                    } else {
                        return absl::InvalidArgumentError("Unsupported content type");
                    }
                }
                // Preserve content array for downstream processors
                // (ImageDecodingProcessor for VLM, TextContentNormalizationProcessor for LM).
                request.chatHistory.last()[memberName] = rapidJsonValueToJsonContainer(member->value);
            }
        }
        auto lastMessage = request.chatHistory.last();
        if (!lastMessage.contains("role")) {
            return absl::InvalidArgumentError("Every message must have 'role' field");
        }
        if (!lastMessage.contains("content")) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Message does not have content field which might be an issue for some chat templates. Adding empty content.");
            lastMessage["content"] = "";
        }
        // If message has tool calls, make sure each tool call has "arguments" field
        auto status = ensureArgumentsInToolCalls(obj);
        if (status != absl::OkStatus()) {
            return status;
        }
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed messages successfully");
    return absl::OkStatus();
}

// --- Unary response serialization ---

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(
    const std::vector<rapidjson::Document>& deltas,
    ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();
    ParsedOutput parsedOutput = parsedOutputFromDeltas(deltas);

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    jsonResponse.StartArray("choices");
    jsonResponse.StartObject();

    auto finishReasonStr = mapFinishReason(finishReason, !parsedOutput.toolCalls.empty());
    if (!finishReasonStr.has_value()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unknown finish reason: {}", static_cast<int>(finishReason));
    }
    jsonResponse.FinishReason(finishReasonStr.value_or("unknown"));
    jsonResponse.Index(0);
    jsonResponse.Null("logprobs");

    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.MessageObject(parsedOutput);
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.Text(parsedOutput);
    }

    jsonResponse.EndObject();
    jsonResponse.EndArray();

    jsonResponse.Int("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());
    jsonResponse.String("model", request.model);

    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.String("object", "chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.String("object", "text_completion");
    }

    jsonResponse.UsageObject(usage);

    if (isVerboseResponse()) {
        jsonResponse.StartObject("__verbose");
        jsonResponse.String("prompt", getVerbosePrompt());
        jsonResponse.String("content", getVerboseRawText());
        jsonResponse.EndObject();
    }

    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(
    const std::vector<std::vector<rapidjson::Document>>& allDeltas,
    const std::vector<ov::genai::GenerationFinishReason>& finishReasons,
    const std::vector<UnaryChoiceLogprobs>& logprobData) {
    OVMS_PROFILE_FUNCTION();

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    jsonResponse.StartArray("choices");
    for (size_t i = 0; i < allDeltas.size(); ++i) {
        ParsedOutput parsedOutput = parsedOutputFromDeltas(allDeltas[i]);

        jsonResponse.StartObject();

        const ov::genai::GenerationFinishReason finishReason =
            (i < finishReasons.size()) ? finishReasons[i] : ov::genai::GenerationFinishReason::STOP;
        auto finishReasonStr = mapFinishReason(finishReason, !parsedOutput.toolCalls.empty());
        if (!finishReasonStr.has_value()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unknown finish reason: {}", static_cast<int>(finishReason));
        }
        jsonResponse.FinishReason(finishReasonStr.value_or("unknown"));
        jsonResponse.Index(static_cast<int>(i));

        const bool hasChoiceLogprobs = !logprobData.empty() &&
                                       i < logprobData.size() &&
                                       !logprobData[i].generatedIds.empty() &&
                                       (request.logprobschat || request.logprobs);
        if (hasChoiceLogprobs) {
            jsonResponse.StartObject("logprobs");
            if (endpoint == Endpoint::CHAT_COMPLETIONS) {
                jsonResponse.StartArray("content");
                for (size_t j = 0; j < logprobData[i].generatedIds.size(); ++j) {
                    std::string token = tokenizer.decode(std::vector<int64_t>({logprobData[i].generatedIds[j]}),
                        ov::genai::skip_special_tokens(request.skipSpecialTokens));
                    const float logprob = (j < logprobData[i].logProbs.size()) ? logprobData[i].logProbs[j] : 0.0f;
                    jsonResponse.LogprobObject(token, logprob);
                }
                jsonResponse.EndArray();
            }
            if (endpoint == Endpoint::COMPLETIONS) {
                jsonResponse.StartArray("tokens");
                for (size_t j = 0; j < logprobData[i].generatedIds.size(); ++j) {
                    jsonResponse.String(tokenizer.decode(std::vector<int64_t>({logprobData[i].generatedIds[j]}),
                        ov::genai::skip_special_tokens(request.skipSpecialTokens)));
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("token_logprobs");
                for (size_t j = 0; j < logprobData[i].generatedIds.size(); ++j) {
                    jsonResponse.LogprobValue((j < logprobData[i].logProbs.size()) ? logprobData[i].logProbs[j] : 0.0f);
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("top_logprobs");
                for (size_t j = 0; j < logprobData[i].generatedIds.size(); ++j) {
                    jsonResponse.StartObject();
                    const std::string token = tokenizer.decode(std::vector<int64_t>({logprobData[i].generatedIds[j]}),
                        ov::genai::skip_special_tokens(request.skipSpecialTokens));
                    jsonResponse.Logprob(token, (j < logprobData[i].logProbs.size()) ? logprobData[i].logProbs[j] : 0.0f);
                    jsonResponse.EndObject();
                }
                jsonResponse.EndArray();

                jsonResponse.StartArray("text_offset");
                size_t offset = 0;
                for (size_t j = 0; j < logprobData[i].generatedIds.size(); ++j) {
                    jsonResponse.TextOffsetValue(static_cast<int>(offset));
                    offset += tokenizer.decode(std::vector<int64_t>({logprobData[i].generatedIds[j]}),
                                           ov::genai::skip_special_tokens(request.skipSpecialTokens))
                                  .size();
                }
                jsonResponse.EndArray();
            }
            jsonResponse.EndObject();
        } else {
            jsonResponse.Null("logprobs");
        }

        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            jsonResponse.MessageObject(parsedOutput);
        } else if (endpoint == Endpoint::COMPLETIONS) {
            jsonResponse.Text(parsedOutput);
        }

        jsonResponse.EndObject();
    }
    jsonResponse.EndArray();

    jsonResponse.Int("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());
    jsonResponse.String("model", request.model);

    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        jsonResponse.String("object", "chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        jsonResponse.String("object", "text_completion");
    }

    jsonResponse.UsageObject(usage);

    if (isVerboseResponse()) {
        jsonResponse.StartObject("__verbose");
        jsonResponse.String("prompt", getVerbosePrompt());
        jsonResponse.String("content", getVerboseRawText());
        jsonResponse.EndObject();
    }

    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

// --- Streaming serialization ---

std::string OpenAIChatCompletionsHandler::serializeStreamingChunk(rapidjson::Document parsedDelta, ov::genai::GenerationFinishReason finishReason) {
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
        // parsedDelta is a pre-parsed Document produced by OVMSTextStreamer::flush_chunk.
        // Shape: {"delta":{...}} for content/reasoning/tool_calls, or an empty Document{}
        // for finish-only chunks (generation ended on a swallowed token).
        if (parsedDelta.HasMember("delta")) {
            choice.AddMember("delta", Value(parsedDelta["delta"], allocator), allocator);
            hasToolCalls = hasToolCallsInStreamingDelta(parsedDelta);
            if (hasToolCalls) {
                toolCallsDetectedInStream = true;
            }
        } else {
            // No delta from the parser (e.g. generation ended on a swallowed token).
            // The OpenAI API requires "delta" to always be present in each choice, so emit an empty object.
            Value emptyDelta(kObjectType);
            choice.AddMember("delta", emptyDelta, allocator);
        }
    } else if (endpoint == Endpoint::COMPLETIONS) {
        // For /v1/completions, extract the plain text from the content delta.
        if (parsedDelta.HasMember("delta") && parsedDelta["delta"].IsObject() &&
            parsedDelta["delta"].HasMember("content") && parsedDelta["delta"]["content"].IsString()) {
            choice.AddMember("text", Value(parsedDelta["delta"]["content"].GetString(), allocator), allocator);
        } else {
            choice.AddMember("text", Value("", allocator), allocator);
        }
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

    // Verbose mode: attach prompt and raw model output to the FINAL chunk only.
    if (isVerboseResponse() && finishReason != ov::genai::GenerationFinishReason::NONE) {
        std::string rawOutput;
        if (!getVerboseRawTokens().empty()) {
            rawOutput = tokenizer.decode(getVerboseRawTokens(), ov::genai::skip_special_tokens(false));
        } else {
            rawOutput = getVerboseRawText();
        }

        Value verboseObject(kObjectType);
        verboseObject.AddMember("prompt", Value(getVerbosePrompt().c_str(), allocator), allocator);
        verboseObject.AddMember("content", Value(rawOutput.c_str(), allocator), allocator);
        doc.AddMember("__verbose", verboseObject, allocator);
    }

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
    // The handshake chunk signals that prefill is complete and generation has started.
    // Emitted on every endpoint so clients can distinguish prefill latency from
    // time-to-first-token.
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
        // Empty string (not null) so the field is present and typed as string.
        choice.AddMember("text", Value("", allocator), allocator);
    }

    choice.AddMember("finish_reason", Value(rapidjson::kNullType), allocator);
    choices.PushBack(choice, allocator);

    doc.AddMember("choices", choices, allocator);

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    doc.AddMember("created", std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count(), allocator);

    // model: string; copied from the request
    doc.AddMember("model", Value(request.model.c_str(), allocator), allocator);

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
    const size_t previousProcessed = processedTokens;
    processedTokens += numTokens;

    if (!request.echo) {
        usage.completionTokens += numTokens;
        return;
    }

    // Echo mode may deliver prompt+completion in one unary batch. Count only
    // the incremental portion that lies beyond prompt_tokens.
    const size_t previousCompletionBoundary =
        (previousProcessed > usage.promptTokens) ? (previousProcessed - usage.promptTokens) : 0;
    const size_t currentCompletionBoundary =
        (processedTokens > usage.promptTokens) ? (processedTokens - usage.promptTokens) : 0;

    if (currentCompletionBoundary > previousCompletionBoundary) {
        usage.completionTokens += (currentCompletionBoundary - previousCompletionBoundary);
    }
}
}  // namespace ovms
