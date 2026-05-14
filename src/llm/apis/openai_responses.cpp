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

#include "openai_responses.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <set>
#include <string>
#include <string.h>
#include <utility>
#include <vector>

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/str_cat.h"
#pragma warning(pop)

using namespace rapidjson;

namespace ovms {

static constexpr const char* OUTPUT_ITEM_ID = "msg-0";
static constexpr const char* REASONING_ITEM_ID = "rs-0";

static std::string joinServerSideEvents(const std::vector<std::string>& events) {
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

// Convert the Responses API tools array (flat function format) into the chat/completions
// nested format ({type:"function", function:{name, description, parameters, ...}}) in place
// on the request document. The chat template (e.g. gpt-oss) and the chat/completions tools
// schema both expect the nested shape; doing this once up front lets every downstream
// consumer (chat history path, processedJson builder for Python Jinja, parseToolsToJsonContainer)
// share the same representation. Tools already in nested form, or non-function tools, are
// left untouched.
static void convertResponsesToolsInPlace(rapidjson::Value& toolsArray, rapidjson::Document::AllocatorType& alloc) {
    if (!toolsArray.IsArray()) {
        return;
    }
    for (auto& tool : toolsArray.GetArray()) {
        if (!tool.IsObject()) {
            continue;
        }
        auto toolObj = tool.GetObject();
        if (toolObj.FindMember("function") != toolObj.MemberEnd()) {
            continue;  // Already in nested chat/completions format.
        }
        auto typeIt = toolObj.FindMember("type");
        const std::string toolType = (typeIt != toolObj.MemberEnd() && typeIt->value.IsString())
                                         ? typeIt->value.GetString()
                                         : "";
        if (toolType != "function") {
            continue;  // Preserve non-function tools as-is.
        }
        rapidjson::Value funcObj(rapidjson::kObjectType);
        for (auto memberIt = toolObj.MemberBegin(); memberIt != toolObj.MemberEnd();) {
            if (!memberIt->name.IsString()) {
                ++memberIt;
                continue;
            }
            const std::string fieldName = memberIt->name.GetString();
            if (fieldName == "type") {
                ++memberIt;
                continue;
            }
            if (fieldName == "response") {
                memberIt = tool.EraseMember(memberIt);
                continue;
            }
            rapidjson::Value keyCopy(memberIt->name, alloc);
            rapidjson::Value valCopy(memberIt->value, alloc);
            funcObj.AddMember(keyCopy, valCopy, alloc);
            memberIt = tool.EraseMember(memberIt);
        }
        tool.AddMember("function", funcObj, alloc);
    }
}

// Pull the reasoning text out of a Responses API "reasoning" item.
// Prefers the newer content[].text shape over the legacy summary[].text shape.
static std::string extractReasoningText(const rapidjson::Value::ConstObject& itemObj) {
    auto contentIt = itemObj.FindMember("content");
    if (contentIt != itemObj.MemberEnd() && contentIt->value.IsArray()) {
        for (const auto& ci : contentIt->value.GetArray()) {
            if (!ci.IsObject())
                continue;
            auto textIt = ci.GetObject().FindMember("text");
            if (textIt != ci.GetObject().MemberEnd() && textIt->value.IsString()) {
                return textIt->value.GetString();
            }
        }
    }
    auto summaryIt = itemObj.FindMember("summary");
    if (summaryIt != itemObj.MemberEnd() && summaryIt->value.IsArray()) {
        for (const auto& si : summaryIt->value.GetArray()) {
            if (!si.IsObject())
                continue;
            auto textIt = si.GetObject().FindMember("text");
            if (textIt != si.GetObject().MemberEnd() && textIt->value.IsString()) {
                return textIt->value.GetString();
            }
        }
    }
    return "";
}

// Extract a flat text string from a Responses API content field which may be
// either a string or an array of {type,text} objects.
static std::string extractTextContent(const rapidjson::Value& contentVal) {
    if (contentVal.IsString()) {
        return contentVal.GetString();
    }
    if (!contentVal.IsArray()) {
        return "";
    }
    for (const auto& ci : contentVal.GetArray()) {
        if (!ci.IsObject())
            continue;
        auto ctTypeIt = ci.GetObject().FindMember("type");
        if (ctTypeIt == ci.GetObject().MemberEnd() || !ctTypeIt->value.IsString())
            continue;
        const std::string ctType = ctTypeIt->value.GetString();
        if (ctType == "input_text" || ctType == "output_text") {
            auto textIt = ci.GetObject().FindMember("text");
            if (textIt != ci.GetObject().MemberEnd() && textIt->value.IsString()) {
                return textIt->value.GetString();
            }
        }
    }
    return "";
}

// Read the three string fields (id, name, arguments) out of a function_call item.
struct FunctionCallFields {
    std::string id;
    std::string name;
    std::string arguments;
};
static FunctionCallFields readFunctionCallFields(const rapidjson::Value& item) {
    FunctionCallFields out;
    auto fcObj = item.GetObject();
    auto idIt = fcObj.FindMember("id");
    if (idIt != fcObj.MemberEnd() && idIt->value.IsString())
        out.id = idIt->value.GetString();
    auto nameIt = fcObj.FindMember("name");
    if (nameIt != fcObj.MemberEnd() && nameIt->value.IsString())
        out.name = nameIt->value.GetString();
    auto argsIt = fcObj.FindMember("arguments");
    if (argsIt != fcObj.MemberEnd() && argsIt->value.IsString())
        out.arguments = argsIt->value.GetString();
    return out;
}

// Classification of a Responses API input item used to dispatch to per-type
// handlers in the builders below.
enum class ResponsesInputItemKind {
    REASONING,
    FUNCTION_CALL,
    FUNCTION_CALL_OUTPUT,
    ROLE_ITEM,
    MISSING_ROLE,
};

static absl::StatusOr<ResponsesInputItemKind> classifyInputItem(const rapidjson::Value& item) {
    if (!item.IsObject()) {
        return absl::InvalidArgumentError("input array items must be objects");
    }
    auto itemObj = item.GetObject();
    auto itemTypeIt = itemObj.FindMember("type");
    const std::string itemType = (itemTypeIt != itemObj.MemberEnd() && itemTypeIt->value.IsString())
                                     ? itemTypeIt->value.GetString()
                                     : "";
    if (itemType == "reasoning")
        return ResponsesInputItemKind::REASONING;
    if (itemType == "function_call")
        return ResponsesInputItemKind::FUNCTION_CALL;
    if (itemType == "function_call_output")
        return ResponsesInputItemKind::FUNCTION_CALL_OUTPUT;
    auto roleIt = itemObj.FindMember("role");
    if (roleIt == itemObj.MemberEnd() || !roleIt->value.IsString())
        return ResponsesInputItemKind::MISSING_ROLE;
    return ResponsesInputItemKind::ROLE_ITEM;
}

// Builds chat/completions-shaped messages from a Responses API input array.
//
// Reasoning items are buffered and attached as `reasoning_content` on the next
// assistant message (matching the gpt-oss template's expected field).
// Reasoning that is not followed by an assistant/function_call item is dropped,
// since emitting a standalone {role:assistant, reasoning_content:...} message
// with no content/tool_calls would confuse most chat templates.
//
// Pending function_call items are merged into the next assistant message as a
// chat/completions-shaped tool_calls[] array. Without this, the assistant turn
// would have no tool_calls field, the chat template would treat it as a final
// answer, and a subsequent tool message would fail (e.g. gpt-oss raises
// "Message has tool role, but there was no previous assistant message with a
// tool call!").
//
// Reasoning that is not followed by an assistant or function_call item is
// emitted as a standalone assistant turn with empty content and the buffered
// reasoning attached as `reasoning_content`. This preserves the model's
// chain-of-thought across turns even when the prior turn produced no visible
// output.
//
// The algorithm is sink-agnostic; concrete output (ov::genai::ChatHistory vs a
// rapidjson messages array) is provided by the Sink template parameter, which
// must implement:
//   absl::Status extractContent(itemObj, index, std::string& outText);
//   void emitToolMessage(callId, output);
//   void emitMessage(role, contentText, reasoning);  // reasoning empty -> skip
//   void emitAssistantWithToolCalls(contentText, reasoning, toolCalls);
//   void emitStandaloneReasoning(reasoning);  // assistant turn carrying only reasoning_content
//   absl::Status onMissingRole(itemObj);
template <typename Sink>
class ResponsesInputBuilder {
public:
    explicit ResponsesInputBuilder(Sink& sink) :
        sink(sink) {}

    absl::Status build(const rapidjson::Value& inputArray) {
        if (!inputArray.IsArray()) {
            return absl::InvalidArgumentError("input is not an array");
        }
        for (rapidjson::SizeType i = 0; i < inputArray.GetArray().Size(); ++i) {
            const auto& item = inputArray.GetArray()[i];
            auto kind = classifyInputItem(item);
            if (!kind.ok())
                return kind.status();
            absl::Status status;
            switch (kind.value()) {
            case ResponsesInputItemKind::REASONING:
                status = onReasoningItem(item.GetObject());
                break;
            case ResponsesInputItemKind::FUNCTION_CALL:
                pendingFunctionCalls.push_back(&item);
                break;
            case ResponsesInputItemKind::FUNCTION_CALL_OUTPUT:
                status = onFunctionCallOutputItem(item.GetObject());
                break;
            case ResponsesInputItemKind::ROLE_ITEM:
                status = onRoleItem(item.GetObject(), i);
                break;
            case ResponsesInputItemKind::MISSING_ROLE:
                status = sink.onMissingRole(item.GetObject());
                break;
            }
            if (!status.ok())
                return status;
        }
        // Flush any trailing buffered function_calls (e.g. input ends with a
        // function_call item that has no corresponding output yet).
        flushPendingFunctionCalls("");
        return absl::OkStatus();
    }

private:
    absl::Status onReasoningItem(const rapidjson::Value::ConstObject& itemObj) {
        std::string text = extractReasoningText(itemObj);
        if (!text.empty()) {
            if (!pendingReasoningContent.empty())
                pendingReasoningContent += "\n";
            pendingReasoningContent += text;
        }
        return absl::OkStatus();
    }

    absl::Status onFunctionCallOutputItem(const rapidjson::Value::ConstObject& itemObj) {
        flushPendingFunctionCalls("");
        std::string callId;
        auto callIdIt = itemObj.FindMember("call_id");
        if (callIdIt != itemObj.MemberEnd() && callIdIt->value.IsString())
            callId = callIdIt->value.GetString();
        std::string output;
        auto outputIt = itemObj.FindMember("output");
        if (outputIt != itemObj.MemberEnd() && outputIt->value.IsString())
            output = outputIt->value.GetString();
        sink.emitToolMessage(callId, output);
        return absl::OkStatus();
    }

    absl::Status onRoleItem(const rapidjson::Value::ConstObject& itemObj, rapidjson::SizeType index) {
        const std::string role = itemObj.FindMember("role")->value.GetString();
        std::string contentText;
        auto status = sink.extractContent(itemObj, index, contentText);
        if (!status.ok())
            return status;

        // Assistant role with buffered function_calls: merge into one message
        // (so the tool_calls field rides on the same assistant turn).
        if (role == "assistant" && !pendingFunctionCalls.empty()) {
            flushPendingFunctionCalls(contentText);
            return absl::OkStatus();
        }
        // Non-assistant items must not absorb pending tool_calls; flush first.
        // (flushPendingFunctionCalls also emits any standalone reasoning content
        // as a standalone assistant turn.)
        if (role != "assistant") {
            flushPendingFunctionCalls("");
        }

        std::string reasoning;
        if (role == "assistant" && !pendingReasoningContent.empty()) {
            reasoning = std::move(pendingReasoningContent);
            pendingReasoningContent.clear();
        }
        sink.emitMessage(role, contentText, reasoning);
        return absl::OkStatus();
    }

    void flushPendingFunctionCalls(const std::string& assistantText) {
        if (pendingFunctionCalls.empty()) {
            // No tool calls, but possibly buffered reasoning to flush as a
            // standalone assistant turn carrying only reasoning_content (no
            // `content` field at all, so templates that gate on `message.content`
            // skip the content branch and templates that gate on
            // `message.reasoning_content` still see the buffered text).
            if (!pendingReasoningContent.empty()) {
                std::string reasoning = std::move(pendingReasoningContent);
                pendingReasoningContent.clear();
                sink.emitStandaloneReasoning(reasoning);
            }
            return;
        }
        std::string reasoning = std::move(pendingReasoningContent);
        pendingReasoningContent.clear();
        sink.emitAssistantWithToolCalls(assistantText, reasoning, pendingFunctionCalls);
        pendingFunctionCalls.clear();
    }

    Sink& sink;
    std::vector<const rapidjson::Value*> pendingFunctionCalls;
    std::string pendingReasoningContent;
};

// Sink that appends to ov::genai::ChatHistory (used when Python is disabled
// or as the fallback C++ chat-history path). Owns a scratch rapidjson document
// whose allocator backs the tool_calls Values until they are deep-copied into
// a JsonContainer.
class ChatHistorySink {
public:
    ChatHistorySink(ov::genai::ChatHistory& chatHistory, ImageHistory& imageHistory,
        const std::optional<std::string>& allowedLocalMediaPath,
        const std::optional<std::vector<std::string>>& allowedMediaDomains) :
        chatHistory(chatHistory),
        imageHistory(imageHistory),
        allowedLocalMediaPath(allowedLocalMediaPath),
        allowedMediaDomains(allowedMediaDomains) {
        scratchDoc.SetObject();
    }

    absl::Status extractContent(const rapidjson::Value::ConstObject& itemObj,
        rapidjson::SizeType index, std::string& outText) {
        outText.clear();
        auto contentIt = itemObj.FindMember("content");
        if (contentIt == itemObj.MemberEnd())
            return absl::OkStatus();
        if (contentIt->value.IsString()) {
            outText = contentIt->value.GetString();
            return absl::OkStatus();
        }
        if (!contentIt->value.IsArray())
            return absl::InvalidArgumentError("input item content must be a string or array");
        for (const auto& contentItem : contentIt->value.GetArray()) {
            if (!contentItem.IsObject())
                return absl::InvalidArgumentError("input content items must be objects");
            auto contentObj = contentItem.GetObject();
            auto typeIt = contentObj.FindMember("type");
            if (typeIt == contentObj.MemberEnd() || !typeIt->value.IsString())
                return absl::InvalidArgumentError("input content item type is missing or invalid");
            const std::string type = typeIt->value.GetString();
            if (type == "input_text" || type == "output_text") {
                auto textIt = contentObj.FindMember("text");
                if (textIt == contentObj.MemberEnd() || !textIt->value.IsString())
                    return absl::InvalidArgumentError(absl::StrCat(type, " requires a valid text field"));
                // Last text-bearing item wins, matching pre-refactor behaviour.
                outText = textIt->value.GetString();
            } else if (type == "input_image") {
                auto status = appendInputImage(contentObj, index);
                if (!status.ok())
                    return status;
            } else {
                // Skip unrecognised content item types for forward compatibility.
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Skipping unsupported content type: {}", type);
            }
        }
        return absl::OkStatus();
    }

    void emitToolMessage(const std::string& callId, const std::string& output) {
        chatHistory.push_back({});
        chatHistory.last()["role"] = "tool";
        if (!callId.empty())
            chatHistory.last()["tool_call_id"] = callId;
        chatHistory.last()["content"] = output;
    }

    void emitMessage(const std::string& role, const std::string& contentText, const std::string& reasoning) {
        chatHistory.push_back({});
        chatHistory.last()["role"] = role;
        chatHistory.last()["content"] = contentText;
        if (!reasoning.empty())
            chatHistory.last()["reasoning_content"] = reasoning;
    }

    void emitAssistantWithToolCalls(const std::string& contentText, const std::string& reasoning,
        const std::vector<const rapidjson::Value*>& toolCalls) {
        chatHistory.push_back({});
        chatHistory.last()["role"] = "assistant";
        chatHistory.last()["content"] = contentText;
        if (!reasoning.empty())
            chatHistory.last()["reasoning_content"] = reasoning;
        auto& alloc = scratchDoc.GetAllocator();
        rapidjson::Value toolCallsArray(rapidjson::kArrayType);
        buildToolCallsArray(toolCalls, toolCallsArray, alloc);
        // rapidJsonValueToJsonContainer deep-copies, so scratchDoc can be reused.
        chatHistory.last()["tool_calls"] = rapidJsonValueToJsonContainer(toolCallsArray);
    }

    // Emit an assistant turn that carries only reasoning_content (no content,
    // no tool_calls). Used when reasoning is not followed by an assistant or
    // function_call item.
    void emitStandaloneReasoning(const std::string& reasoning) {
        chatHistory.push_back({});
        chatHistory.last()["role"] = "assistant";
        chatHistory.last()["reasoning_content"] = reasoning;
    }

    absl::Status onMissingRole(const rapidjson::Value::ConstObject&) {
        return absl::InvalidArgumentError("input item role is missing or invalid");
    }

private:
    absl::Status appendInputImage(const rapidjson::Value::ConstObject& contentObj, rapidjson::SizeType index) {
        auto imageUrlIt = contentObj.FindMember("image_url");
        if (imageUrlIt == contentObj.MemberEnd())
            return absl::InvalidArgumentError("input_image requires image_url field");

        std::string imageUrl;
        if (imageUrlIt->value.IsString()) {
            imageUrl = imageUrlIt->value.GetString();
        } else if (imageUrlIt->value.IsObject()) {
            auto imageUrlObj = imageUrlIt->value.GetObject();
            auto urlIt = imageUrlObj.FindMember("url");
            if (urlIt == imageUrlObj.MemberEnd() || !urlIt->value.IsString())
                return absl::InvalidArgumentError("input_image.image_url.url is missing or invalid");
            imageUrl = urlIt->value.GetString();
        } else {
            return absl::InvalidArgumentError("input_image.image_url must be a string or object");
        }

        auto tensorResult = loadImage(imageUrl, allowedLocalMediaPath, allowedMediaDomains);
        if (!tensorResult.ok())
            return tensorResult.status();
        imageHistory.push_back({index, tensorResult.value()});
        return absl::OkStatus();
    }

    // Build a chat/completions tool_calls[] array into outArr using the given allocator.
    static void buildToolCallsArray(const std::vector<const rapidjson::Value*>& toolCalls,
        rapidjson::Value& outArr, rapidjson::Document::AllocatorType& alloc) {
        for (const auto* fc : toolCalls) {
            const FunctionCallFields fields = readFunctionCallFields(*fc);
            rapidjson::Value funcObj(rapidjson::kObjectType);
            funcObj.AddMember("name", rapidjson::Value(fields.name.c_str(), alloc), alloc);
            funcObj.AddMember("arguments", rapidjson::Value(fields.arguments.c_str(), alloc), alloc);
            rapidjson::Value tcObj(rapidjson::kObjectType);
            tcObj.AddMember("id", rapidjson::Value(fields.id.c_str(), alloc), alloc);
            tcObj.AddMember("type", rapidjson::Value("function", alloc), alloc);
            tcObj.AddMember("function", funcObj, alloc);
            outArr.PushBack(tcObj, alloc);
        }
    }

    ov::genai::ChatHistory& chatHistory;
    ImageHistory& imageHistory;
    const std::optional<std::string>& allowedLocalMediaPath;
    const std::optional<std::vector<std::string>>& allowedMediaDomains;
    rapidjson::Document scratchDoc;
};

#if (PYTHON_DISABLE == 0)
// Sink that appends to a rapidjson messages array, used to feed the Python
// Jinja chat template path. Image content items are silently dropped (the
// Python path receives only text).
class ProcessedJsonSink {
public:
    ProcessedJsonSink(rapidjson::Value& messagesArray, rapidjson::Document::AllocatorType& alloc) :
        messagesArray(messagesArray),
        alloc(alloc) {}

    absl::Status extractContent(const rapidjson::Value::ConstObject& itemObj,
        rapidjson::SizeType /*index*/, std::string& outText) {
        auto contentIt = itemObj.FindMember("content");
        outText = (contentIt != itemObj.MemberEnd()) ? extractTextContent(contentIt->value) : "";
        return absl::OkStatus();
    }

    void emitToolMessage(const std::string& callId, const std::string& output) {
        rapidjson::Value msgObj(rapidjson::kObjectType);
        msgObj.AddMember("role", rapidjson::Value("tool", alloc), alloc);
        if (!callId.empty())
            msgObj.AddMember("tool_call_id", rapidjson::Value(callId.c_str(), alloc), alloc);
        msgObj.AddMember("content", rapidjson::Value(output.c_str(), alloc), alloc);
        messagesArray.PushBack(msgObj, alloc);
    }

    void emitMessage(const std::string& role, const std::string& contentText, const std::string& reasoning) {
        rapidjson::Value msgObj(rapidjson::kObjectType);
        msgObj.AddMember("role", rapidjson::Value(role.c_str(), alloc), alloc);
        msgObj.AddMember("content", rapidjson::Value(contentText.c_str(), alloc), alloc);
        if (!reasoning.empty())
            msgObj.AddMember("reasoning_content", rapidjson::Value(reasoning.c_str(), alloc), alloc);
        messagesArray.PushBack(msgObj, alloc);
    }

    // Emit an assistant turn that carries only reasoning_content (no content,
    // no tool_calls). See ChatHistorySink::emitStandaloneReasoning for rationale.
    void emitStandaloneReasoning(const std::string& reasoning) {
        rapidjson::Value msgObj(rapidjson::kObjectType);
        msgObj.AddMember("role", rapidjson::Value("assistant", alloc), alloc);
        msgObj.AddMember("reasoning_content", rapidjson::Value(reasoning.c_str(), alloc), alloc);
        messagesArray.PushBack(msgObj, alloc);
    }

    void emitAssistantWithToolCalls(const std::string& contentText, const std::string& reasoning,
        const std::vector<const rapidjson::Value*>& toolCalls) {
        rapidjson::Value msgObj(rapidjson::kObjectType);
        msgObj.AddMember("role", rapidjson::Value("assistant", alloc), alloc);
        msgObj.AddMember("content", rapidjson::Value(contentText.c_str(), alloc), alloc);
        if (!reasoning.empty())
            msgObj.AddMember("reasoning_content", rapidjson::Value(reasoning.c_str(), alloc), alloc);
        rapidjson::Value toolCallsArray(rapidjson::kArrayType);
        for (const auto* fc : toolCalls) {
            const FunctionCallFields fields = readFunctionCallFields(*fc);
            rapidjson::Value funcObj(rapidjson::kObjectType);
            funcObj.AddMember("name", rapidjson::Value(fields.name.c_str(), alloc), alloc);
            funcObj.AddMember("arguments", rapidjson::Value(fields.arguments.c_str(), alloc), alloc);
            rapidjson::Value tcObj(rapidjson::kObjectType);
            tcObj.AddMember("id", rapidjson::Value(fields.id.c_str(), alloc), alloc);
            tcObj.AddMember("type", rapidjson::Value("function", alloc), alloc);
            tcObj.AddMember("function", funcObj, alloc);
            toolCallsArray.PushBack(tcObj, alloc);
        }
        msgObj.AddMember("tool_calls", toolCallsArray, alloc);
        messagesArray.PushBack(msgObj, alloc);
    }

    absl::Status onMissingRole(const rapidjson::Value::ConstObject&) {
        // Silently skip unknown items without a role in the processed JSON path.
        return absl::OkStatus();
    }

private:
    rapidjson::Value& messagesArray;
    rapidjson::Document::AllocatorType& alloc;
};
#endif  // PYTHON_DISABLE == 0

// --- Request parsing ---

absl::Status OpenAIResponsesHandler::parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (status != absl::OkStatus())
        return status;
    status = parseResponsesPart(maxTokensLimit, allowedLocalMediaPath, allowedMediaDomains);
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
        ChatHistorySink sink(request.chatHistory, request.imageHistory,
            allowedLocalMediaPath, allowedMediaDomains);
        ResponsesInputBuilder<ChatHistorySink> builder(sink);
        auto status = builder.build(inputIt->value);
        if (!status.ok()) {
            return status;
        }
    } else {
        return absl::InvalidArgumentError("input is not a string or array");
    }

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed responses input directly to chat history without mutating request JSON");
    return absl::OkStatus();
}

absl::Status OpenAIResponsesHandler::parseResponsesPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    // Reject stream_options — usage is always included in response.completed event
    if (doc.FindMember("stream_options") != doc.MemberEnd()) {
        return absl::InvalidArgumentError("stream_options is not supported in Responses API.");
    }

    // input: string; required
    auto it = doc.FindMember("input");
    if (it == doc.MemberEnd()) {
        return absl::InvalidArgumentError("input missing in request");
    }

    // Convert tools array (Responses-flat -> chat/completions-nested) once, in place,
    // before any consumer reads it. parseInput, parseToolsToJsonContainer and the
    // processedJson builder all rely on the nested shape.
    auto toolsIt = doc.FindMember("tools");
    if (toolsIt != doc.MemberEnd() && toolsIt->value.IsArray()) {
        convertResponsesToolsInPlace(toolsIt->value, doc.GetAllocator());
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
            // Inject enable_thinking: true into chat_template_kwargs (merge with existing if present)
            auto kwargsIt = doc.FindMember("chat_template_kwargs");
            if (kwargsIt != doc.MemberEnd() && kwargsIt->value.IsObject()) {
                // Merge into existing kwargs
                if (kwargsIt->value.FindMember("enable_thinking") == kwargsIt->value.MemberEnd()) {
                    kwargsIt->value.AddMember("enable_thinking", true, doc.GetAllocator());
                }
            } else {
                rapidjson::Value kwargs(rapidjson::kObjectType);
                kwargs.AddMember("enable_thinking", true, doc.GetAllocator());
                doc.AddMember("chat_template_kwargs", kwargs, doc.GetAllocator());
            }
        }
        // summary field is accepted but ignored
    }

#if (PYTHON_DISABLE == 0)
    // Build processedJson with a "messages" array in chat/completions format so that
    // the Python Jinja template path can consume Responses API input without a separate code path.
    // Handles reasoning, function_call (merged into assistant tool_calls), and
    // function_call_output (converted to role:tool messages).
    {
        Document processedDoc;
        processedDoc.SetObject();
        auto& alloc = processedDoc.GetAllocator();

        Value messagesArray(kArrayType);

        auto inputArrIt = doc.FindMember("input");
        if (inputArrIt != doc.MemberEnd() && inputArrIt->value.IsArray()) {
            ProcessedJsonSink sink(messagesArray, alloc);
            ResponsesInputBuilder<ProcessedJsonSink> builder(sink);
            auto processedStatus = builder.build(inputArrIt->value);
            if (!processedStatus.ok()) {
                return processedStatus;
            }
        } else if (inputArrIt != doc.MemberEnd() && inputArrIt->value.IsString()) {
            // String input: emit a single user message so the Python Jinja path
            // sees the same content the C++ chatHistory path does.
            Value msgObj(kObjectType);
            msgObj.AddMember("role", Value("user", alloc), alloc);
            msgObj.AddMember("content", Value(inputArrIt->value.GetString(), alloc), alloc);
            messagesArray.PushBack(msgObj, alloc);
        }

        processedDoc.AddMember("messages", messagesArray, alloc);

        // Tools were already normalised to chat/completions nested format by
        // convertResponsesToolsInPlace earlier in parseResponsesPart — just copy verbatim.
        auto processedToolsIt = doc.FindMember("tools");
        if (processedToolsIt != doc.MemberEnd() && !processedToolsIt->value.IsNull()) {
            Value toolsCopy(processedToolsIt->value, alloc);
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

    return parseResponseFormat();
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

void OpenAIResponsesHandler::serializeCommonResponseParameters(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
    const std::string& status,
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
    // TODO: temperature are only included when explicitly provided in the request, but should be always in the response
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
    // TODO: top_p are only included when explicitly provided in the request, but should be always in the response
    if (request.topP.has_value()) {
        writer.String("top_p");
        writer.Double(static_cast<double>(request.topP.value()));
    }
    if (request.minP.has_value()) {
        writer.String("min_p");
        writer.Double(static_cast<double>(request.minP.value()));
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
}

void OpenAIResponsesHandler::serializeResponseObject(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
    const std::string& status, const std::string& fullOutputText, bool includeUsage,
    const std::optional<std::string>& incompleteReason, const std::optional<std::string>& errorMessage, ResponsesErrorCode errorCode) const {
    serializeCommonResponseParameters(writer, responseId, createdAt, status, incompleteReason, errorMessage, errorCode);

    writer.String("output");
    writer.StartArray();
    // Include reasoning output item if reasoning was produced during streaming
    if (!responsesState.reasoningText.empty()) {
        writer.StartObject();
        writer.String("id");
        writer.String(REASONING_ITEM_ID);
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
        writer.String(OUTPUT_ITEM_ID);
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
    const std::string responseId = "resp-" + std::to_string(createdAt);
    std::optional<std::string> incompleteReason = isIncomplete ? std::optional<std::string>("max_tokens") : std::nullopt;

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    serializeCommonResponseParameters(writer, responseId, createdAt, responseStatus, incompleteReason);

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
    if (results.finish_reasons.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Missing finish reason in unary LM responses generation result, defaulting to STOP");
    }
    std::vector<ParsedOutput> parsedOutputs;
    ov::genai::GenerationFinishReason responsesFinishReason = ov::genai::GenerationFinishReason::STOP;
    for (const auto& tokens : results.tokens) {
        parsedOutputs.push_back(parseOutputIfNeeded(tokens));
    }
    for (const auto& finishReason : results.finish_reasons) {
        if (finishReason == ov::genai::GenerationFinishReason::LENGTH) {
            responsesFinishReason = ov::genai::GenerationFinishReason::LENGTH;
            break;
        }
    }
    return serializeUnaryResponseImpl(parsedOutputs, responsesFinishReason);
}

std::string OpenAIResponsesHandler::serializeUnaryResponse(ov::genai::VLMDecodedResults& results, const std::string& textResponse) {
    OVMS_PROFILE_FUNCTION();
    usage.promptTokens = results.perf_metrics.get_num_input_tokens();
    usage.completionTokens = results.perf_metrics.get_num_generated_tokens();
    if (results.finish_reasons.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Missing finish reason in unary VLM responses generation result, defaulting to STOP");
    }
    // Usage is already correctly set from perf_metrics above — no need for updateUsage.
    std::vector<ParsedOutput> parsedOutputs;
    if (!textResponse.empty()) {
        if (outputParser != nullptr) {
            // Same workaround as in chat completions
            auto generatedTokens = encodeTextToTokens(textResponse);
            parsedOutputs.push_back(parseOutputIfNeeded(generatedTokens));
        } else {
            // Fast path: no output parser, use decoded text directly.
            ParsedOutput output;
            output.content = textResponse;
            parsedOutputs.push_back(std::move(output));
        }
    }
    ov::genai::GenerationFinishReason responsesFinishReason = ov::genai::GenerationFinishReason::STOP;
    for (const auto& finishReason : results.finish_reasons) {
        if (finishReason == ov::genai::GenerationFinishReason::LENGTH) {
            responsesFinishReason = ov::genai::GenerationFinishReason::LENGTH;
            break;
        }
    }
    return serializeUnaryResponseImpl(parsedOutputs, responsesFinishReason);
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

std::string OpenAIResponsesHandler::serializeStreamingCreatedEvent() {
    if (responsesState.createdSent) {
        return "";
    }
    responsesState.createdSent = true;
    const auto createdAt = std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.created");
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, "in_progress", "", false);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeStreamingInProgressEvent() {
    if (responsesState.inProgressSent) {
        return "";
    }
    responsesState.inProgressSent = true;
    const auto createdAt = std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    writeEventHeader(writer, "response.in_progress");
    writer.String("response");
    serializeResponseObject(writer, responseId, createdAt, "in_progress", "", false);
    writer.EndObject();
    return buffer.GetString();
}

std::string OpenAIResponsesHandler::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) {
    OVMS_PROFILE_FUNCTION();
    const auto createdAt = std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);
    const std::string outputItemId = OUTPUT_ITEM_ID;
    const std::string reasoningItemId = REASONING_ITEM_ID;

    std::vector<std::string> events;
    // Fallback: emit any lifecycle events not yet sent (methods are idempotent)
    std::string createdEvent = serializeStreamingCreatedEvent();
    if (!createdEvent.empty()) {
        events.emplace_back(std::move(createdEvent));
    }
    std::string inProgressEvent = serializeStreamingInProgressEvent();
    if (!inProgressEvent.empty()) {
        events.emplace_back(std::move(inProgressEvent));
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
            if (!responsesState.messageInitialized) {
                events.emplace_back(serializeOutputItemAddedEvent(outputItemId));
                events.emplace_back(serializeContentPartAddedEvent(outputItemId));
                responsesState.messageInitialized = true;
            }
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

    return joinServerSideEvents(events);
}

std::string OpenAIResponsesHandler::serializeFailedEvent(const std::string& errorMessage, ResponsesErrorCode errorCode) {
    const auto createdAt = std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count();
    const std::string responseId = "resp-" + std::to_string(createdAt);

    std::vector<std::string> events;
    // Emit any lifecycle events not yet sent (methods are idempotent)
    std::string createdEvent = serializeStreamingCreatedEvent();
    if (!createdEvent.empty()) {
        events.emplace_back(std::move(createdEvent));
    }
    std::string inProgressEvent = serializeStreamingInProgressEvent();
    if (!inProgressEvent.empty()) {
        events.emplace_back(std::move(inProgressEvent));
    }

    events.emplace_back(serializeFailedEventBody(responseId, createdAt, errorMessage, errorCode));

    return joinServerSideEvents(events);
}

std::string OpenAIResponsesHandler::serializeStreamingUsageChunk() {
    return "";
}

std::string OpenAIResponsesHandler::serializeStreamingHandshakeChunk() {
    return "";
}

}  // namespace ovms
