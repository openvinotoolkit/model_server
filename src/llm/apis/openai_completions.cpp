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

#include <limits>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../../logging.hpp"
#include "../../profiler.hpp"

using namespace rapidjson;

namespace ovms {

ov::genai::GenerationConfig OpenAIChatCompletionsRequest::createGenerationConfig() const {
    ov::genai::GenerationConfig config;

    // Generic
    if (maxTokens.has_value())
        config.max_new_tokens = maxTokens.value();
    // TODO: max_length = ?
    if (ignoreEOS.has_value())
        config.ignore_eos = ignoreEOS.value();

    // Beam search specific
    config.num_beam_groups = 1;  // OpenAI hardcoded
    config.num_beams = 1;        // OpenAI hardcoded
    config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();

    if (bestOf.has_value())
        config.num_beams = bestOf.value();

    if (diversityPenalty.has_value())
        config.diversity_penalty = diversityPenalty.value();  // TODO: Not available in OpenAI nor vLLM
    // TODO: stop_criteria = ?
    if (numReturnSequences.has_value())
        config.num_return_sequences = numReturnSequences.value();
    if (repetitionPenalty.has_value())
        config.repetition_penalty = repetitionPenalty.value();
    if (lengthPenalty.has_value())
        config.length_penalty = lengthPenalty.value();
    // TODO: no_repeat_ngram_size = ?
    // TODO: early_finish = ?
    // TODO use_beam_search is unused ?

    // Multinomial specific
    if (temperature.has_value())
        config.temperature = temperature.value();
    if (topK.has_value())
        config.top_k = topK.value();
    if (topP.has_value())
        config.top_p = topP.value();
    if (seed.has_value())
        config.rng_seed = seed.value();
    if (stop.has_value())
        config.stop_strings = stop.value();
    if (includeStopStrInOutput.has_value())
        config.include_stop_str_in_output = includeStopStrInOutput.value();
    if (frequencyPenalty.has_value())
        config.frequency_penalty = frequencyPenalty.value();
    if (presencePenalty.has_value())
        config.presence_penalty = presencePenalty.value();
    config.do_sample = config.temperature > 0.0f && config.num_beams == 1;

    return config;
}

absl::Status OpenAIChatCompletionsHandler::processCompletionsPart() {
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
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::processChatCompletionsPart() {
    // messages: [{role: content}, {role: content}, ...]; required
    auto it = doc.FindMember("messages");
    if (it == doc.MemberEnd())
        return absl::InvalidArgumentError("Messages missing in request");
    if (!it->value.IsArray())
        return absl::InvalidArgumentError("Messages are not an array");
    if (it->value.GetArray().Size() == 0)
        return absl::InvalidArgumentError("Messages array cannot be empty");
    request.messages.clear();
    request.messages.reserve(it->value.GetArray().Size());
    for (size_t i = 0; i < it->value.GetArray().Size(); i++) {
        const auto& obj = it->value.GetArray()[i];
        if (!obj.IsObject())
            return absl::InvalidArgumentError("Message is not a JSON object");
        auto& chat = request.messages.emplace_back(chat_entry_t{});
        for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
            if (!member->name.IsString())
                return absl::InvalidArgumentError("Invalid message structure");
            if (!member->value.IsString())
                return absl::InvalidArgumentError("Invalid message structure");
            chat[member->name.GetString()] = member->value.GetString();
        }
    }

    if (request.messages.size() <= 0) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "messages are missing");
    }
    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::processCommonPart(uint32_t maxTokensLimit, uint32_t bestOfLimit) {
    OVMS_PROFILE_FUNCTION();
    // stream: bool; optional
    if (!doc.IsObject())
        return absl::InvalidArgumentError("Received json is not an object");
    auto it = doc.FindMember("stream");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("Stream is not bool");
        request.stream = it->value.GetBool();
    }

    it = doc.FindMember("stream_options");
    if (it != doc.MemberEnd()) {
        if (!request.stream)
            return absl::InvalidArgumentError("stream_options provided, but stream not set to true");
        if (!it->value.IsObject())
            return absl::InvalidArgumentError("stream_options is not an object");
        auto streamOptionsObj = it->value.GetObject();

        int streamOptionsFound = 0;
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
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("ignore_eos accepts values true or false");
        request.ignoreEOS = it->value.GetBool();
    }

    // max_tokens: uint; optional
    it = doc.FindMember("max_tokens");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
        }
        if (it->value.GetUint() == 0)
            return absl::InvalidArgumentError("max_tokens value should be greater than 0");
        if (!(it->value.GetUint() < maxTokensLimit))
            return absl::InvalidArgumentError(absl::StrCat("max_tokens exceeds limit provided in graph config: ", maxTokensLimit));
        request.maxTokens = it->value.GetUint();
    }
    if (request.ignoreEOS.value_or(false)) {
        if (request.maxTokens.has_value()) {
            if (it->value.GetUint() > IGNORE_EOS_MAX_TOKENS_LIMIT)
                return absl::InvalidArgumentError("when ignore_eos is true max_tokens can not be greater than 4000");
        } else {
            request.maxTokens = IGNORE_EOS_MAX_TOKENS_LIMIT;
        }
    }

    // frequency_penalty: float; optional - defaults to 0
    it = doc.FindMember("frequency_penalty");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("frequency_penalty is not a valid number");
        request.frequencyPenalty = it->value.GetDouble();
        if (request.frequencyPenalty < -2.0f || request.frequencyPenalty > 2.0f)
            return absl::InvalidArgumentError("frequency_penalty out of range(-2.0, 2.0)");
    }

    // presence_penalty: float; optional - defaults to 0
    it = doc.FindMember("presence_penalty");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("presence_penalty is not a valid number");
        request.presencePenalty = it->value.GetDouble();
        if (request.presencePenalty < -2.0f || request.presencePenalty > 2.0f)
            return absl::InvalidArgumentError("presence_penalty out of range(-2.0, 2.0)");
    }

    // repetition_penalty: float; optional - defaults to 1.0
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("repetition_penalty");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("repetition_penalty is not a valid number");
        request.repetitionPenalty = it->value.GetDouble();
    }

    // diversity_penalty: float; optional - defaults to 1.0
    // Extension, unsupported by OpenAI API and vLLM, however available in CB lib
    it = doc.FindMember("diversity_penalty");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("diversity_penalty is not a valid number");
        request.diversityPenalty = it->value.GetDouble();
    }

    // length_penalty: float; optional - defaults to 1.0
    // Extension, unsupported by OpenAI API however supported by vLLM and CB lib
    it = doc.FindMember("length_penalty");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("length_penalty is not a valid number");
        request.lengthPenalty = it->value.GetDouble();
    }

    // temperature: float; optional - defaults to 0.0 (different than OpenAI which is 1.0)
    it = doc.FindMember("temperature");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("temperature is not a valid number");
        request.temperature = it->value.GetDouble();
        if (request.temperature < 0.0f || request.temperature > 2.0f)
            return absl::InvalidArgumentError("temperature out of range(0.0, 2.0)");
    }

    // top_p: float; optional - defaults to 1
    it = doc.FindMember("top_p");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsDouble() && !it->value.IsInt())
            return absl::InvalidArgumentError("top_p is not a valid number");
        request.topP = it->value.GetDouble();
        if (request.topP < 0.0f || request.topP > 1.0f)
            return absl::InvalidArgumentError("top_p out of range(0.0, 1.0)");
    }

    // top_k: int; optional - defaults to 0
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("top_k");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsInt())
            return absl::InvalidArgumentError("top_k is not an integer");
        request.topK = it->value.GetInt();
    }

    // seed: int; optional - defaults to 0 (not set)
    it = doc.FindMember("seed");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("seed is not an unsigned integer");
        request.seed = it->value.GetUint();
    }

    // stop: string or array; optional - defaults to null (not set)
    it = doc.FindMember("stop");
    if (it != doc.MemberEnd()) {
        if (it->value.IsString()) {
            request.stop = std::set<std::string>{it->value.GetString()};
        } else if (it->value.IsArray()) {
            auto stopArray = it->value.GetArray();
            // TODO: OpenAI API defines upper bound but do we want it?
            if (stopArray.Size() < 1 || stopArray.Size() > 4)
                return absl::InvalidArgumentError("stop array must have a least 1 and no more than 4 strings");

            request.stop = std::set<std::string>{};
            for (size_t i = 0; i < stopArray.Size(); i++) {
                const auto& element = stopArray[i];
                if (!element.IsString())
                    return absl::InvalidArgumentError("stop array contains non string element");
                request.stop->insert(element.GetString());
            }
        } else {
            return absl::InvalidArgumentError("stop is not a string or array of strings");
        }
    }

    // include_stop_str_in_output: bool; optional - defaults to false
    // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
    it = doc.FindMember("include_stop_str_in_output");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("include_stop_str_in_output accepts values true or false");
        if (!it->value.GetBool() && request.stream)
            return absl::InvalidArgumentError("include_stop_str_in_output must be explicitly set to true if streaming is used");
        request.includeStopStrInOutput = it->value.GetBool();
    }

    // best_of: int; optional - defaults to 1
    // Extension, unsupported by OpenAI API, however supported by vLLM, supported in CB lib by mapping to group_size param
    it = doc.FindMember("best_of");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint())
            return absl::InvalidArgumentError("best_of is not an unsigned integer");
        if (it->value.GetUint() == 0)
            return absl::InvalidArgumentError("best_of value should be greater than 0");
        if (!(it->value.GetUint() < bestOfLimit))
            return absl::InvalidArgumentError(absl::StrCat("best_of exceeds limit provided in graph config: ", bestOfLimit));
        request.bestOf = it->value.GetUint();
    }

    // n: int; optional - defaults to 1
    // Supported by OpenAI API and vLLM, supported in CB lib by mapping to num_return_sequences param
    it = doc.FindMember("n");
    if (it != doc.MemberEnd()) {
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

    // use_beam_search: bool; optional - defaults to false
    // Extension from vLLM, unsupported by OpenAI API, not available directly in CB lib
    // Use best_of>1 to steer into beams search
    // it = doc.FindMember("use_beam_search");
    // if (it != doc.MemberEnd()) {
    //     if (!it->value.IsBool())
    //         return false;
    //     request.useBeamSearch = it->value.GetBool();
    // }

    // logit_bias TODO
    // logprops TODO
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

void OpenAIChatCompletionsHandler::setPromptTokensUsage(int promptTokens) {
    usage.promptTokens = promptTokens;
}

void OpenAIChatCompletionsHandler::incrementCompletionTokensUsage() {
    usage.completionTokens++;
}

ov::genai::GenerationConfig OpenAIChatCompletionsHandler::createGenerationConfig() const {
    return request.createGenerationConfig();
}

absl::Status OpenAIChatCompletionsHandler::processRequest(uint32_t maxTokensLimit, uint32_t bestOfLimit) {
    absl::Status status = processCommonPart(maxTokensLimit, bestOfLimit);

    if (status != absl::OkStatus())
        return status;

    if (endpoint == Endpoint::COMPLETIONS)
        status = processCompletionsPart();
    else
        status = processChatCompletionsPart();

    return status;
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs, ov::genai::Tokenizer tokenizer) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int i = 0;
    int n = request.numReturnSequences.value_or(1);
    usage.completionTokens = 0;
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        if (i >= n)
            break;

        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generationOutput.generated_ids);
        usage.completionTokens += generationOutput.generated_ids.size();
        std::string completeResponse = tokenizer.decode(generationOutput.generated_ids);
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
        writer.Int(i++);
        // logprobs: object/null; Log probability information for the choice. TODO
        writer.String("logprobs");
        writer.Null();
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

}  // namespace ovms
