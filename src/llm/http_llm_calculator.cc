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
#include <algorithm>
#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include <fmt/ranges.h>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../profiler.hpp"
#include "http_payload.hpp"
#include "llmnoderesources.hpp"
#include "text_processor.hpp"

// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

using namespace rapidjson;
using namespace ovms;

namespace mediapipe {

enum class Endpoint {
    CHAT_COMPLETIONS,
    COMPLETIONS,
};

using chat_entry_t = std::unordered_map<std::string, std::string>;
using chat_t = std::vector<chat_entry_t>;

#define IGNORE_EOS_MAX_TOKENS_LIMIT 4000

class OpenAIChatCompletionsRequest {
    Document& doc;

    chat_t messages;
    std::optional<std::string> prompt{std::nullopt};
    bool stream{false};
    std::string model;
    std::optional<int> maxTokens{std::nullopt};
    std::optional<float> frequencePenalty{std::nullopt};
    std::optional<float> presencePenalty{std::nullopt};
    std::optional<float> diversityPenalty{std::nullopt};
    std::optional<float> repetitionPenalty{std::nullopt};
    std::optional<float> lengthPenalty{std::nullopt};
    std::optional<int> numReturnSequences{std::nullopt};
    std::optional<float> temperature{std::nullopt};
    std::optional<float> topP{std::nullopt};
    std::optional<int> topK{std::nullopt};
    std::optional<int> seed{std::nullopt};
    std::optional<int> bestOf{std::nullopt};
    // std::optional<bool> useBeamSearch{std::nullopt};
    std::optional<bool> ignoreEOS{std::nullopt};
    Endpoint endpoint;

public:
    OpenAIChatCompletionsRequest(Document& doc, Endpoint endpoint) :
        doc(doc),
        endpoint(endpoint) {}

    ov::genai::GenerationConfig createGenerationConfig() const {
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
        if (frequencePenalty.has_value())
            config.frequency_penalty = frequencePenalty.value();
        if (presencePenalty.has_value())
            config.presence_penalty = presencePenalty.value();
        config.do_sample = config.temperature > 0.0f && config.num_beams == 1;

        return config;
    }

    chat_t getMessages() const { return this->messages; }
    Endpoint getEndpoint() const { return this->endpoint; }
    std::optional<std::string> getPrompt() const { return this->prompt; }
    std::optional<int> getNumReturnSequences() const { return this->numReturnSequences; }

    bool isStream() const { return this->stream; }
    std::string getModel() const { return this->model; }

    absl::Status parse(uint32_t maxTokensLimit, uint32_t bestOfLimit) {
        OVMS_PROFILE_FUNCTION();
        // stream: bool; optional
        if (!this->doc.IsObject())
            return absl::InvalidArgumentError("Received json is not an object");
        auto it = this->doc.FindMember("stream");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsBool())
                return absl::InvalidArgumentError("Stream is not bool");
            this->stream = it->value.GetBool();
        }

        // messages: [{role: content}, {role: content}, ...]; required
        if (this->endpoint == Endpoint::CHAT_COMPLETIONS) {
            it = doc.FindMember("messages");
            if (it == doc.MemberEnd())
                return absl::InvalidArgumentError("Messages missing in request");
            if (!it->value.IsArray())
                return absl::InvalidArgumentError("Messages are not an array");
            if (it->value.GetArray().Size() == 0)
                return absl::InvalidArgumentError("Messages array cannot be empty");
            this->messages.clear();
            this->messages.reserve(it->value.GetArray().Size());
            for (size_t i = 0; i < it->value.GetArray().Size(); i++) {
                const auto& obj = it->value.GetArray()[i];
                if (!obj.IsObject())
                    return absl::InvalidArgumentError("Message is not a JSON object");
                auto& chat = this->messages.emplace_back(chat_entry_t{});
                for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
                    if (!member->name.IsString())
                        return absl::InvalidArgumentError("Invalid message structure");
                    if (!member->value.IsString())
                        return absl::InvalidArgumentError("Invalid message structure");
                    chat[member->name.GetString()] = member->value.GetString();
                }
            }
        }

        // prompt: string
        if (this->endpoint == Endpoint::COMPLETIONS) {
            it = this->doc.FindMember("prompt");
            if (it != this->doc.MemberEnd()) {
                if (!it->value.IsString()) {
                    return absl::InvalidArgumentError("prompt is not a string");
                } else {
                    this->prompt = it->value.GetString();
                }
            }
        }
        // model: string; required
        it = this->doc.FindMember("model");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsString())
                return absl::InvalidArgumentError("model is not a string");
            this->model = it->value.GetString();
        } else {
            return absl::InvalidArgumentError("model missing in request");
        }

        // ignore_eos: bool; optional - defaults to false
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("ignore_eos");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsBool())
                return absl::InvalidArgumentError("ignore_eos accepts values true or false");
            this->ignoreEOS = it->value.GetBool();
        }

        // max_tokens: uint; optional
        it = this->doc.FindMember("max_tokens");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint()) {
                if (it->value.IsUint64())
                    return absl::InvalidArgumentError("max_tokens value can't be greater than 4294967295");
                return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
            }
            if (it->value.GetUint() == 0)
                return absl::InvalidArgumentError("max_tokens value should be greater than 0");
            if (!(it->value.GetUint() < maxTokensLimit))
                return absl::InvalidArgumentError(absl::StrCat("max_tokens exceeds limit provided in graph config: ", maxTokensLimit));
            this->maxTokens = it->value.GetUint();
        }
        if (this->ignoreEOS.value_or(false)) {
            if (this->maxTokens.has_value()) {
                if (it->value.GetUint() > IGNORE_EOS_MAX_TOKENS_LIMIT)
                    return absl::InvalidArgumentError("when ignore_eos is true max_tokens can not be greater than 4000");
            } else {
                this->maxTokens = IGNORE_EOS_MAX_TOKENS_LIMIT;
            }
        }

        // frequence_penalty: float; optional - defaults to 0
        it = this->doc.FindMember("frequence_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("frequence_penalty is not a valid number");
            this->frequencePenalty = it->value.GetDouble();
            if (this->frequencePenalty < -2.0f || this->frequencePenalty > 2.0f)
                return absl::InvalidArgumentError("frequence_penalty out of range(-2.0, 2.0)");
        }

        // presence_penalty: float; optional - defaults to 0
        it = this->doc.FindMember("presence_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("presence_penalty is not a valid number");
            this->presencePenalty = it->value.GetDouble();
            if (this->presencePenalty < -2.0f || this->presencePenalty > 2.0f)
                return absl::InvalidArgumentError("presence_penalty out of range(-2.0, 2.0)");
        }

        // repetition_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("repetition_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("repetition_penalty is not a valid number");
            this->repetitionPenalty = it->value.GetDouble();
        }

        // diversity_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API and vLLM, however available in CB lib
        it = this->doc.FindMember("diversity_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("diversity_penalty is not a valid number");
            this->diversityPenalty = it->value.GetDouble();
        }

        // length_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API however supported by vLLM and CB lib
        it = this->doc.FindMember("length_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("length_penalty is not a valid number");
            this->lengthPenalty = it->value.GetDouble();
        }

        // temperature: float; optional - defaults to 0.0 (different than OpenAI which is 1.0)
        it = this->doc.FindMember("temperature");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("temperature is not a valid number");
            this->temperature = it->value.GetDouble();
            if (this->temperature < 0.0f || this->temperature > 2.0f)
                return absl::InvalidArgumentError("temperature out of range(0.0, 2.0)");
        }

        // top_p: float; optional - defaults to 1
        it = this->doc.FindMember("top_p");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble() && !it->value.IsInt())
                return absl::InvalidArgumentError("top_p is not a valid number");
            this->topP = it->value.GetDouble();
            if (this->topP < 0.0f || this->topP > 1.0f)
                return absl::InvalidArgumentError("top_p out of range(0.0, 1.0)");
        }

        // top_k: int; optional - defaults to 0
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("top_k");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsInt())
                return absl::InvalidArgumentError("top_k is not an integer");
            this->topK = it->value.GetInt();
        }

        // seed: int; optional - defaults to 0 (not set)
        it = this->doc.FindMember("seed");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return absl::InvalidArgumentError("seed is not an unsigned integer");
            this->seed = it->value.GetUint();
        }

        // best_of: int; optional - defaults to 1
        // Extension, unsupported by OpenAI API, however supported by vLLM, supported in CB lib by mapping to group_size param
        it = this->doc.FindMember("best_of");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return absl::InvalidArgumentError("best_of is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return absl::InvalidArgumentError("best_of value should be greater than 0");
            if (!(it->value.GetUint() < bestOfLimit))
                return absl::InvalidArgumentError(absl::StrCat("best_of exceeds limit provided in graph config: ", bestOfLimit));
            this->bestOf = it->value.GetUint();
        }

        // n: int; optional - defaults to 1
        // Supported by OpenAI API and vLLM, supported in CB lib by mapping to num_return_sequences param
        it = this->doc.FindMember("n");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return absl::InvalidArgumentError("n is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return absl::InvalidArgumentError("n value should be greater than 0");
            size_t bestOf = this->bestOf.has_value() ? this->bestOf.value() : 1;  // 1 is default best_of value
            if (bestOf < it->value.GetUint()) {
                return absl::InvalidArgumentError("n value cannot be greater than best_of");
            }
            this->numReturnSequences = it->value.GetUint();
        }

        // use_beam_search: bool; optional - defaults to false
        // Extension from vLLM, unsupported by OpenAI API, not available directly in CB lib
        // Use best_of>1 to steer into beams search
        // it = this->doc.FindMember("use_beam_search");
        // if (it != this->doc.MemberEnd()) {
        //     if (!it->value.IsBool())
        //         return false;
        //     this->useBeamSearch = it->value.GetBool();
        // }

        // logit_bias TODO
        // logprops TODO
        // top_logprobs TODO
        // response_format TODO
        // stop TODO
        // stream_options TODO
        // tools TODO
        // tool_choice TODO
        // user TODO
        // function_call TODO (deprecated)
        // functions TODO (deprecated)
        return absl::OkStatus();
    }
};

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

static std::string packIntoServerSideEventMessage(const std::string& message) {
    std::stringstream ss;
    ss << "data: " << message << "\n\n";
    return ss.str();
}

// CB lib internals rely on request_id, so for now we provide increasing ID
static std::atomic<uint64_t> currentRequestId = 0;

class HttpLLMCalculator : public CalculatorBase {
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    ov::genai::GenerationHandle generationHandle;
    std::shared_ptr<OpenAIChatCompletionsRequest> request;
    std::shared_ptr<ClientConnection> client;

    // TODO: To be  moved to CB library
    std::shared_ptr<TextStreamer> streamer;

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs, Endpoint endpoint);
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason, Endpoint endpoint);

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->Outputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Open start", cc->NodeName());
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        RET_CHECK(it != nodeResourcesMap.end()) << "Could not find initialized LLM node named: " << cc->NodeName();
        nodeResources = it->second;
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        RET_CHECK(this->nodeResources != nullptr);

        // For cases where MediaPipe decides to trigger Process() when there are no inputs
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }
        try {
            // First iteration of Process()
            if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
                OVMS_PROFILE_SCOPE("Deserialization of first request");
                // Check if we did not receive the payload twice
                RET_CHECK(this->request == nullptr);
                RET_CHECK(this->generationHandle == nullptr);
                RET_CHECK(this->streamer == nullptr);
                RET_CHECK(this->client == nullptr);

                // Register resource creation time
                this->created = std::chrono::system_clock::now();

                InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
                Endpoint endpoint;
                if (payload.uri == "/v3/chat/completions") {
                    endpoint = Endpoint::CHAT_COMPLETIONS;
                } else if (payload.uri == "/v3/completions") {
                    endpoint = Endpoint::COMPLETIONS;
                } else {
                    return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
                }
                this->request = std::make_shared<OpenAIChatCompletionsRequest>(*payload.parsedJson, endpoint);
                this->client = payload.client;

                // TODO: Support chat scenario once atobisze adds that to CB library
                auto status = this->request->parse(nodeResources->maxTokensLimit, nodeResources->bestOfLimit);
                if (status != absl::OkStatus())
                    return status;

                std::string finalPrompt = "";
                switch (endpoint) {
                case Endpoint::CHAT_COMPLETIONS: {
                    if (this->request->getMessages().size() <= 0) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "There are no messages to apply for chat");
                    }
                    if (!TextProcessor::applyChatTemplate(this->nodeResources->textProcessor, this->nodeResources->modelsPath, payload.body, finalPrompt)) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, finalPrompt);
                    }
                    if (finalPrompt.size() == 0) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
                    }
                    break;
                }
                case Endpoint::COMPLETIONS: {
                    if (!this->request->getPrompt().has_value() || !this->request->getPrompt().value().size()) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "Prompt is missing");
                    }
                    finalPrompt = this->request->getPrompt().value();
                }
                }

                {
                    OVMS_PROFILE_SCOPE("pipeline->add_request()");

                    // Check if client disconnected while waiting in HTTP requests queue
                    if (this->client->isDisconnected()) {
                        return absl::CancelledError();
                    }

                    this->generationHandle = nodeResources->cbPipe->add_request(
                        currentRequestId++, /*to be removed from API?*/
                        finalPrompt,
                        this->request->createGenerationConfig());

                    this->client->registerDisconnectionCallback([genHandle = this->generationHandle]() {
                        genHandle->drop();
                    });
                }
                nodeResources->notifyExecutorThread();
                this->streamer = std::make_shared<TextStreamer>(
                    std::make_shared<ov::genai::Tokenizer>(nodeResources->cbPipe->get_tokenizer()));
            }

            RET_CHECK(this->generationHandle != nullptr);
            RET_CHECK(this->request != nullptr);
            RET_CHECK(this->streamer != nullptr);
            RET_CHECK(this->client != nullptr);

            // Unary scenario
            if (!this->request->isStream()) {
                OVMS_PROFILE_SCOPE("Unary generation cycle");

                std::vector<ov::genai::GenerationOutput> generationOutput = this->generationHandle->read_all();
                if (this->generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
                    return absl::CancelledError();
                }
                RET_CHECK(generationOutput.size() >= 1);
                std::sort(generationOutput.begin(), generationOutput.end(), [](ov::genai::GenerationOutput& r1, ov::genai::GenerationOutput& r2) {
                    return r1.score > r2.score;
                });

                // legacy
                if (generationOutput.size() == 1) {
                    std::vector<int64_t> tokens = generationOutput[0].generated_token_ids;
                    std::shared_ptr<ov::genai::Tokenizer> tokenizer = std::make_shared<ov::genai::Tokenizer>(nodeResources->cbPipe->get_tokenizer());
                    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
                    std::string completion = tokenizer->decode(tokens);

                    std::string response = serializeUnaryResponse(completion, this->request->getEndpoint());
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", response);
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                } else {
                    // Beam search only supported for unary
                    std::vector<std::string> completions;
                    for (ov::genai::GenerationOutput& out : generationOutput) {
                        std::vector<int64_t> tokens = out.generated_token_ids;
                        std::shared_ptr<ov::genai::Tokenizer> tokenizer = std::make_shared<ov::genai::Tokenizer>(nodeResources->cbPipe->get_tokenizer());
                        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
                        std::string completion = tokenizer->decode(tokens);
                        completions.emplace_back(completion);
                    }

                std::string response = serializeUnaryResponse(generationOutputs, this->request->getEndpoint());
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", response);
                cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
            } else {
                OVMS_PROFILE_SCOPE("Stream generation cycle");
                // Streaming scenario
                // Each iteration is single execution of Process() method

                if (this->generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
                    return absl::CancelledError();
                }

                if (this->generationHandle->get_status() == ov::genai::GenerationStatus::RUNNING || this->generationHandle->can_read()) {
                    // Subsequent iteration
                    OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
                    ov::genai::GenerationOutputs generationOutputs = this->generationHandle->read();
                    RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
                    RET_CHECK(generationOutputs.begin()->second.generated_token_ids.size() == 1);

                    // TODO(dkalinow): Move this logic to CB library
                    int64_t token = generationOutputs.begin()->second.generated_token_ids[0];
                    auto chunk = this->streamer->put(token);
                    ov::genai::GenerationFinishReason finishReason = generationOutputs.begin()->second.finish_reason;
                    if (finishReason == ov::genai::GenerationFinishReason::NONE) {  // continue
                        if (chunk.has_value()) {
                            std::string response = packIntoServerSideEventMessage(
                                serializeStreamingChunk(chunk.value(), finishReason, this->request->getEndpoint()));
                            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", response);
                            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{std::move(response)}, timestamp);
                        }
                        cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);
                    } else {  // finish generation
                        OVMS_PROFILE_SCOPE("Generation of last streaming response");
                        std::string response = packIntoServerSideEventMessage(serializeStreamingChunk(this->streamer->end(), finishReason, this->request->getEndpoint()));
                        response += packIntoServerSideEventMessage("[DONE]");
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", response);
                        // Produce last message, but do not produce loopback packets anymore so this is last Process() call
                        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{std::move(response)}, timestamp);
                    }
                }
            }
        } catch (ov::AssertFailure& e) {
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            return absl::InvalidArgumentError("Response generation failed");
        }
        timestamp = timestamp.NextAllowedInStream();

        return absl::OkStatus();
    }
};

std::string HttpLLMCalculator::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs, Endpoint endpoint) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int i = 0;
    int n = this->request->getNumReturnSequences().value_or(1);
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        if (i >= n)
            break;

        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generationOutput.generated_token_ids);
        std::string completeResponse = nodeResources->cbPipe->get_tokenizer().decode(generationOutput.generated_token_ids);
        writer.StartObject();  // {
        // finish_reason: string; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)
        // "stop" => natural stop point due to stopping criteria <---------------- the only used so far, remaining are TODO
        // "length" => due to reaching max_tokens parameter TODO
        // "content_filter" => when produced restricted output
        // "tool_calls" => generation stopped and waiting for tool output
        // "function_call" => deprecated
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
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(this->created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(this->request->getModel().c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion");
    }

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // TODO
    // usage: object; Usage statistics for the completion request.
    // Might be crucial - possibly required for benchmarking purposes?

    writer.EndObject();  // }
    return buffer.GetString();
}

std::string HttpLLMCalculator::serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason, Endpoint endpoint) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    // Can also be empty for the last chunk if you set stream_options: {"include_usage": true} TODO
    writer.String("choices");
    writer.StartArray();   // [
    writer.StartObject();  // {
    // finish_reason: string or null; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)/null
    // "stop" => natural stop point due to stopping criteria <---------------- the only used so far, remaining are TODO
    // "length" => due to reaching max_tokens parameter TODO
    // "content_filter" => when produced restricted output
    // "tool_calls" => generation stopped and waiting for tool output
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
        // Will make sense once we have chat templates? TODO(atobisze)
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
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(this->created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(this->request->getModel().c_str());

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion.chunk");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion.chunk");
    }

    // TODO
    // id: string; A unique identifier for the chat completion. Each chunk has the same ID.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // TODO
    // usage: object; An optional field that will only be present when you set stream_options: {"include_usage": true} in your request.
    // When present, it contains a null value except for the last chunk which contains the token usage statistics for the entire request.
    // Might be crucial - possibly required for benchmarking purposes?

    writer.EndObject();  // }
    return buffer.GetString();
}

// TODO: Names to be decided
const std::string HttpLLMCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string HttpLLMCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string HttpLLMCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(HttpLLMCalculator);

}  // namespace mediapipe
