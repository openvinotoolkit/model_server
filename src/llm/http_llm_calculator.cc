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

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../profiler.hpp"
#include "http_payload.hpp"
#include "llmnoderesources.hpp"

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
    // float frequencyPenalty{0.0f};
    // float presencePenalty{0.0f};
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

    GenerationConfig createGenerationConfig() const {
        GenerationConfig config;

        // Generic
        if (maxTokens.has_value())
            config.max_new_tokens = maxTokens.value();
        // TODO: max_length = ?
        if (ignoreEOS.has_value())
            config.ignore_eos = ignoreEOS.value();

        // Beam search specific
        config.num_groups = 1;  // OpenAI hardcoded
        if (bestOf.has_value())
            config.group_size = bestOf.value();
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
        config.do_sample = config.temperature > 0.0f && config.group_size == 1;

        return config;
    }

    chat_t getMessages() const { return this->messages; }
    Endpoint getEndpoint() const { return this->endpoint; }
    std::optional<std::string> getPrompt() const { return this->prompt; }

    bool isStream() const { return this->stream; }
    std::string getModel() const { return this->model; }

    absl::Status parse() {
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
            this->messages.clear();
            this->messages.reserve(it->value.GetArray().Size());
            for (int i = 0; i < it->value.GetArray().Size(); i++) {
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

        // max_tokens: int; optional
        it = this->doc.FindMember("max_tokens");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return absl::InvalidArgumentError("max_tokens value should be greater than 0");
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
        // TODO: Supported by OpenAI and vLLM, however unsupported by CB lib
        // // frequency_penalty: float; optional - defaults to 0
        // it = this->doc.FindMember("frequency_penalty");
        // if (it != this->doc.MemberEnd()) {
        //     return false;  // TODO: Unsupported by CB
        //     if (!it->value.IsDouble())
        //         return false;
        //     this->frequencyPenalty = it->value.GetDouble();
        //     if (this->frequencyPenalty < -2.0f || this->frequencyPenalty > 2.0f)
        //         return false;
        // }

        // TODO: Supported by OpenAI and vLLM, however unsupported by CB lib
        // // presence_penalty: float; optional - defaults to 0
        // it = this->doc.FindMember("presence_penalty");
        // if (it != this->doc.MemberEnd()) {
        //     return false;  // TODO: Unsupported by CB
        //     if (!it->value.IsDouble())
        //         return false;
        //     this->presencePenalty = it->value.GetDouble();
        //     if (this->presencePenalty < -2.0f || this->presencePenalty > 2.0f)
        //         return false;
        // }

        // repetition_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("repetition_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return absl::InvalidArgumentError("repetition_penalty is not a floating point number");
            this->repetitionPenalty = it->value.GetDouble();
        }

        // diversity_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API and vLLM, however available in CB lib
        it = this->doc.FindMember("diversity_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return absl::InvalidArgumentError("diversity_penalty is not a floating point number");
            this->diversityPenalty = it->value.GetDouble();
        }

        // length_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API however supported by vLLM and CB lib
        it = this->doc.FindMember("length_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return absl::InvalidArgumentError("length_penalty is not a floating point number");
            this->lengthPenalty = it->value.GetDouble();
        }

        // temperature: float; optional - defaults to 0.0 (different than OpenAI which is 1.0)
        it = this->doc.FindMember("temperature");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return absl::InvalidArgumentError("temperature is not a floating point number");
            this->temperature = it->value.GetDouble();
            if (this->temperature < 0.0f || this->temperature > 2.0f)
                return absl::InvalidArgumentError("temperature out of range(0.0, 2.0)");
        }

        // top_p: float; optional - defaults to 1
        it = this->doc.FindMember("top_p");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return absl::InvalidArgumentError("top_p is not a floating point number");
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
        // Use best_of>1 to steer into beams earch
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

// TODO: To be moved to CB library.
class TextStreamer {
    std::shared_ptr<Tokenizer> tokenizer;
    std::vector<int64_t> tokenCache;
    size_t printLen{0};

public:
    TextStreamer(std::shared_ptr<Tokenizer> tokenizer) :
        tokenizer(tokenizer) {}

    std::optional<std::string> put(int64_t token) {
        tokenCache.push_back(token);
        std::string text = tokenizer->decode(tokenCache);

        if (!text.empty() && '\n' == text.back()) {
            // The chunk is ready if the generated text ends with new line.
            // Also, clear the cache.
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            tokenCache.clear();
            printLen = 0;
            return chunk;
        } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {  // NOLINT
            return std::nullopt;
        } else if (text.size() > printLen) {
            // The chunk is ready if the new text in the cache contains space.
            // The chunk is constructed from the new text, however only up to the last space character (including it)
            // Does not clear the cache.
            auto lastSpacePos = text.rfind(' ');
            if (lastSpacePos == std::string::npos || lastSpacePos < printLen) {
                return std::nullopt;
            }
            std::string chunk = std::string{text.data() + printLen, lastSpacePos - printLen + 1};
            printLen = lastSpacePos + 1;
            return chunk;
        }
        return std::nullopt;
    }
};

static bool applyChatTemplate(TextProcessor& textProcessor, std::string modelsPath, std::string& requestBody, std::string& output) {
    if (textProcessor.chatTemplate == nullptr) {
        output = "Error: Chat template not loaded correctly, so it cannot be applied";
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("request_body"_a = requestBody, "chat_template"_a = textProcessor.chatTemplate->getObject(),
            "bos_token"_a = textProcessor.bosToken, "eos_token"_a = textProcessor.eosToken);
        py::exec(R"(
            output = ""
            error = ""
            try:
                messages = json.loads(request_body)["messages"]
                output = chat_template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True)
            except Exception as e:
                error = str(e)            
        )",
            py::globals(), locals);

        std::string result = locals["output"].cast<std::string>();
        std::string error = locals["error"].cast<std::string>();

        if (error != "") {
            output = error;
            return false;
        }

        output = result;
        return true;
    } catch (const pybind11::error_already_set& e) {
        LOG(INFO) << "Error occured when applying chat template: " << e.what();
        output = "Unexpected error occurred when applying chat template";
    } catch (...) {
        LOG(INFO) << "Unexpected error occurred when applying chat template";
        output = "Unexpected error occurred when applying chat template";
    }
    return false;
}

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
    GenerationHandle generationHandle;
    std::shared_ptr<OpenAIChatCompletionsRequest> request;

    // TODO: To be  moved to CB library
    std::shared_ptr<TextStreamer> streamer;

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;

    std::string serializeUnaryResponse(const std::string& completeResponse, Endpoint endpoint);
    std::string serializeUnaryResponse(const std::vector<std::string>& completeResponse, Endpoint endpoint);
    std::string serializeStreamingChunk(const std::string& chunkResponse, bool stop, Endpoint endpoint);

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
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        RET_CHECK(it != nodeResourcesMap.end()) << "Could not find initialized LLM node named: " << cc->NodeName();
        nodeResources = it->second;
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        OVMS_PROFILE_FUNCTION();
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
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

                // Register resource creation time
                this->created = std::chrono::system_clock::now();

                InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
                LOG(INFO) << "Request body: " << payload.body;
                LOG(INFO) << "Request uri: " << payload.uri;
                Endpoint endpoint;
                if (payload.uri == "/v3/chat/completions") {
                    endpoint = Endpoint::CHAT_COMPLETIONS;
                } else if (payload.uri == "/v3/completions") {
                    endpoint = Endpoint::COMPLETIONS;
                } else {
                    return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
                }
                this->request = std::make_shared<OpenAIChatCompletionsRequest>(*payload.parsedJson, endpoint);

                // TODO: Support chat scenario once atobisze adds that to CB library
                auto status = this->request->parse();
                if (status != absl::OkStatus())
                    return status;

                std::string finalPrompt = "";

                // LOG(INFO) << "Input prompt:" << templateApplyOutput;

                std::string prompt;
                switch (endpoint) {
                case Endpoint::CHAT_COMPLETIONS: {
                    if (this->request->getMessages().size() <= 0) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "There are no messages to apply for chat");
                    }
                    if (!applyChatTemplate(this->nodeResources->textProcessor, this->nodeResources->modelsPath, payload.body, finalPrompt)) {
                        return absl::Status(absl::StatusCode::kInvalidArgument, finalPrompt);
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
                    this->generationHandle = nodeResources->cbPipe->add_request(
                        currentRequestId++, /*to be removed from API?*/
                        finalPrompt,
                        this->request->createGenerationConfig());
                }
                nodeResources->notifyExecutorThread();
                this->streamer = std::make_shared<TextStreamer>(
                    nodeResources->cbPipe->get_tokenizer());
            }

            RET_CHECK(this->generationHandle != nullptr);
            RET_CHECK(this->request != nullptr);
            RET_CHECK(this->streamer != nullptr);

            // Unary scenario
            if (!this->request->isStream()) {
                OVMS_PROFILE_SCOPE("Unary generation cycle");
                std::vector<GenerationOutput> generationOutput = this->generationHandle->read_all();

                RET_CHECK(generationOutput.size() >= 1);
                // legacy
                if (generationOutput.size() == 1) {
                    std::vector<int64_t> tokens = generationOutput[0].generated_token_ids;
                    std::shared_ptr<Tokenizer> tokenizer = nodeResources->cbPipe->get_tokenizer();
                    std::string completion = tokenizer->decode(tokens);

                    std::string response = serializeUnaryResponse(tokenizer->decode(tokens), this->request->getEndpoint());
                    LOG(INFO) << "Complete unary response: " << response;
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                } else {
                    // Beam search only supported for unary
                    std::vector<std::string> completions;
                    for (GenerationOutput& out : generationOutput) {
                        std::vector<int64_t> tokens = out.generated_token_ids;
                        std::shared_ptr<Tokenizer> tokenizer = nodeResources->cbPipe->get_tokenizer();
                        std::string completion = tokenizer->decode(tokens);
                        completions.emplace_back(completion);
                    }

                    std::string response = serializeUnaryResponse(completions, this->request->getEndpoint());
                    LOG(INFO) << "Complete unary response: " << response;
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                }
            } else {
                OVMS_PROFILE_SCOPE("Stream generation cycle");
                // Streaming scenario
                // Each iteration is single execution of Process() method

                if (this->generationHandle->get_status() == GenerationStatus::RUNNING || this->generationHandle->can_read()) {
                    // Subsequent iteration
                    OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
                    GenerationOutputs generationOutputs = this->generationHandle->read();
                    RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
                    RET_CHECK(generationOutputs.begin()->second.generated_token_ids.size() == 1);

                    // TODO(dkalinow): Move this logic to CB library
                    int64_t token = generationOutputs.begin()->second.generated_token_ids[0];
                    auto chunk = this->streamer->put(token);
                    if (chunk.has_value()) {
                        std::string response = packIntoServerSideEventMessage(
                            serializeStreamingChunk(chunk.value(), false, this->request->getEndpoint()));
                        LOG(INFO) << "Partial response (continue): " << response;
                        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                    }
                    // Continue the loop
                    cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);

                } else {
                    OVMS_PROFILE_SCOPE("Generation of last streaming response");
                    std::string response = packIntoServerSideEventMessage(serializeStreamingChunk("", true, this->request->getEndpoint()));
                    response += packIntoServerSideEventMessage("[DONE]");
                    LOG(INFO) << "Partial response (generation finished): " << response;
                    // Produce last message, but do not produce loopback packets anymore so this is last Process() call
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                }
            }
        } catch (ov::AssertFailure& e) {
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            return absl::InvalidArgumentError("Response generation failed");
        }
        timestamp = timestamp.NextAllowedInStream();

        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

std::string HttpLLMCalculator::serializeUnaryResponse(const std::string& completeResponse, Endpoint endpoint) {
    return serializeUnaryResponse(std::vector<std::string>{completeResponse}, endpoint);
}

std::string HttpLLMCalculator::serializeUnaryResponse(const std::vector<std::string>& completeResponses, Endpoint endpoint) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int i = 0;
    for (const std::string& completeResponse : completeResponses) {
        writer.StartObject();  // {
        // finish_reason: string; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)
        // "stop" => natural stop point due to stopping criteria <---------------- the only used so far, remaining are TODO
        // "length" => due to reaching max_tokens parameter TODO
        // "content_filter" => when produced restricted output
        // "tool_calls" => generation stopped and waiting for tool output
        // "function_call" => deprecated
        writer.String("finish_reason");
        writer.String("stop");
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

std::string HttpLLMCalculator::serializeStreamingChunk(const std::string& chunkResponse, bool stop, Endpoint endpoint) {
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
    if (stop)
        writer.String("stop");
    else
        writer.Null();
    // index: integer; Choice index, only n=1 supported anyway
    writer.String("index");
    writer.Int(0);
    // logprobs: object/null; Log probability information for the choice. TODO
    writer.String("logprobs");
    writer.Null();
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("delta");
        writer.StartObject();  // {
        if (!stop) {
            writer.String("content");
            // writer.String("role");
            // writer.String("assistant");
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String(chunkResponse.c_str());
        }
        writer.EndObject();  // }
    } else if (endpoint == Endpoint::COMPLETIONS) {
        if (!stop) {
            writer.String("text");
            writer.String(chunkResponse.c_str());
        }
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
