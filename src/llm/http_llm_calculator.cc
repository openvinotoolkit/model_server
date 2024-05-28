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
#include <string>
#include <thread>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <optional>
#include <sstream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include <openvino/openvino.hpp>
#include <continuous_batching_pipeline.hpp>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "llmnoderesources.hpp"
#include "http_payload.hpp"

using namespace rapidjson;

namespace mediapipe {

using chat_entry_t = std::unordered_map<std::string, std::string>;
using chat_t = std::vector<chat_entry_t>;

class OpenAIChatCompletionsRequest {
    Document& doc;

    chat_t messages;
    bool stream{false};
    std::string model;
    float frequencyPenalty{0.0f};
    std::optional<int64_t> maxTokens{std::nullopt};
    float presencePenalty{0.0f};
    float temperature{1.0f};
    float topP{1.0f};
public:
    OpenAIChatCompletionsRequest(Document& doc) : doc(doc) {}

    chat_t                          getMessages()           const { return this->messages;          }
    bool                            isStream()              const { return this->stream;            }
    std::string                     getModel()              const { return this->model;             }
    float                           getFrequencyPenalty()   const { return this->frequencyPenalty;  }
    const std::optional<int64_t>&   getMaxTokens()          const { return this->maxTokens;         }
    float                           getPresencePenalty()    const { return this->presencePenalty;   }
    float                           getTemperature()        const { return this->temperature;       }
    float                           getTopP()               const { return this->topP;              }
// getTopK ?
// 

    // TODO: Use exceptions to sneak error mesages into response
    bool parse() {
        // stream: bool; optional
        if (!this->doc.IsObject())
            return false;
        auto it = this->doc.FindMember("stream");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsBool())
                return false;
            this->stream = it->value.GetBool();
        }

        // messages: [{role: content}, {role: content}, ...]; required
        it = doc.FindMember("messages");
        if (it == doc.MemberEnd())
            return false;
        if (!it->value.IsArray())
            return false;
        this->messages.clear();
        this->messages.reserve(it->value.GetArray().Size());
        for (int i = 0; i < it->value.GetArray().Size(); i++) {
            const auto& obj = it->value.GetArray()[i];
            if (!obj.IsObject())
                return false;
            auto& chat = this->messages.emplace_back(chat_entry_t{});
            for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
                if (!member->name.IsString())
                    return false;
                if (!member->value.IsString())
                    return false;
                chat[member->name.GetString()] = member->value.GetString();
            }
        }

        // model: string; required
        it = this->doc.FindMember("model");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsString())
                return false;
            this->model = it->value.GetString();
        } else {
            return false;
        }

        // frequency_penalty: float; optional - defaults to 0
        it = this->doc.FindMember("frequency_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return false;
            this->frequencyPenalty = it->value.GetDouble();
            if (this->frequencyPenalty < -2.0f || this->frequencyPenalty > 2.0f)
                return false;
        }

        // max_tokens: int; optional
        it = this->doc.FindMember("max_tokens");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsInt())
                return false;
            this->maxTokens = it->value.GetInt();
            if (this->maxTokens.value() <= 0)
                return false;
        }

        // presence_penalty: float; optional - defaults to 0
        it = this->doc.FindMember("presence_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return false;
            this->presencePenalty = it->value.GetDouble();
            if (this->presencePenalty < -2.0f || this->presencePenalty > 2.0f)
                return false;
        }

        // temperature: float; optional - defaults to 1
        it = this->doc.FindMember("temperature");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return false;
            this->temperature = it->value.GetDouble();
            if (this->temperature < 0.0f || this->temperature > 2.0f)
                return false;
        }

        // top_p: float; optional - defaults to 1
        it = this->doc.FindMember("top_p");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return false;
            this->topP = it->value.GetDouble();
            if (this->topP < 0.0f || this->topP > 1.0f)
                return false;
        }

        // logit_bias TODO
        // logprops TODO
        // top_logprobs TODO
        // n TODO
        // response_format TODO
        // seed TODO
        // stop TODO
        // stream_options TODO
        // tools TODO
        // tool_choice TODO
        // user TODO
        // function_call TODO (deprecated)
        // functions TODO (deprecated)
        return true;
    }
};

// TODO: To be moved to CB library.
class TextStreamer {
    std::shared_ptr<Tokenizer> tokenizer;
    std::vector<int64_t> tokenCache;
    size_t printLen{0};
public:
    TextStreamer(std::shared_ptr<Tokenizer> tokenizer) : tokenizer(tokenizer) {}

    std::optional<std::string> put(int64_t token) {
        tokenCache.push_back(token);
        std::string text = tokenizer->decode(tokenCache);

        if (!text.empty() && '\n' == text.back()) {
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            tokenCache.clear();
            printLen = 0;
            return chunk;
        }
        else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            return std::nullopt;
        } else {
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            printLen = text.size();
            return chunk;
        }
        return std::nullopt;
    }
};

using InputDataType = ovms::HttpPayload;
using OutputDataType = std::string;

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

static std::string packIntoServerSideEventMessage(const std::string& message);

class HttpLLMCalculator : public CalculatorBase {
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    GenerationHandle generationHandle;
    std::shared_ptr<OpenAIChatCompletionsRequest> request;

    ////////////////
    // TODO: To be  moved to CB library
    // Streaming related
    std::stringstream res;
    std::vector<int64_t> tokenCache;
    int64_t printLen = 0;
    std::shared_ptr<TextStreamer> streamer;
    ////////////////

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::chrono::time_point<std::chrono::system_clock> created;

    std::string serializeUnaryResponse(const std::string& completeResponse);
    std::string serializeStreamingChunk(const std::string& chunkResponse, bool stop);

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
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        RET_CHECK(it != nodeResourcesMap.end()) << "Could not find initialized LLM node named: " << cc->NodeName();

        nodeResources = it->second;
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    // greedy (no params)
    // beam-search (num groups, num beams, penalties)
    // random sampling - in-progress (top-p top-k)
    absl::Status Process(CalculatorContext* cc) final {
        RET_CHECK(this->nodeResources != nullptr);

        // For cases where MediaPipe decides to trigger Process() when there are no inputs
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }

        // First iteration of Process()
        if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            // Check if we did not receive the payload twice
            RET_CHECK(this->request == nullptr);
            RET_CHECK(this->generationHandle == nullptr);
            RET_CHECK(this->streamer == nullptr);

            // Register resource creation time
            this->created = std::chrono::system_clock::now();

            InputDataType payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
            LOG(INFO) << "HTTP Request: \n" << payload.body << "\n";
    
            this->request = std::make_shared<OpenAIChatCompletionsRequest>(*payload.parsedJson);

            // TODO: Support chat scenario once atobisze adds that to CB library
            RET_CHECK(this->request->parse());  // TODO: try catch and expose error message
            RET_CHECK(this->request->getMessages().size() >= 1);
            RET_CHECK(this->request->getMessages()[0].count("content") >= 1);
    
            std::string prompt = this->request->getMessages()[0]["content"];

            // TODO: Support other samplings
            // TODO: Pass more params from request
            GenerationConfig config = GenerationConfig::greedy();
            config.max_new_tokens = this->request->getMaxTokens().value_or(30);
            config.ignore_eos = false;
            this->generationHandle = nodeResources->cbPipe->add_request(
                    0/*to be removed from API?*/,
                    prompt/* to be replaced with chat*/,
                    config);
            nodeResources->notifyExecutorThread();
            this->streamer = std::make_shared<TextStreamer>(
                nodeResources->cbPipe->get_tokenizer());
        }

        RET_CHECK(this->generationHandle != nullptr);
        RET_CHECK(this->request != nullptr);
        RET_CHECK(this->streamer != nullptr);

        // Unary scenario
        if (!this->request->isStream()) {
            std::vector<GenerationOutput> generationOutput = this->generationHandle->read_all();
            RET_CHECK(generationOutput.size() == 1);

            std::vector<int64_t> tokens = generationOutput[0].generated_token_ids;
            std::shared_ptr<Tokenizer> tokenizer = nodeResources->cbPipe->get_tokenizer();
            std::string completion = tokenizer->decode(tokens);

            std::string response = serializeUnaryResponse(tokenizer->decode(tokens));
            //LOG(INFO) << response;

            cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
        } else {
            // Streaming scenario
            // Each iteration is single execution of Process() method

            // Last iteration
            if (this->generationHandle->generation_finished()) {
                std::string response = packIntoServerSideEventMessage(serializeStreamingChunk("", true));
                response += packIntoServerSideEventMessage("[DONE]");
                //LOG(INFO) << response;
                // Produce last message, but do not producce loopback packets anymore so this is last Process() call
                cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
            } else {
                // Subsequent iteration
                GenerationOutputs generationOutputs = this->generationHandle->read();
                RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
                RET_CHECK(generationOutputs.begin()->second.generated_token_ids.size() == 1);

                // TODO(dkalinow): Move this logic to CB library
                int64_t token = generationOutputs.begin()->second.generated_token_ids[0];
                auto chunk = this->streamer->put(token);
                if (chunk.has_value()) {
                    std::string response = packIntoServerSideEventMessage(
                        serializeStreamingChunk(chunk.value(), false));
                    //LOG(INFO) << response;
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{response}, timestamp);
                }
                // Continue the loop
                cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);
            }
        }

        timestamp = timestamp.NextAllowedInStream();

        return absl::OkStatus();
    }
};

// TODO: Different samplings may produce multiple generations.
// Consider switching from string to vector<string>?
std::string HttpLLMCalculator::serializeUnaryResponse(const std::string& completeResponse) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
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
    writer.Int(0);
    // logprobs: object/null; Log probability information for the choice. TODO
    writer.String("logprobs");
    writer.Null();
    // message: object
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
                
    writer.EndObject();  // }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(this->created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(this->request->getModel().c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    writer.String("object");
    writer.String("chat.completion");

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

std::string HttpLLMCalculator::serializeStreamingChunk(const std::string& chunkResponse, bool stop) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    // Can also be empty for the last chunk if you set stream_options: {"include_usage": true} TODO
    writer.String("choices");
    writer.StartArray();  // [
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
    // delta: object
    writer.String("delta");
    writer.StartObject();  // {
    // content: string; Actual content of the text produced
    if (!stop) {
        writer.String("content");
        writer.String(chunkResponse.c_str());
    }
    // role: string; Role of the text producer
    // Will make sense once we have chat templates? TODO(atobisze)
    // writer.String("role");
    // writer.String("assistant");
    // TODO: tools_call
    // TODO: function_call (deprecated)
    writer.EndObject();  // }
                
    writer.EndObject();  // }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(this->created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(this->request->getModel().c_str());

    // object: string; defined that the type streamed chunk rather than complete response
    writer.String("object");
    writer.String("chat.completion.chunk");

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

std::string packIntoServerSideEventMessage(const std::string& message) {
    std::stringstream ss;
    ss << "data: " << message << "\n\n";
    return ss.str();
}

// TODO: Names to be decided
const std::string HttpLLMCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string HttpLLMCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string HttpLLMCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(HttpLLMCalculator);
}  // namespace mediapipe
