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

#include <opencv2/opencv.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"

using namespace rapidjson;

namespace ovms {

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
    if (it != doc.MemberEnd()) {
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

static ov::element::Type_t getOvTypeFromMatType(int matType) {
    switch (matType) {
    case CV_32F:
        return ov::element::f32;
    case CV_64F:
        return ov::element::f64;
    case CV_16F:
        return ov::element::f64;
    case CV_16S:
        return ov::element::f16;
    case CV_8U:
        return ov::element::u8;
    case CV_8S:
        return ov::element::i8;
    case CV_16U:
        return ov::element::u16;
    case CV_32S:
        return ov::element::i32;
    default:
        return ov::element::undefined;
    }
}

absl::Status OpenAIChatCompletionsHandler::parseMessages() {
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
        for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
            if (!member->name.IsString())
                return absl::InvalidArgumentError("Invalid message structure");
            if (member->value.IsString()) {
                continue;
            } else {
                if (member->name.GetString() == std::string("content") && member->value.IsArray()) {
                    if (member->value.GetArray().Size() == 0) {
                        return absl::InvalidArgumentError("Invalid message structure - content array is empty");
                    }
                    jsonChanged = true;
                    Value contentText;
                    for (auto& v : member->value.GetArray()) {
                        if (!v.IsObject()) {
                            return absl::InvalidArgumentError("Invalid message structure - content array should contain objects");
                        }
                        auto entry = v.GetObject();
                        if (!entry.HasMember("type") || !entry["type"].IsString()) {
                            return absl::InvalidArgumentError("Invalid message structure - content object type missing");
                        }
                        auto type = entry["type"].GetString();
                        if (type == std::string("text")) {
                            if (!entry.HasMember("text") || !entry["text"].IsString()) {
                                return absl::InvalidArgumentError("Invalid message structure - content text missing");
                            }
                            contentText = entry["text"];
                            continue;
                        } else if (type == std::string("image_url")) {
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
                            size_t offset = pos + pattern.length();
                            if (pos == std::string::npos) {
                                return absl::InvalidArgumentError("Url should contain base64 encoded string followed by \"base64,\" prefix");
                            }
                            std::string decoded;
                            if (!absl::Base64Unescape(std::string_view(url.data() + offset, url.size() - offset), &decoded)) {
                                return absl::InvalidArgumentError("Invalid base64 string in request");
                            }
                            size_t rows = 1;
                            size_t cols = decoded.size();
                            cv::Mat rawData(rows, cols, CV_8UC1, (void*)decoded.data());
                            cv::Mat image;
                            try {
                                image = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
                            } catch (const cv::Exception& e) {
                                return absl::InvalidArgumentError("Error during string to mat conversion");
                            }
                            std::vector<size_t> shape;
                            shape.push_back(image.rows);
                            shape.push_back(image.cols);
                            shape.push_back(image.channels());
                            auto type = getOvTypeFromMatType(image.depth());
                            if (type == ov::element::undefined) {
                                return absl::InvalidArgumentError("Image type is invalid");
                            }
                            ov::Tensor tensor(type, shape);
                            if (image.total() * image.elemSize() != tensor.get_size()) {
                                return absl::InvalidArgumentError("Image size invalid");
                            }
                            memcpy((char*)tensor.data(), (char*)image.data, image.total() * image.elemSize());
                            request.images.push_back(tensor);
                        } else {
                            return absl::InvalidArgumentError("Unsupported content type");
                        }
                    }
                    member->value = contentText;
                } else {
                    return absl::InvalidArgumentError("Invalid message structure - content should be string or array");
                }
            }
        }
    }
    if (jsonChanged) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);
        request.processedJson = buffer.GetString();
    }
    return absl::OkStatus();
}

const std::string& OpenAIChatCompletionsHandler::getProcessedJson() const {
    return request.processedJson;
}
const std::vector<ov::Tensor> OpenAIChatCompletionsHandler::getImages() const {
    return request.images;
}

absl::Status OpenAIChatCompletionsHandler::parseChatCompletionsPart(uint32_t maxTokensLimit) {
    // messages: [{role: content}, {role: content}, ...]; required
    auto status = parseMessages();
    if (status != absl::OkStatus()) {
        return status;
    }
    // logprobs: bool; optional - defaults to false
    auto it = doc.FindMember("logprobs");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("logprobs accepts values true or false");
        request.logprobschat = it->value.GetBool();
    }
    if (request.logprobschat && request.stream) {
        return absl::InvalidArgumentError("logprobs are not supported in streaming mode.");
    }
    // max_completion_tokens: uint; optional
    it = doc.FindMember("max_completion_tokens");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_completion_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_completion_tokens is not an unsigned integer");
        }
        if (!(it->value.GetUint() < maxTokensLimit))
            return absl::InvalidArgumentError(absl::StrCat("max_completion_tokens exceeds limit provided in graph config: ", maxTokensLimit));
        if (request.ignoreEOS.value_or(false)) {
            if (it->value.GetUint() > IGNORE_EOS_MAX_TOKENS_LIMIT)
                return absl::InvalidArgumentError("when ignore_eos is true max_completion_tokens can not be greater than 4000");
        }
        request.maxTokens = it->value.GetUint();
    }
    // specific part of max_tokens validation due to echo dependency
    if (request.maxTokens == 0) {
        return absl::InvalidArgumentError("max_tokens value should be greater than 0");
    }

    return absl::OkStatus();
}

absl::Status OpenAIChatCompletionsHandler::parseCommonPart(uint32_t maxTokensLimit, uint32_t bestOfLimit) {
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
    // Common part checked here, specific parts are checked in parseCompletionsPart and parseChatCompletionsPart
    // Deprecated for chat completions TODO move to parseCompletionsPart
    it = doc.FindMember("max_tokens");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsUint()) {
            if (it->value.IsUint64())
                return absl::InvalidArgumentError("max_tokens value can't be greater than 4294967295");
            return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
        }
        if (!(it->value.GetUint() < maxTokensLimit))
            return absl::InvalidArgumentError(absl::StrCat("max_tokens exceeds limit provided in graph config: ", maxTokensLimit));
        request.maxTokens = it->value.GetUint();
    }
    if (request.ignoreEOS.value_or(false)) {
        if (request.maxTokens.has_value()) {
            if (request.maxTokens.value() > IGNORE_EOS_MAX_TOKENS_LIMIT)
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

    // temperature: float; optional - defaults to 1.0
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
            if (stopArray.Size() > 4)
                return absl::InvalidArgumentError("stop array must have no more than 4 strings");
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
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("include_stop_str_in_output accepts values true or false");
        if (!it->value.GetBool() && request.stream)
            return absl::InvalidArgumentError("include_stop_str_in_output cannot be set to false if streaming is used");
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
        if (request.stream)
            return absl::InvalidArgumentError("best_of cannot be used in streaming mode");
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

void OpenAIChatCompletionsHandler::incrementProcessedTokens(int numTokens) {
    processedTokens += numTokens;
    if (!request.echo || processedTokens > usage.promptTokens)
        usage.completionTokens += numTokens;
}

ov::genai::GenerationConfig OpenAIChatCompletionsHandler::createGenerationConfig() const {
    return request.createGenerationConfig();
}

absl::Status OpenAIChatCompletionsHandler::parseRequest(uint32_t maxTokensLimit, uint32_t bestOfLimit) {
    absl::Status status = parseCommonPart(maxTokensLimit, bestOfLimit);

    if (status != absl::OkStatus())
        return status;

    if (endpoint == Endpoint::COMPLETIONS)
        status = parseCompletionsPart();
    else
        status = parseChatCompletionsPart(maxTokensLimit);

    return status;
}

std::string OpenAIChatCompletionsHandler::serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int index = 0;
    usage.completionTokens = 0;
    for (const ov::genai::GenerationOutput& generationOutput : generationOutputs) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", generationOutput.generated_ids);
        usage.completionTokens += generationOutput.generated_ids.size();
        if (request.echo)
            usage.completionTokens -= usage.promptTokens;
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
        writer.Int(index++);
        // logprobs: object/null; Log probability information for the choice. TODO
        writer.String("logprobs");
        if (this->request.logprobschat || this->request.logprobs > 0) {
            if (endpoint == Endpoint::CHAT_COMPLETIONS) {
                writer.StartObject();  // {
                writer.String("content");
                writer.StartArray();  // [

                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    writer.StartObject();  // {

                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String("token");
                    writer.String(token.c_str());

                    float logprob = generationOutput.generated_log_probs[i];
                    writer.String("logprob");
                    writeLogprob(writer, logprob);
                    writer.String("bytes");
                    writer.StartArray();  // [
                    // Assuming tokenizer returned UTF-8 encoded string
                    const unsigned char* tokenBytes = reinterpret_cast<const unsigned char*>(token.c_str());
                    for (int j = 0; tokenBytes[j] != 0; j++)
                        writer.Int(tokenBytes[j]);
                    writer.EndArray();  // ]

                    // top_logprobs are currently hardcoded to return empty array to comply with the API
                    // for full support significant changes on GenAI side are required
                    writer.String("top_logprobs");
                    writer.StartArray();  // [
                                          /*                  
                    Commented out due to supported only top_logprobs 1
                    writer.StartObject();  // {

                    writer.String("token");
                    writer.String(token.c_str());

                    writer.String("logprob");
                    writeLogprob(writer, logprob);
                    writer.String("bytes");
                    writer.StartArray();  // [
                    for (int j = 0; tokenBytes[j] != 0; j++)
                        writer.Int(tokenBytes[j]);
                    writer.EndArray();  // ]

                    writer.EndObject();  // } */
                    writer.EndArray();    // ]

                    writer.EndObject();  // }
                }
                writer.EndArray();   // ]
                writer.EndObject();  // }
            }
            if (endpoint == Endpoint::COMPLETIONS) {
                writer.StartObject();  // {
                writer.String("tokens");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String(token.c_str());
                }
                writer.EndArray();  // ]

                writer.String("token_logprobs");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    float logprob = generationOutput.generated_log_probs[i];
                    writeLogprob(writer, logprob);
                }
                writer.EndArray();  // ]

                writer.String("top_logprobs");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    writer.StartObject();  // {
                    std::string token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids[i]}));
                    writer.String(token.c_str());
                    float logprob = generationOutput.generated_log_probs[i];
                    writeLogprob(writer, logprob);
                    writer.EndObject();  // }
                }
                writer.EndArray();  // ]

                writer.String("text_offset");
                writer.StartArray();  // [
                for (int i = 0; i < generationOutput.generated_ids.size(); i++) {
                    if (i == 0) {
                        writer.Int(0);
                    } else {
                        std::string text_before_token = tokenizer.decode(std::vector<int64_t>({generationOutput.generated_ids.begin(), generationOutput.generated_ids.begin() + i}));
                        writer.Int(text_before_token.size());
                    }
                }
                writer.EndArray();   // ]
                writer.EndObject();  // }
            }
        } else {
            writer.Null();
        }
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

void OpenAIChatCompletionsHandler::writeLogprob(Writer<StringBuffer>& writer, float logprob) {
    // genai returns logaritm of probability per token which should be in the range of -inf-0
    // other values could be potentially invalid and should be treated as such
    if (logprob <= 0.0)
        writer.Double(logprob);
    else
        writer.Null();
}
}  // namespace ovms
