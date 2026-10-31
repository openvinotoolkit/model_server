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

#include "openai_decisions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>

#include "openai_json_response.hpp"

#include "../../logging.hpp"
#include "../../profiler.hpp"
#include "../io_processing/video_utils.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/str_cat.h"
#pragma warning(pop)

using namespace rapidjson;

namespace ovms {

namespace {

constexpr size_t MAX_QUESTIONS = 64;
constexpr size_t MAX_CHOICES = 26;
constexpr size_t MAX_SCORE_LEVELS = 10;

const char* questionTypeName(OpenAIDecisionsHandler::QuestionType type) {
    switch (type) {
    case OpenAIDecisionsHandler::QuestionType::PREDICATE:
        return "predicate";
    case OpenAIDecisionsHandler::QuestionType::CHOICE:
        return "choice";
    case OpenAIDecisionsHandler::QuestionType::SCORE:
        return "score";
    }
    return "unknown";
}

}  // namespace

absl::Status OpenAIDecisionsHandler::parseRequestImpl(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) {
    (void)maxTokensLimit;
    (void)bestOfLimit;
    (void)maxModelLength;
    (void)allowedLocalMediaPath;
    (void)allowedMediaDomains;

    if (!doc.IsObject())
        return absl::InvalidArgumentError("Received json is not an object");

    // model: string; required
    auto it = doc.FindMember("model");
    if (it == doc.MemberEnd())
        return absl::InvalidArgumentError("model missing in request");
    if (!it->value.IsString())
        return absl::InvalidArgumentError("model is not a string");
    request.model = it->value.GetString();

    // vLLM/OpenAI contract: unknown top-level fields are rejected with 400. Sampling is
    // forced by the endpoint (single greedy token) and cannot be overridden, so only the
    // fields below are accepted; anything else (temperature, tools, logit_bias, stream,
    // chat_template_kwargs, ...) is rejected.
    static const std::set<std::string> allowedParams = {
        "model", "input", "questions", "safety_identifier", "joint", "cache_salt"};
    for (auto m = doc.MemberBegin(); m != doc.MemberEnd(); ++m) {
        if (!allowedParams.count(m->name.GetString()))
            return absl::InvalidArgumentError(absl::StrCat("parameter '", m->name.GetString(), "' is not supported for /v1/decisions"));
    }

    auto status = parseInput();
    if (!status.ok())
        return status;

    status = parseQuestions();
    if (!status.ok())
        return status;

    // safety_identifier: string; optional (accepted for API compatibility, not used)
    it = doc.FindMember("safety_identifier");
    if (it != doc.MemberEnd() && !it->value.IsNull() && !it->value.IsString())
        return absl::InvalidArgumentError("safety_identifier is not a string");

    // joint: bool, default false (vLLM accepts the field; true is diffusion-only)
    it = doc.FindMember("joint");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("joint is not a boolean");
        if (it->value.GetBool())
            return absl::InvalidArgumentError("joint reads are not supported for /v1/decisions");
    }

    // cache_salt: string 1-1024 (accepted for API compatibility, not used)
    it = doc.FindMember("cache_salt");
    if (it != doc.MemberEnd() && !it->value.IsNull()) {
        if (!it->value.IsString() || it->value.GetStringLength() < 1 || it->value.GetStringLength() > 1024)
            return absl::InvalidArgumentError("cache_salt must be a string of 1 to 1024 characters");
    }

    // Each question generates exactly one token; setting maxTokens lets
    // prepareInputs run the prompt + maxTokens <= maxModelLength length check.
    request.maxTokens = 1;

    return encodeLabels();
}

absl::Status OpenAIDecisionsHandler::parseInput() {
    auto it = doc.FindMember("input");
    if (it == doc.MemberEnd())
        return absl::InvalidArgumentError("input missing in request");
    if (it->value.IsString()) {
        if (it->value.GetStringLength() == 0)
            return absl::InvalidArgumentError("input string cannot be empty");
        // Normalize a string input to a single user message so every decisions
        // input goes through the chat template. The raw-prompt path is not used:
        // on VLM servables it skips TokenizationProcessor, leaving inputIds empty
        // and crashing prepareInputs.
        request.chatHistory.push_back({{"role", "user"}, {"content", std::string(it->value.GetString())}});
        return absl::OkStatus();
    }
    if (!it->value.IsArray())
        return absl::InvalidArgumentError("input must be a string or an array of messages");
    if (it->value.Empty())
        return absl::InvalidArgumentError("input messages array cannot be empty");
    // OpenAI/vLLM contract: every input message is a user message. Content is a
    // string or an array of parts: text (input_text/text), image_url, video_url, and
    // the OpenAI decisions image part (input_image). Media parts are preserved for the
    // VLM input processors (ImageDecodingProcessor, VideoFramesProcessor); LLM servables
    // reject them in validateInputCompatibility.
    for (auto message = it->value.GetArray().Begin(); message != it->value.GetArray().End(); ++message) {
        if (!message->IsObject())
            return absl::InvalidArgumentError("input message is not a JSON object");
        bool hasRole = false;
        bool hasContent = false;
        for (auto field = message->MemberBegin(); field != message->MemberEnd(); ++field) {
            if (!field->name.IsString())
                return absl::InvalidArgumentError("invalid input message structure");
            std::string fieldName = field->name.GetString();
            if (fieldName == "role") {
                if (!field->value.IsString() || std::string(field->value.GetString()) != "user")
                    return absl::InvalidArgumentError("input message role must be 'user'");
                hasRole = true;
            } else if (fieldName == "type") {
                // The OpenAI decisions schema allows an explicit "message" type tag.
                if (!field->value.IsString() || std::string(field->value.GetString()) != "message")
                    return absl::InvalidArgumentError("input message type must be 'message'");
            } else if (fieldName == "content") {
                if (field->value.IsString()) {
                    if (field->value.GetStringLength() == 0)
                        return absl::InvalidArgumentError("input message content cannot be empty");
                    request.chatHistory.push_back({{"role", "user"}, {"content", std::string(field->value.GetString())}});
                    hasContent = true;
                } else if (field->value.IsArray()) {
                    if (field->value.Empty())
                        return absl::InvalidArgumentError("input message content cannot be empty");
                    std::string textContent;
                    bool hasMedia = false;
                    for (const auto& part : field->value.GetArray()) {
                        if (!part.IsObject())
                            return absl::InvalidArgumentError("content array should contain objects");
                        if (!part.HasMember("type") || !part["type"].IsString())
                            return absl::InvalidArgumentError("content object type missing");
                        const std::string partType = part["type"].GetString();
                        if (partType == "input_text" || partType == "text") {
                            if (!part.HasMember("text") || !part["text"].IsString())
                                return absl::InvalidArgumentError("content text missing");
                            textContent += part["text"].GetString();
                        } else if (partType == "image_url") {
                            if (!part.HasMember("image_url") || !part["image_url"].IsObject())
                                return absl::InvalidArgumentError("content image_url missing");
                            const auto imageUrl = part["image_url"].GetObject();
                            if (!imageUrl.HasMember("url") || !imageUrl["url"].IsString())
                                return absl::InvalidArgumentError("image_url does not have url field");
                            hasMedia = true;
                        } else if (partType == "video_url") {
                            if (!part.HasMember("video_url") || !part["video_url"].IsObject())
                                return absl::InvalidArgumentError("content video_url missing");
                            const auto videoUrl = part["video_url"].GetObject();
                            if (!videoUrl.HasMember("url") || !videoUrl["url"].IsArray())
                                return absl::InvalidArgumentError("video_url does not have url array field");
                            const auto frames = videoUrl["url"].GetArray();
                            if (frames.Size() == 0)
                                return absl::InvalidArgumentError("video_url url array cannot be empty");
                            if (static_cast<int64_t>(frames.Size()) > MAX_VIDEO_FRAMES)
                                return absl::InvalidArgumentError("video_url url array exceeds the allowed maximum of " + std::to_string(MAX_VIDEO_FRAMES) + " frames");
                            for (const auto& frame : frames)
                                if (!frame.IsString())
                                    return absl::InvalidArgumentError("video_url url array must contain only strings");
                            hasMedia = true;
                        } else if (partType == "input_image") {
                            // OpenAI decisions image part: {"type": "input_image",
                            // "image_url": "<http(s) url | base64 data url>"}; normalized to
                            // the internal image_url object consumed by the input processors.
                            if (!part.HasMember("image_url") || !part["image_url"].IsString())
                                return absl::InvalidArgumentError("input_image does not have an image_url string field");
                            if (part["image_url"].GetStringLength() == 0)
                                return absl::InvalidArgumentError("input_image image_url cannot be empty");
                            hasMedia = true;
                        } else {
                            return absl::InvalidArgumentError("unsupported content type: " + partType);
                        }
                    }
                    if (hasMedia) {
                        // Multimodal content: preserve the parts for the VLM input
                        // processors (mirrors OpenAIChatCompletionsHandler::parseMessages);
                        // input_image parts are normalized to image_url objects.
                        ov::genai::JsonContainer contentParts = ov::genai::JsonContainer::array();
                        for (const auto& part : field->value.GetArray()) {
                            if (std::string(part["type"].GetString()) == "input_image") {
                                contentParts.push_back(ov::genai::JsonContainer({{"type", "image_url"}, {"image_url", ov::genai::JsonContainer({{"url", std::string(part["image_url"].GetString())}})}}));
                            } else {
                                contentParts.push_back(rapidJsonValueToJsonContainer(part));
                            }
                        }
                        request.chatHistory.push_back({{"role", "user"}, {"content", std::move(contentParts)}});
                    } else {
                        if (textContent.empty())
                            return absl::InvalidArgumentError("input message content cannot be empty");
                        request.chatHistory.push_back({{"role", "user"}, {"content", textContent}});
                    }
                    hasContent = true;
                } else {
                    return absl::InvalidArgumentError("input message content must be a string or an array of parts");
                }
            } else {
                return absl::InvalidArgumentError("unsupported field in input message: " + fieldName);
            }
        }
        if (!hasRole || !hasContent)
            return absl::InvalidArgumentError("input message must contain non-empty 'role' and 'content' fields");
    }
    return absl::OkStatus();
}

absl::Status OpenAIDecisionsHandler::parseQuestions() {
    auto it = doc.FindMember("questions");
    if (it == doc.MemberEnd())
        return absl::InvalidArgumentError("questions missing in request");
    if (!it->value.IsArray())
        return absl::InvalidArgumentError("questions is not an array");
    if (it->value.Empty())
        return absl::InvalidArgumentError("questions array cannot be empty");
    if (it->value.Size() > MAX_QUESTIONS)
        return absl::InvalidArgumentError(absl::StrCat("at most ", MAX_QUESTIONS, " questions are supported"));

    for (auto questionIt = it->value.GetArray().Begin(); questionIt != it->value.GetArray().End(); ++questionIt) {
        if (!questionIt->IsObject())
            return absl::InvalidArgumentError("question is not a JSON object");
        Question question;

        auto typeIt = questionIt->FindMember("type");
        if (typeIt == questionIt->MemberEnd() || !typeIt->value.IsString())
            return absl::InvalidArgumentError("question.type is missing or not a string");
        std::string type = typeIt->value.GetString();
        if (type == "predicate") {
            question.type = QuestionType::PREDICATE;
        } else if (type == "choice") {
            question.type = QuestionType::CHOICE;
        } else if (type == "score") {
            question.type = QuestionType::SCORE;
        } else {
            return absl::InvalidArgumentError("question.type must be one of: predicate, choice, score");
        }

        auto instructionsIt = questionIt->FindMember("instructions");
        if (instructionsIt == questionIt->MemberEnd() || !instructionsIt->value.IsString() || instructionsIt->value.GetStringLength() == 0)
            return absl::InvalidArgumentError("question.instructions is missing or not a non-empty string");
        question.instructions = instructionsIt->value.GetString();

        auto nameIt = questionIt->FindMember("name");
        if (nameIt != questionIt->MemberEnd()) {
            if (!nameIt->value.IsString())
                return absl::InvalidArgumentError("question.name must be a string when provided");
            question.name = nameIt->value.GetString();
        }

        // Reject duplicate option values within a single question: two options
        // sharing a value would push the label mass above 1 (the same label token
        // is read twice). The same value across DIFFERENT questions is fine
        // (each question reads its own labels).
        std::set<std::string> seenValues;

        if (question.type == QuestionType::PREDICATE) {
            question.values = {"yes", "no"};
            question.labels = {"yes", "no"};
        } else if (question.type == QuestionType::CHOICE) {
            auto choicesIt = questionIt->FindMember("choices");
            if (choicesIt == questionIt->MemberEnd() || !choicesIt->value.IsArray())
                return absl::InvalidArgumentError("question.choices is missing or not an array");
            if (choicesIt->value.Size() < 2)
                return absl::InvalidArgumentError("question.choices must contain at least 2 choices");
            if (choicesIt->value.Size() > MAX_CHOICES)
                return absl::InvalidArgumentError(absl::StrCat("at most ", MAX_CHOICES, " choices are supported"));
            for (auto choiceIt = choicesIt->value.GetArray().Begin(); choiceIt != choicesIt->value.GetArray().End(); ++choiceIt) {
                if (!choiceIt->IsObject())
                    return absl::InvalidArgumentError("choice is not a JSON object");
                auto valueIt = choiceIt->FindMember("value");
                if (valueIt == choiceIt->MemberEnd())
                    return absl::InvalidArgumentError("choice.value is missing");
                bool isBool = false;
                std::string value;
                if (valueIt->value.IsString()) {
                    if (valueIt->value.GetStringLength() == 0)
                        return absl::InvalidArgumentError("choice.value must be a non-empty string or a boolean");
                    value = valueIt->value.GetString();
                } else if (valueIt->value.IsBool()) {
                    isBool = true;
                    value = valueIt->value.GetBool() ? "true" : "false";
                } else {
                    return absl::InvalidArgumentError("choice.value must be a non-empty string or a boolean");
                }
                // The bool prefix keeps "true" (string) and true (bool) distinct.
                if (!seenValues.insert((isBool ? "b" : "s") + value).second)
                    return absl::InvalidArgumentError(absl::StrCat("duplicate choice value '", value, "' in question"));
                question.values.push_back(value);
                question.choiceIsBool.push_back(isBool);
                question.labels.push_back(std::string(1, static_cast<char>('A' + question.labels.size())));
            }
        } else {  // SCORE
            auto levelsIt = questionIt->FindMember("levels");
            if (levelsIt == questionIt->MemberEnd() || !levelsIt->value.IsArray())
                return absl::InvalidArgumentError("question.levels is missing or not an array");
            if (levelsIt->value.Size() < 2)
                return absl::InvalidArgumentError("question.levels must contain at least 2 levels");
            if (levelsIt->value.Size() > MAX_SCORE_LEVELS)
                return absl::InvalidArgumentError(absl::StrCat("at most ", MAX_SCORE_LEVELS, " levels are supported"));
            for (auto levelIt = levelsIt->value.GetArray().Begin(); levelIt != levelsIt->value.GetArray().End(); ++levelIt) {
                if (!levelIt->IsObject())
                    return absl::InvalidArgumentError("level is not a JSON object");
                auto labelIt = levelIt->FindMember("label");
                if (labelIt == levelIt->MemberEnd() || !labelIt->value.IsString() || labelIt->value.GetStringLength() == 0)
                    return absl::InvalidArgumentError("level.label is missing or not a non-empty string");
                const std::string level = labelIt->value.GetString();
                if (!seenValues.insert(level).second)
                    return absl::InvalidArgumentError(absl::StrCat("duplicate level label '", level, "' in question"));
                question.values.push_back(level);
                question.labels.push_back(std::to_string(question.labels.size()));
            }
        }
        questions.push_back(std::move(question));
    }
    return absl::OkStatus();
}

absl::Status OpenAIDecisionsHandler::encodeLabels() {
    // Labels are the generated single tokens (A..Z / 0..9 / "yes","no"), which are
    // single tokens in standard BPE tokenizers. The check is a safety net; it is on
    // the generated labels (not the user values), so multi-word choice values work.
    // There are at most 38 distinct labels, so encode each one once per request and
    // reuse the token id for the rest.
    std::unordered_map<std::string, int64_t> labelTokenCache;
    for (auto& question : questions) {
        question.labelTokenIds.reserve(question.labels.size());
        for (const auto& label : question.labels) {
            auto cached = labelTokenCache.find(label);
            if (cached != labelTokenCache.end()) {
                question.labelTokenIds.push_back(cached->second);
                continue;
            }
            ov::genai::TokenizedInputs tokens;
            try {
                tokens = tokenizer.encode(label, {ov::genai::add_special_tokens(false)});
            } catch (const std::exception& e) {
                return absl::InvalidArgumentError(absl::StrCat("failed to encode label '", label, "': ", e.what()));
            }
            if (tokens.input_ids.get_size() != 1)
                return absl::InvalidArgumentError(absl::StrCat("label '", label, "' is not a single token"));
            const int64_t tokenId = tokens.input_ids.data<int64_t>()[0];
            labelTokenCache.emplace(label, tokenId);
            question.labelTokenIds.push_back(tokenId);
        }
    }
    return absl::OkStatus();
}

std::string OpenAIDecisionsHandler::buildQuestionBlock(const Question& question) {
    const char* replyInstruction = "Answer with the letter of one option only.";
    if (question.type == QuestionType::SCORE)
        replyInstruction = "Answer with the number of one level only.";
    else if (question.type == QuestionType::PREDICATE)
        replyInstruction = "Answer with yes or no only.";

    // vLLM QuestionType::prompt: "Question: {instructions}\n" + reply lines joined
    // by "\n" with the reply instruction as the final line.
    std::string block = "Question: " + question.instructions + "\n";
    for (size_t i = 0; i < question.labels.size(); ++i) {
        // choice: "A: <value>"; score: "0: <level>"; predicate: bare "yes"/"no".
        if (question.type == QuestionType::PREDICATE)
            block += question.labels[i];
        else
            block += question.labels[i] + ": " + question.values[i];
        block += "\n";
    }
    block += replyInstruction;
    return block;
}

void OpenAIDecisionsHandler::appendQuestionBlock(ov::genai::ChatHistory& history, const std::string& block) {
    // Append to the last message so the block reads as part of the state the model answers
    // (a combined message, not a new turn, so the chat template is not disturbed). If the
    // history is empty or its last message is not a user message (e.g. it ends on an
    // assistant turn), the question cannot be folded into an existing user turn, so a fresh
    // user message is started.
    if (history.size() == 0 || history[history.size() - 1]["role"].get_string() != "user") {
        history.push_back({{"role", "user"}, {"content", block}});
        return;
    }
    // operator[] returns a path handle into `history`; writes go to the stored message, not a copy.
    auto&& msg = history[history.size() - 1];
    auto content = msg["content"];
    if (content.is_string()) {
        msg["content"] = content.get_string() + "\n\n" + block;
    } else if (content.is_array()) {
        // Multimodal content (VLM): add the block as an extra text part.
        msg["content"].push_back(ov::genai::JsonContainer({{"type", "text"}, {"text", "\n\n" + block}}));
    }
}

absl::StatusOr<std::vector<InputRequest>> OpenAIDecisionsHandler::extractInputRequests(GenerationConfigBuilder& configBuilder) {
    auto baseResult = OpenAIApiHandler::extractInputRequest(configBuilder);
    if (!baseResult.ok())
        return baseResult.status();
    InputRequest base = std::move(*baseResult);

    std::vector<InputRequest> requests;
    requests.reserve(questions.size());
    for (const auto& question : questions) {
        InputRequest req = base;  // deep copy: isolates this question's chat history
        appendQuestionBlock(std::get<ov::genai::ChatHistory>(req.input), buildQuestionBlock(question));
        auto& config = req.generationConfig;
        // Decisions always run one greedy forward pass with requested-token logprobs. The
        // config is seeded from the model's generation_config, so clear min_new_tokens: a
        // value > 1 would survive max_new_tokens = 1 and be rejected by GenerationConfig::validate.
        config.max_new_tokens = 1;
        config.min_new_tokens = 0;
        config.temperature = 0.0f;
        config.do_sample = false;
        config.num_beams = 1;
        config.num_return_sequences = 1;
        // Only this question's label tokens: each engine request reads its own labels.
        // Speculative decoding / prompt lookup servables are rejected up front
        // (validateEndpoint), so the assisting-generation fields never reach the pipeline.
        config.logprob_token_ids = question.labelTokenIds;
        requests.push_back(std::move(req));
    }
    return requests;
}

std::string OpenAIDecisionsHandler::serializeUnaryResponse(const std::vector<Delta>& deltas, ov::genai::GenerationFinishReason finishReason) {
    return serializeUnaryResponse({deltas}, {finishReason}, {});
}

absl::Status OpenAIDecisionsHandler::validateLogprobData(const std::vector<UnaryChoiceLogprobs>& logprobData) const {
    if (logprobData.empty() || logprobData.size() != questions.size()) {
        return absl::InternalError("/v1/decisions: incomplete requested-token logprobs for this request");
    }
    for (size_t qi = 0; qi < questions.size(); ++qi) {
        const auto& entry = logprobData[qi];
        if (entry.generatedIds.empty() || entry.requestedTokenLogprobs.empty()) {
            return absl::InternalError(absl::StrCat("/v1/decisions: incomplete requested-token logprobs for question ", qi));
        }
        const auto& positionLogprobs = entry.requestedTokenLogprobs.front();
        double maxLogprob = -std::numeric_limits<double>::infinity();
        for (int64_t labelTokenId : questions[qi].labelTokenIds) {
            bool found = false;
            for (const auto& [tokenId, value] : positionLogprobs) {
                if (tokenId == labelTokenId) {
                    maxLogprob = std::max(maxLogprob, static_cast<double>(value));
                    found = true;
                    break;
                }
            }
            if (!found) {
                return absl::InternalError(absl::StrCat("/v1/decisions: incomplete requested-token logprobs for question ", qi));
            }
        }
        // All labels present but masked with -inf: the classification cannot be computed.
        if (!std::isfinite(maxLogprob)) {
            return absl::InternalError(absl::StrCat("/v1/decisions: incomplete requested-token logprobs for question ", qi));
        }
    }
    return absl::OkStatus();
}

std::string OpenAIDecisionsHandler::serializeUnaryResponse(
    const std::vector<std::vector<Delta>>& allDeltas,
    const std::vector<ov::genai::GenerationFinishReason>& finishReasons,
    const std::vector<UnaryChoiceLogprobs>& logprobData) {
    OVMS_PROFILE_FUNCTION();
    (void)allDeltas;
    (void)finishReasons;

    OpenAiJsonResponse jsonResponse;
    jsonResponse.StartObject();

    jsonResponse.String("model", request.model);

    // One engine request per question: logprobData[i] is question i's result.
    // Defensive only — unreachable via the HTTP paths: the servables call
    // validateLogprobData before serialization and surface a non-OK status (HTTP 400)
    // on incomplete data, and the serializer itself cannot return a status.
    if (logprobData.empty() || logprobData.size() != questions.size()) {
        jsonResponse.String("error", "requested token logprobs are not available for this request");
        jsonResponse.EndObject();
        return jsonResponse.ToString();
    }

    jsonResponse.StartArray("answers");
    for (size_t qi = 0; qi < questions.size(); ++qi) {
        const auto& question = questions[qi];
        const auto& entry = logprobData[qi];
        const bool hasResult = !entry.generatedIds.empty() && !entry.requestedTokenLogprobs.empty();
        const std::vector<std::pair<int64_t, float>> emptyPositionLogprobs;
        const auto& positionLogprobs = hasResult ? entry.requestedTokenLogprobs.front() : emptyPositionLogprobs;

        // Gather the full-vocabulary-normalized logprob of each of this question's labels
        std::vector<double> labelLogprobs;
        labelLogprobs.reserve(question.labels.size());
        double maxLogprob = -std::numeric_limits<double>::infinity();
        bool complete = hasResult;
        for (size_t i = 0; i < question.labelTokenIds.size(); ++i) {
            float logprob = 0.0f;
            bool found = false;
            for (const auto& [tokenId, value] : positionLogprobs) {
                if (tokenId == question.labelTokenIds[i]) {
                    logprob = value;
                    found = true;
                    break;
                }
            }
            if (!found) {
                complete = false;
                break;
            }
            labelLogprobs.push_back(logprob);
            maxLogprob = std::max(maxLogprob, static_cast<double>(logprob));
        }

        jsonResponse.StartObject();
        jsonResponse.String("type", questionTypeName(question.type));
        if (question.name.has_value()) {
            jsonResponse.String("name", question.name.value());
        } else {
            jsonResponse.Null("name");
        }

        // !isfinite(maxLogprob) covers the all-labels-present-but-masked case: a model that
        // masks part of the vocabulary with -inf logits yields maxLogprob == -inf, which would
        // otherwise make exp(logprob - maxLogprob) = exp(NaN) and leak NaN into the response.
        // A finite maxLogprob guarantees expSum >= 1 (the max term is exp(0)), so no div-by-zero.
        if (!complete || !std::isfinite(maxLogprob)) {
            // Defensive shape (kept for consumers of the pre-strict contract):
            if (question.type == QuestionType::SCORE) {
                jsonResponse.String("answer");
                jsonResponse.Null();
                jsonResponse.StartObject("legend");
                for (size_t i = 0; i < question.labels.size(); ++i) {
                    jsonResponse.String(question.labels[i]);
                    jsonResponse.String(question.values[i]);
                }
                jsonResponse.EndObject();
            } else {
                jsonResponse.String("answer");
                jsonResponse.Null();
            }
            jsonResponse.StartObject("probabilities");
            jsonResponse.EndObject();
            jsonResponse.String("confidence");
            jsonResponse.Null();
            jsonResponse.EndObject();
            continue;
        }

        // Softmax over the labels (numerically stabilised by subtracting the max)
        double expSum = 0.0;
        for (double logprob : labelLogprobs)
            expSum += std::exp(logprob - maxLogprob);
        size_t winnerIndex = 0;
        for (size_t i = 1; i < labelLogprobs.size(); ++i)
            if (labelLogprobs[i] > labelLogprobs[winnerIndex])
                winnerIndex = i;
        // Full-vocabulary probability of the winning label token (= vLLM probs[top] * label_mass).
        // Clamp the winner's logprob to at most zero: float32 rounding can yield a small
        // positive logprob for near-certain predictions, which would leak confidence > 1
        // (same treatment as LogprobValue in openai_json_response.hpp).
        const double confidence = std::exp(std::min(labelLogprobs[winnerIndex], 0.0));

        if (question.type == QuestionType::PREDICATE) {
            // probability = P("yes"); "yes" is label index 0
            jsonResponse.String("probability");
            jsonResponse.Double(std::exp(labelLogprobs[0] - maxLogprob) / expSum);
        } else if (question.type == QuestionType::SCORE) {
            // Expected level (0-indexed) over the label probabilities
            double expectedValue = 0.0;
            for (size_t i = 0; i < labelLogprobs.size(); ++i)
                expectedValue += static_cast<double>(i) * (std::exp(labelLogprobs[i] - maxLogprob) / expSum);
            jsonResponse.String("score");
            jsonResponse.Double(expectedValue);
            jsonResponse.StartArray("probabilities");
            for (size_t i = 0; i < question.labels.size(); ++i) {
                jsonResponse.StartObject();
                jsonResponse.Int("value", static_cast<int>(i));
                jsonResponse.String("label", question.values[i]);
                jsonResponse.String("probability");
                jsonResponse.Double(std::exp(labelLogprobs[i] - maxLogprob) / expSum);
                jsonResponse.EndObject();
            }
            jsonResponse.EndArray();
        } else {  // CHOICE: echo the winning option's original value (string or JSON bool)
            if (question.choiceIsBool[winnerIndex]) {
                jsonResponse.String("choice");
                jsonResponse.Bool(question.values[winnerIndex] == "true");
            } else {
                jsonResponse.String("choice", question.values[winnerIndex]);
            }
            jsonResponse.StartArray("probabilities");
            for (size_t i = 0; i < question.labels.size(); ++i) {
                jsonResponse.StartObject();
                if (question.choiceIsBool[i]) {
                    jsonResponse.String("value");
                    jsonResponse.Bool(question.values[i] == "true");
                } else {
                    jsonResponse.String("value", question.values[i]);
                }
                jsonResponse.String("probability");
                jsonResponse.Double(std::exp(labelLogprobs[i] - maxLogprob) / expSum);
                jsonResponse.EndObject();
            }
            jsonResponse.EndArray();
        }
        if (question.type != QuestionType::PREDICATE) {
            jsonResponse.String("confidence");
            jsonResponse.Double(confidence);
        }
        jsonResponse.EndObject();
    }
    jsonResponse.EndArray();

    // Decisions usage contract (vLLM/OpenAI): input_tokens/output_tokens naming.
    // Inlined here (not the shared UsageObject) because completions keep prompt_tokens.
    // With prefix caching the common prompt prefix is shared across the per-question reads,
    // so cached_tokens is part of the contract.
    jsonResponse.StartObject("usage");
    jsonResponse.Int("input_tokens", usage.promptTokens);
    jsonResponse.StartObject("input_tokens_details");
    jsonResponse.Uint("cached_tokens", static_cast<size_t>(usage.cachedPromptTokens));
    jsonResponse.Int("cache_write_tokens", 0);
    jsonResponse.EndObject();
    jsonResponse.Int("output_tokens", usage.completionTokens);
    jsonResponse.StartObject("output_tokens_details");
    jsonResponse.Int("reasoning_tokens", 0);
    jsonResponse.EndObject();
    jsonResponse.Int("total_tokens", usage.calculateTotalTokens());
    jsonResponse.EndObject();

    jsonResponse.EndObject();
    return jsonResponse.ToString();
}

std::string OpenAIDecisionsHandler::serializeStreamingChunk(Delta delta, ov::genai::GenerationFinishReason finishReason) {
    (void)delta;
    (void)finishReason;
    return "";
}

std::string OpenAIDecisionsHandler::serializeStreamingUsageChunk() {
    return "";
}

std::string OpenAIDecisionsHandler::serializeStreamingHandshakeChunk() {
    return "";
}

}  // namespace ovms
