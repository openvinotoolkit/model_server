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
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "openai_api_handler.hpp"

namespace ovms {

// Handler for the /v1/decisions endpoint (vLLM structured-decisions compatible):
// one greedy forward pass PER QUESTION, each with its own prompt embedding the
// question's instructions and options. The first-token log-probabilities of the
// question's label tokens are gathered and reduced to per-question probabilities.
//
// Label scheme (vLLM parity):
//   - choice    -> options labeled A..Z in the prompt; read the letter token;
//                  answer = the option value; multi-word values are allowed.
//   - score     -> levels labeled 0..9 in the prompt; read the digit token;
//                  answer = expected value (a number); a legend maps index->level.
//   - predicate -> labels "yes"/"no"; answer = "yes"/"no".
class OpenAIDecisionsHandler : public OpenAIApiHandler {
public:
    enum class QuestionType {
        PREDICATE,
        CHOICE,
        SCORE,
    };

    struct Question {
        QuestionType type;
        std::optional<std::string> name;
        std::string instructions;
        // Answer-facing values: choice option value / score level label / "yes","no".
        std::vector<std::string> values;
        // Prompt-facing tokens read from the model: A,B,C / 0,1,2 / yes,no.
        std::vector<std::string> labels;
        std::vector<int64_t> labelTokenIds;  // parallel to labels
    };

    using OpenAIApiHandler::OpenAIApiHandler;  // Inherit constructors

    absl::Status parseRequestImpl(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
        std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains) override;

    // Emits one InputRequest per question: the shared input (chat history) with the
    // question block appended to the last user message, plus a greedy config whose
    // logprob_token_ids are that question's label token ids.
    absl::StatusOr<std::vector<InputRequest>> extractInputRequests(GenerationConfigBuilder& configBuilder) override;

    // The decisions response is built from requested-token logprobs, which are
    // not gated on the logprobs request flags.
    bool requiresLogprobData() const override { return true; }

    std::string serializeUnaryResponse(const std::vector<Delta>& deltas, ov::genai::GenerationFinishReason finishReason) override;
    std::string serializeUnaryResponse(const std::vector<std::vector<Delta>>& allDeltas,
        const std::vector<ov::genai::GenerationFinishReason>& finishReasons,
        const std::vector<UnaryChoiceLogprobs>& logprobData) override;
    std::string serializeStreamingChunk(Delta delta, ov::genai::GenerationFinishReason finishReason) override;
    std::string serializeStreamingUsageChunk() override;
    std::string serializeStreamingHandshakeChunk() override;

private:
    absl::Status parseInput();
    absl::Status parseQuestions();
    absl::Status encodeLabels();

    // The per-question prompt block (vLLM `QuestionType::prompt`): the question
    // heading, one line per option, and the reply instruction.
    static std::string buildQuestionBlock(const Question& question);
    // Appends the question block to the last user message of the chat history
    // (or pushes a new user message when there is none).
    static void appendQuestionBlock(ov::genai::ChatHistory& history, const std::string& block);

    std::vector<Question> questions;
};

}  // namespace ovms
