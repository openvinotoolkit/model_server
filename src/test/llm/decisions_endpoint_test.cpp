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
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../../http_rest_api_handler.hpp"
#include "../../http_status_code.hpp"
#include "../../json_parser.hpp"
#include "../../llm/apis/openai_decisions.hpp"
#include "../../llm/servable.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "../platform_utils.hpp"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"

using namespace ovms;

class LLMHttpDecisionsTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointDecisions = "/v1/decisions";
    const std::string modelName = "lm_cb_regular";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/config.json").c_str(), 60);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointDecisions, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};

std::unique_ptr<std::thread> LLMHttpDecisionsTest::t;

namespace {

std::string threeQuestionsBody(const std::string& model) {
    return R"({
        "model": ")" + model + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"},
            {"type": "choice", "instructions": "Pick a color", "choices": [{"value": "red"}, {"value": "blue"}]},
            {"type": "score", "instructions": "Rate the statement", "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}]}
        ]
    })";
}

}  // namespace

// One request with all three question types: the response must be structurally
// consistent with the strict vLLM/OpenAI contract ({model, answers, usage};
// probabilities form a distribution over the labels, confidence is a raw
// probability). Numeric values are not asserted against constants — the fixture
// model output is FP32 and platform-sensitive.
TEST_F(LLMHttpDecisionsTest, decisionsHappyPath) {
    std::string requestBody = threeQuestionsBody(modelName);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;

    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("model"));
    ASSERT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    // The strict contract exposes exactly {model, answers, usage}
    ASSERT_FALSE(parsedResponse.HasMember("id"));
    ASSERT_FALSE(parsedResponse.HasMember("object"));
    ASSERT_FALSE(parsedResponse.HasMember("created"));
    ASSERT_FALSE(parsedResponse.HasMember("diagnostics"));

    ASSERT_TRUE(parsedResponse.HasMember("usage"));
    // Each question generates exactly one token, so output = number of questions.
    ASSERT_EQ(parsedResponse["usage"]["output_tokens"].GetInt(), 3);
    ASSERT_EQ(parsedResponse["usage"]["total_tokens"].GetInt(), parsedResponse["usage"]["input_tokens"].GetInt() + 3);
    ASSERT_GE(parsedResponse["usage"]["input_tokens_details"]["cached_tokens"].GetInt(), 0);
    ASSERT_EQ(parsedResponse["usage"]["input_tokens_details"]["cache_write_tokens"].GetInt(), 0);
    ASSERT_EQ(parsedResponse["usage"]["output_tokens_details"]["reasoning_tokens"].GetInt(), 0);

    ASSERT_TRUE(parsedResponse.HasMember("answers"));
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 3);

    const std::vector<const char*> expectedTypes = {"predicate", "choice", "score"};
    // Option values per question: the choice answer echoes one of them, the score
    // probabilities carry the level labels.
    const std::vector<std::vector<const char*>> expectedValues = {
        {},
        {"red", "blue"},
        {"low", "mid", "high"},
    };

    for (rapidjson::SizeType i = 0; i < answers.Size(); ++i) {
        const auto& answer = answers[i];
        ASSERT_TRUE(answer.IsObject());
        ASSERT_TRUE(answer.HasMember("type"));
        ASSERT_STREQ(answer["type"].GetString(), expectedTypes[i]);
        ASSERT_TRUE(answer.HasMember("name"));
        ASSERT_TRUE(answer["name"].IsNull());

        if (std::string(expectedTypes[i]) == "predicate") {
            // probability = P("yes"), a normalized label probability
            ASSERT_TRUE(answer.HasMember("probability"));
            ASSERT_TRUE(answer["probability"].IsNumber());
            const double probability = answer["probability"].GetDouble();
            ASSERT_GT(probability, 0.0);
            ASSERT_LE(probability, 1.0 + 1e-6);
            ASSERT_FALSE(answer.HasMember("confidence"));
            ASSERT_FALSE(answer.HasMember("probabilities"));
            continue;
        }

        ASSERT_TRUE(answer.HasMember("confidence"));
        const double confidence = answer["confidence"].GetDouble();
        ASSERT_GT(confidence, 0.0);
        ASSERT_LE(confidence, 1.0 + 1e-6);

        ASSERT_TRUE(answer.HasMember("probabilities"));
        const auto& probabilities = answer["probabilities"];
        ASSERT_TRUE(probabilities.IsArray());
        ASSERT_EQ(probabilities.Size(), expectedValues[i].size());
        double probabilitySum = 0.0;
        for (rapidjson::SizeType j = 0; j < probabilities.Size(); ++j) {
            const auto& entry = probabilities[j];
            ASSERT_TRUE(entry.IsObject());
            ASSERT_TRUE(entry.HasMember("value"));
            ASSERT_TRUE(entry.HasMember("probability"));
            ASSERT_TRUE(entry["probability"].IsNumber());
            ASSERT_GE(entry["probability"].GetDouble(), 0.0);
            probabilitySum += entry["probability"].GetDouble();
            if (std::string(expectedTypes[i]) == "choice") {
                // value echoes the original option value
                ASSERT_TRUE(entry["value"].IsString());
                bool valueKnown = false;
                for (const char* value : expectedValues[i])
                    if (std::string(entry["value"].GetString()) == value)
                        valueKnown = true;
                ASSERT_TRUE(valueKnown) << "unexpected choice value '" << entry["value"].GetString() << "'";
            } else {
                // score: value is the 0-based level index, label is the level text
                ASSERT_EQ(entry["value"].GetInt(), static_cast<int>(j));
                ASSERT_TRUE(entry["label"].IsString());
                ASSERT_STREQ(entry["label"].GetString(), expectedValues[i][j]);
            }
        }
        ASSERT_NEAR(probabilitySum, 1.0, 1e-3);

        if (std::string(expectedTypes[i]) == "score") {
            // Score is the expected level: a number in [0, numLevels).
            ASSERT_TRUE(answer.HasMember("score"));
            ASSERT_TRUE(answer["score"].IsNumber());
            const double scoreValue = answer["score"].GetDouble();
            ASSERT_GE(scoreValue, -1e-9);
            ASSERT_LE(scoreValue, static_cast<double>(expectedValues[i].size() - 1) + 1e-9);
        } else {
            // The choice answer echoes the winning option value
            ASSERT_TRUE(answer.HasMember("choice"));
            ASSERT_TRUE(answer["choice"].IsString());
            const std::string choiceValue = answer["choice"].GetString();
            bool choiceKnown = false;
            for (const char* value : expectedValues[i])
                if (choiceValue == value)
                    choiceKnown = true;
            ASSERT_TRUE(choiceKnown) << "choice '" << choiceValue << "' is not one of the option values";
        }
    }
}

// Messages-style input (chat history) is accepted and produces the same shape
TEST_F(LLMHttpDecisionsTest, decisionsMessagesInput) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": [
            {"role": "user", "content": "The sky is"}
        ],
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;

    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    ASSERT_TRUE(parsedResponse.IsObject());
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 1);
    ASSERT_STREQ(answers[0]["type"].GetString(), "predicate");
    ASSERT_TRUE(answers[0]["probability"].IsNumber());

    // Only role "user" is accepted (vLLM/OpenAI contract): a non-user message is rejected.
    std::string trailingAssistantBody = R"({
        "model": ")" + modelName + R"(",
        "input": [
            {"role": "user", "content": "The sky is"},
            {"role": "assistant", "content": "blue"}
        ],
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";
    std::string trailingAssistantResponse;
    auto trailingAssistantStatus = handler->dispatchToProcessor(endpointDecisions, trailingAssistantBody, &trailingAssistantResponse, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(trailingAssistantStatus, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << trailingAssistantStatus.string();
    ASSERT_NE(trailingAssistantStatus.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(trailingAssistantStatus.string().find("input message role must be 'user'"), std::string::npos);
}

// Greedy single-token generation must be deterministic: identical requests yield
// identical answers and confidences
TEST_F(LLMHttpDecisionsTest, decisionsDeterministic) {
    std::string requestBody = threeQuestionsBody(modelName);

    std::string firstResponse;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &firstResponse, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << firstResponse;
    rapidjson::Document first;
    first.Parse(firstResponse.c_str());
    ASSERT_FALSE(first.HasParseError());

    std::string secondResponse;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &secondResponse, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << secondResponse;
    rapidjson::Document second;
    second.Parse(secondResponse.c_str());
    ASSERT_FALSE(second.HasParseError());

    ASSERT_EQ(first["answers"].Size(), second["answers"].Size());
    for (rapidjson::SizeType i = 0; i < first["answers"].Size(); ++i) {
        // Greedy single-token reads are deterministic: the whole answer object
        // (value fields + probabilities array) must be identical.
        ASSERT_TRUE(first["answers"][i] == second["answers"][i]);
    }
}

// Duplicate option values within a single question are rejected: two options sharing
// a value would collide in the probabilities object and push the label mass above 1.
// (Multi-word option VALUES are now allowed — the single-token check is on the label
// letter, not the value.)
TEST_F(LLMHttpDecisionsTest, decisionsDuplicateChoiceValueRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "choice", "instructions": "Pick a letter", "choices": [{"value": "same"}, {"value": "same"}]}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("duplicate choice value"), std::string::npos);
}

// More than 10 score levels is rejected: levels are labeled with the digits 0..9 in the
// prompt (one token each), so at most 10 levels are supported.
TEST_F(LLMHttpDecisionsTest, decisionsTooManyScoreLevels) {
    std::string levels = "[";
    for (int i = 0; i < 11; ++i) {
        if (i > 0)
            levels += ",";
        levels += R"({"label": "level)" + std::to_string(i) + R"("})";
    }
    levels += "]";
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "score", "instructions": "Rate", "levels": )" + levels + R"(}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("at most 10 levels"), std::string::npos);
}

// Exactly 10 score levels (labeled with the digits 0..9) is the maximum and is accepted.
TEST_F(LLMHttpDecisionsTest, decisionsMaxScoreLevelsAccepted) {
    std::string levels = "[";
    for (int i = 0; i < 10; ++i) {
        if (i > 0)
            levels += ",";
        levels += R"({"label": "level)" + std::to_string(i) + R"("})";
    }
    levels += "]";
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "score", "instructions": "Rate", "levels": )" + levels + R"(}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answer = parsedResponse["answers"][0];
    ASSERT_TRUE(answer["score"].IsNumber());
    ASSERT_TRUE(answer["probabilities"].IsArray());
    ASSERT_EQ(answer["probabilities"].Size(), 10);
}

// A string input on a VLM servable is normalized to a chat message (not a raw prompt),
// so the VLM input processors run and inputIds are populated. Regression for the raw
// prompt path, which skipped tokenization and crashed in prepareInputs.
TEST_F(LLMHttpDecisionsTest, decisionsVlmStringInput) {
    std::string requestBody = R"({
        "model": "vlm_cb_regular",
        "input": "Describe the scene",
        "questions": [
            {"type": "predicate", "instructions": "Is there an animal?"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 1);
    ASSERT_STREQ(answers[0]["type"].GetString(), "predicate");
    ASSERT_TRUE(answers[0]["probability"].IsNumber());
    // VLM prompt tokens come from the pipeline metrics (expanded visual tokens), not from
    // the text-only inputIds placeholders.
    ASSERT_GT(parsedResponse["usage"]["input_tokens"].GetInt(), 0);
}

// Media content parts are rejected: decisions input is text-only (vLLM/OpenAI contract),
// even on a VLM servable.
TEST_F(LLMHttpDecisionsTest, decisionsImageInputRejected) {
    std::string requestBody = R"({
        "model": "vlm_cb_regular",
        "input": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is in the image?"},
                {"type": "image_url", "image_url": {"url": "base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="}}
            ]}
        ],
        "questions": [
            {"type": "predicate", "instructions": "Is the main subject an animal?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("Decisions supports text input only; images are not supported"), std::string::npos);
}

// Two predicate questions in one request are each answered from their own forward pass
// (per-question prompt), not a shared first-token distribution. The dummy model is
// deterministic, so we assert both are answered independently with valid structure and
// the correct per-question names — this exercises the N-request plumbing end to end.
TEST_F(LLMHttpDecisionsTest, decisionsOppositePredicates) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The cat sat on the mat",
        "questions": [
            {"type": "predicate", "instructions": "Is the main subject an animal?", "name": "is_animal"},
            {"type": "predicate", "instructions": "Is the main subject a vehicle?", "name": "is_vehicle"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 2);
    for (rapidjson::SizeType i = 0; i < answers.Size(); ++i) {
        ASSERT_TRUE(answers[i]["probability"].IsNumber());
        const double probability = answers[i]["probability"].GetDouble();
        ASSERT_GT(probability, 0.0);
        ASSERT_LE(probability, 1.0 + 1e-6);
        ASSERT_STREQ(answers[i]["name"].GetString(), i == 0 ? "is_animal" : "is_vehicle");
    }
    // output tokens = number of questions (each generates exactly one token)
    ASSERT_EQ(parsedResponse["usage"]["output_tokens"].GetInt(), 2);
}

// More than 64 questions is rejected
TEST_F(LLMHttpDecisionsTest, decisionsTooManyQuestions) {
    std::string questions = "[";
    for (int i = 0; i < 65; ++i) {
        if (i > 0)
            questions += ",";
        questions += R"({"type": "predicate", "instructions": "q)" + std::to_string(i) + R"("})";
    }
    questions += "]";
    std::string requestBody = R"({"model": ")" + modelName + R"(", "input": "The sky is", "questions": )" + questions + "}";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("at most 64 questions"), std::string::npos);
}

// More than 26 choices is rejected
TEST_F(LLMHttpDecisionsTest, decisionsTooManyChoices) {
    std::string choices = "[";
    for (int i = 0; i < 26; ++i) {
        if (i > 0)
            choices += ",";
        choices += R"({"value": ")" + std::string(1, 'A' + i) + R"("})";
    }
    choices += R"(,{"value": "a"})";
    choices += "]";
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "choice", "instructions": "Pick a letter", "choices": )" + choices + R"(}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("at most 26 choices"), std::string::npos);
}

// Streaming is not supported: stream is not part of the decisions request contract
TEST_F(LLMHttpDecisionsTest, decisionsStreamRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "stream": true,
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("parameter 'stream' is not supported"), std::string::npos);
}

// Sampling parameters are forced by the endpoint and cannot be overridden
TEST_F(LLMHttpDecisionsTest, decisionsSamplingParamRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "temperature": 0.5,
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("is not supported for /v1/decisions"), std::string::npos);

    // Any field outside the allowlist is rejected (vLLM contract), not just the known sampling knobs.
    std::string logitBiasBody = R"({
        "model": ")" + modelName + R"(",
        "logit_bias": {},
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";
    std::string logitBiasResponse;
    auto logitBiasStatus = handler->dispatchToProcessor(endpointDecisions, logitBiasBody, &logitBiasResponse, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(logitBiasStatus, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << logitBiasStatus.string();
    ASSERT_NE(logitBiasStatus.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(logitBiasStatus.string().find("is not supported for /v1/decisions"), std::string::npos);
}

// Missing instructions is rejected
TEST_F(LLMHttpDecisionsTest, decisionsMissingInstructions) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "predicate"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("question.instructions is missing"), std::string::npos);
}

// Choice values may be JSON booleans: the winning value is echoed back as a JSON bool
TEST_F(LLMHttpDecisionsTest, decisionsBoolChoiceValue) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "choice", "instructions": "Pick one", "choices": [{"value": true}, {"value": false}]}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answer = parsedResponse["answers"][0];
    ASSERT_TRUE(answer["choice"].IsBool());
    ASSERT_TRUE(answer["probabilities"].IsArray());
    ASSERT_EQ(answer["probabilities"].Size(), 2);
    ASSERT_TRUE(answer["probabilities"][0]["value"].IsBool());
    ASSERT_TRUE(answer["probabilities"][1]["value"].IsBool());
}

// Input messages accept the OpenAI "input_text" part type and an explicit "message" type tag
TEST_F(LLMHttpDecisionsTest, decisionsInputTextParts) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": [
            {"role": "user", "type": "message", "content": [
                {"type": "input_text", "text": "The sky is"}
            ]}
        ],
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answer = parsedResponse["answers"][0];
    ASSERT_STREQ(answer["type"].GetString(), "predicate");
    ASSERT_TRUE(answer["probability"].IsNumber());
}

// An explicit null name is rejected (vLLM: name must be a string when provided)
TEST_F(LLMHttpDecisionsTest, decisionsNameNullRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?", "name": null}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("question.name must be a string when provided"), std::string::npos);
}

// joint: true is diffusion-only in vLLM and rejected here
TEST_F(LLMHttpDecisionsTest, decisionsJointTrueRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "joint": true,
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("joint reads are not supported"), std::string::npos);
}

// cache_salt and joint: false are accepted (for API compatibility, not used)
TEST_F(LLMHttpDecisionsTest, decisionsCacheSaltAccepted) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "cache_salt": "salt-1",
        "joint": false,
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    ASSERT_TRUE(parsedResponse["answers"][0]["probability"].IsNumber());
}

// Fewer than 2 choices is rejected (a single-option question has no decision to make)
TEST_F(LLMHttpDecisionsTest, decisionsSingleChoiceRejected) {
    std::string requestBody = R"({
        "model": ")" + modelName + R"(",
        "input": "The sky is",
        "questions": [
            {"type": "choice", "instructions": "Pick a letter", "choices": [{"value": "only"}]}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("INVALID_ARGUMENT"), std::string::npos);
    ASSERT_NE(status.string().find("at least 2 choices"), std::string::npos);
}

// The legacy one-shot servable serves /v1/decisions: one greedy forward pass per
// question, serialized from the requested-token logprobs.
TEST_F(LLMHttpDecisionsTest, decisionsLegacyServableHappyPath) {
    const std::string legacyModel = "lm_legacy_regular";
    std::string requestBody = threeQuestionsBody(legacyModel);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;

    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_STREQ(parsedResponse["model"].GetString(), legacyModel.c_str());
    // Each question generates exactly one token, so output = number of questions.
    ASSERT_EQ(parsedResponse["usage"]["output_tokens"].GetInt(), 3);
    ASSERT_EQ(parsedResponse["usage"]["total_tokens"].GetInt(), parsedResponse["usage"]["input_tokens"].GetInt() + 3);

    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 3);

    // The strict contract shape per question type
    ASSERT_TRUE(answers[0]["probability"].IsNumber());  // predicate
    ASSERT_TRUE(answers[1]["choice"].IsString());       // choice
    ASSERT_TRUE(answers[1]["probabilities"].IsArray());
    ASSERT_EQ(answers[1]["probabilities"].Size(), 2);
    ASSERT_TRUE(answers[1]["confidence"].IsNumber());
    ASSERT_TRUE(answers[2]["score"].IsNumber());        // score
    ASSERT_TRUE(answers[2]["probabilities"].IsArray());
    ASSERT_EQ(answers[2]["probabilities"].Size(), 3);
    ASSERT_TRUE(answers[2]["confidence"].IsNumber());
}

// Two named predicates on the legacy servable: per-question answers and usage.
TEST_F(LLMHttpDecisionsTest, decisionsLegacyServableTwoQuestions) {
    std::string requestBody = R"({
        "model": "lm_legacy_regular",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the main subject an animal?", "name": "is_animal"},
            {"type": "predicate", "instructions": "Is the main subject a vehicle?", "name": "is_vehicle"}
        ]
    })";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;
    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 2);
    for (rapidjson::SizeType i = 0; i < answers.Size(); ++i) {
        ASSERT_TRUE(answers[i]["probability"].IsNumber());
        const double probability = answers[i]["probability"].GetDouble();
        ASSERT_GT(probability, 0.0);
        ASSERT_LE(probability, 1.0 + 1e-6);
        ASSERT_STREQ(answers[i]["name"].GetString(), i == 0 ? "is_animal" : "is_vehicle");
    }
    ASSERT_EQ(parsedResponse["usage"]["output_tokens"].GetInt(), 2);
}

// Assisted-decoding servables (speculative decoding / prompt lookup) reject /v1/decisions:
// their pipelines never fill requested-token logprobs.
class LLMHttpDecisionsAssistedTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointDecisions = "/v1/decisions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9174";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/assisted_decoding_config.json").c_str(), 60);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointDecisions, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};

std::unique_ptr<std::thread> LLMHttpDecisionsAssistedTest::t;

// A fast-draft (speculative decoding) servable rejects /v1/decisions
TEST_F(LLMHttpDecisionsAssistedTest, decisionsSpeculativeServableRejected) {
    std::string requestBody = R"({
        "model": "lm_cb_speculative",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("UNIMPLEMENTED"), std::string::npos);
    ASSERT_NE(status.string().find("speculative decoding or prompt lookup decoding"), std::string::npos);
}

// A prompt lookup decoding servable rejects /v1/decisions
TEST_F(LLMHttpDecisionsAssistedTest, decisionsPromptLookupServableRejected) {
    std::string requestBody = R"({
        "model": "lm_cb_prompt_lookup",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("UNIMPLEMENTED"), std::string::npos);
    ASSERT_NE(status.string().find("speculative decoding or prompt lookup decoding"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Deterministic serializer tests: no HTTP, no model. The handler is constructed
// directly and fed hand-built requested-token logprobs, so the exact reduction
// math (label-normalized probabilities, full-vocabulary confidence, expected
// score) is checked against precomputed constants. The HTTP e2e tests above only
// verify the contract shape, so a uniform or label-normalized-confidence
// implementation would have passed them.
// ---------------------------------------------------------------------------
class DecisionsSerializerTest : public ::testing::Test {
protected:
    rapidjson::Document doc;
    // GPT-2 BPE: every decisions label (yes/no, A/B/C, 0/1/2) is a single token.
    std::shared_ptr<ov::genai::Tokenizer> tokenizer =
        std::make_shared<ov::genai::Tokenizer>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-gpt2-ov"));

    // Encodes a label the same way OpenAIDecisionsHandler::encodeLabels does.
    int64_t encodeLabel(const std::string& label) {
        auto tokens = tokenizer->encode(label, {ov::genai::add_special_tokens(false)});
        EXPECT_EQ(tokens.input_ids.get_size(), 1) << "label '" << label << "' is not a single token";
        if (tokens.input_ids.get_size() != 1)
            return -1;
        return tokens.input_ids.data<int64_t>()[0];
    }

    // Three questions covering all types: predicate, choice (3 options), score (3 levels).
    std::unique_ptr<ovms::OpenAIDecisionsHandler> parseHandler() {
        doc.Parse(R"({
            "model": "dummy",
            "input": "The sky is",
            "questions": [
                {"type": "predicate", "instructions": "Is the statement true?"},
                {"type": "choice", "instructions": "Pick a color",
                 "choices": [{"value": "red"}, {"value": "blue"}, {"value": "green"}]},
                {"type": "score", "instructions": "Rate the statement",
                 "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}]}
            ]
        })");
        if (doc.HasParseError()) {
            ADD_FAILURE() << "test request doc failed to parse";
            return nullptr;
        }
        auto handler = std::make_unique<ovms::OpenAIDecisionsHandler>(
            doc, ovms::Endpoint::DECISIONS, std::chrono::system_clock::now(), *tokenizer);
        auto status = handler->parseRequest(std::nullopt, 0, std::nullopt);
        if (!status.ok()) {
            ADD_FAILURE() << "parseRequest failed: " << status.message();
            return nullptr;
        }
        return handler;
    }

    // One UnaryChoiceLogprobs entry: a single generated token + one position of
    // requested-token logprobs (the label ids, plus a dominant non-label token).
    ovms::UnaryChoiceLogprobs makeEntry(const std::vector<std::pair<int64_t, float>>& positionLogprobs) {
        ovms::UnaryChoiceLogprobs entry;
        entry.generatedIds = {positionLogprobs.front().first};
        entry.requestedTokenLogprobs = {positionLogprobs};
        return entry;
    }
};

TEST_F(DecisionsSerializerTest, serializeUnaryResponseMath) {
    auto handler = parseHandler();
    ASSERT_NE(handler, nullptr);
    const int64_t yesId = encodeLabel("yes"), noId = encodeLabel("no");
    const int64_t aId = encodeLabel("A"), bId = encodeLabel("B"), cId = encodeLabel("C");
    const int64_t zeroId = encodeLabel("0"), oneId = encodeLabel("1"), twoId = encodeLabel("2");
    const int64_t nonLabelId = encodeLabel("the");
    for (int64_t id : {yesId, noId, aId, bId, cId, zeroId, oneId, twoId})
        EXPECT_NE(nonLabelId, id) << "the filler token must not collide with a label id";

    // Non-uniform per-question label logprobs; the dominant non-label token makes
    // the full-vocabulary confidence distinct from the label-normalized winner.
    std::vector<ovms::UnaryChoiceLogprobs> logprobData = {
        makeEntry({{yesId, -0.5f}, {noId, -2.0f}, {nonLabelId, -0.1f}}),
        makeEntry({{aId, -1.0f}, {bId, -2.0f}, {cId, -3.0f}, {nonLabelId, -0.1f}}),
        makeEntry({{zeroId, -3.0f}, {oneId, -1.0f}, {twoId, -2.0f}, {nonLabelId, -0.1f}})};

    rapidjson::Document response;
    response.Parse(handler->serializeUnaryResponse({}, {}, logprobData).c_str());
    ASSERT_FALSE(response.HasParseError());
    const auto& answers = response["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 3);

    // predicate: probability = P(yes) = exp(-0.5) / (exp(-0.5) + exp(-2.0))
    ASSERT_STREQ(answers[0]["type"].GetString(), "predicate");
    EXPECT_NEAR(answers[0]["probability"].GetDouble(), 0.8175744761936437, 1e-9);
    ASSERT_FALSE(answers[0].HasMember("confidence"));

    // choice: winner A ("red"); label-normalized probabilities, while confidence is
    // the full-vocabulary probability of the winning label token, exp(-1.0).
    ASSERT_STREQ(answers[1]["type"].GetString(), "choice");
    ASSERT_STREQ(answers[1]["choice"].GetString(), "red");
    const auto& choiceProbs = answers[1]["probabilities"];
    ASSERT_TRUE(choiceProbs.IsArray());
    ASSERT_EQ(choiceProbs.Size(), 3);
    ASSERT_STREQ(choiceProbs[0]["value"].GetString(), "red");
    EXPECT_NEAR(choiceProbs[0]["probability"].GetDouble(), 0.6652409557748218, 1e-9);
    ASSERT_STREQ(choiceProbs[1]["value"].GetString(), "blue");
    EXPECT_NEAR(choiceProbs[1]["probability"].GetDouble(), 0.24472847105479764, 1e-9);
    ASSERT_STREQ(choiceProbs[2]["value"].GetString(), "green");
    EXPECT_NEAR(choiceProbs[2]["probability"].GetDouble(), 0.09003057317038046, 1e-9);
    EXPECT_NEAR(answers[1]["confidence"].GetDouble(), 0.36787944117144233, 1e-9);

    // score: expected 0-indexed level = 1*p1 + 2*p2
    ASSERT_STREQ(answers[2]["type"].GetString(), "score");
    EXPECT_NEAR(answers[2]["score"].GetDouble(), 1.1546978978844171, 1e-9);
    const auto& scoreProbs = answers[2]["probabilities"];
    ASSERT_TRUE(scoreProbs.IsArray());
    ASSERT_EQ(scoreProbs.Size(), 3);
    ASSERT_EQ(scoreProbs[0]["value"].GetInt(), 0);
    ASSERT_STREQ(scoreProbs[0]["label"].GetString(), "low");
    EXPECT_NEAR(scoreProbs[0]["probability"].GetDouble(), 0.09003057317038046, 1e-9);
    ASSERT_EQ(scoreProbs[1]["value"].GetInt(), 1);
    ASSERT_STREQ(scoreProbs[1]["label"].GetString(), "mid");
    EXPECT_NEAR(scoreProbs[1]["probability"].GetDouble(), 0.6652409557748218, 1e-9);
    ASSERT_EQ(scoreProbs[2]["value"].GetInt(), 2);
    ASSERT_STREQ(scoreProbs[2]["label"].GetString(), "high");
    EXPECT_NEAR(scoreProbs[2]["probability"].GetDouble(), 0.24472847105479764, 1e-9);
    EXPECT_NEAR(answers[2]["confidence"].GetDouble(), 0.36787944117144233, 1e-9);
}

TEST_F(DecisionsSerializerTest, serializeUnaryResponseClampsPositiveLogprob) {
    auto handler = parseHandler();
    ASSERT_NE(handler, nullptr);
    const int64_t aId = encodeLabel("A"), bId = encodeLabel("B"), cId = encodeLabel("C");
    const int64_t zeroId = encodeLabel("0"), oneId = encodeLabel("1"), twoId = encodeLabel("2");
    // float32 rounding can yield a small positive logprob for a near-certain winner:
    // exp(0.0001f) is 1.0001000049976403, so the confidence must be clamped to 1.0.
    std::vector<ovms::UnaryChoiceLogprobs> logprobData = {
        makeEntry({{encodeLabel("yes"), -1.0f}, {encodeLabel("no"), -2.0f}}),
        makeEntry({{aId, 0.0001f}, {bId, -1.0f}, {cId, -2.0f}}),
        makeEntry({{zeroId, -1.0f}, {oneId, -2.0f}, {twoId, -3.0f}})};

    rapidjson::Document response;
    response.Parse(handler->serializeUnaryResponse({}, {}, logprobData).c_str());
    ASSERT_FALSE(response.HasParseError());
    const auto& answers = response["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 3);
    EXPECT_DOUBLE_EQ(answers[1]["confidence"].GetDouble(), 1.0);
    EXPECT_NEAR(answers[2]["confidence"].GetDouble(), 0.36787944117144233, 1e-9);
}

TEST_F(DecisionsSerializerTest, validateLogprobData) {
    auto handler = parseHandler();
    ASSERT_NE(handler, nullptr);
    const int64_t yesId = encodeLabel("yes"), noId = encodeLabel("no");
    const int64_t aId = encodeLabel("A"), bId = encodeLabel("B"), cId = encodeLabel("C");
    const int64_t zeroId = encodeLabel("0"), oneId = encodeLabel("1"), twoId = encodeLabel("2");

    auto completeData = [&]() {
        return std::vector<ovms::UnaryChoiceLogprobs>{
            makeEntry({{yesId, -0.5f}, {noId, -2.0f}}),
            makeEntry({{aId, -1.0f}, {bId, -2.0f}, {cId, -3.0f}}),
            makeEntry({{zeroId, -3.0f}, {oneId, -1.0f}, {twoId, -2.0f}})};
    };

    // Complete data passes.
    EXPECT_TRUE(handler->validateLogprobData(completeData()).ok());

    // Result count mismatch.
    auto mismatched = completeData();
    mismatched.pop_back();
    EXPECT_FALSE(handler->validateLogprobData(mismatched).ok());

    // A label token missing from the position logprobs.
    auto missingLabel = completeData();
    missingLabel[1].requestedTokenLogprobs.front() = {{aId, -1.0f}, {cId, -3.0f}};
    EXPECT_FALSE(handler->validateLogprobData(missingLabel).ok());

    // All labels masked with -inf.
    auto allMasked = completeData();
    const float negInf = -std::numeric_limits<float>::infinity();
    allMasked[2].requestedTokenLogprobs.front() = {{zeroId, negInf}, {oneId, negInf}, {twoId, negInf}};
    EXPECT_FALSE(handler->validateLogprobData(allMasked).ok());

    // No token generated for the question.
    auto emptyIds = completeData();
    emptyIds[0].generatedIds.clear();
    EXPECT_FALSE(handler->validateLogprobData(emptyIds).ok());
}
