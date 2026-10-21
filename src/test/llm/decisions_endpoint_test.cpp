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
#include <cmath>
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
// consistent (probabilities form a distribution over the labels, confidence is a
// raw probability bounded by the label mass). Numeric values are not asserted
// against constants — the fixture model output is FP32 and platform-sensitive.
TEST_F(LLMHttpDecisionsTest, decisionsHappyPath) {
    std::string requestBody = threeQuestionsBody(modelName);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << response;

    parsedResponse.Parse(response.c_str());
    ASSERT_FALSE(parsedResponse.HasParseError());
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("object"));
    ASSERT_STREQ(parsedResponse["object"].GetString(), "decision");
    ASSERT_TRUE(parsedResponse.HasMember("model"));
    ASSERT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    ASSERT_TRUE(parsedResponse.HasMember("created"));
    ASSERT_TRUE(parsedResponse["created"].IsInt());
    ASSERT_TRUE(parsedResponse.HasMember("usage"));
    // Each question generates exactly one token, so completion = number of questions.
    ASSERT_EQ(parsedResponse["usage"]["completion_tokens"].GetInt(), 3);
    ASSERT_EQ(parsedResponse["usage"]["total_tokens"].GetInt(), parsedResponse["usage"]["prompt_tokens"].GetInt() + 3);

    ASSERT_TRUE(parsedResponse.HasMember("answers"));
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 3);

    ASSERT_TRUE(parsedResponse.HasMember("diagnostics"));
    const auto& diagnostics = parsedResponse["diagnostics"];
    ASSERT_TRUE(diagnostics.IsArray());
    ASSERT_EQ(diagnostics.Size(), 3);

    const std::vector<const char*> expectedTypes = {"predicate", "choice", "score"};
    // Probability keys: choice is keyed by the option value, score/predicate by the
    // label token read from the model.
    const std::vector<std::vector<const char*>> expectedProbKeys = {
        {"yes", "no"},
        {"red", "blue"},
        {"0", "1", "2"},
    };

    for (rapidjson::SizeType i = 0; i < answers.Size(); ++i) {
        const auto& answer = answers[i];
        ASSERT_TRUE(answer.IsObject());
        ASSERT_TRUE(answer.HasMember("type"));
        ASSERT_STREQ(answer["type"].GetString(), expectedTypes[i]);
        ASSERT_TRUE(answer.HasMember("name"));
        ASSERT_TRUE(answer["name"].IsNull());

        ASSERT_TRUE(answer.HasMember("answer"));
        if (std::string(expectedTypes[i]) == "score") {
            // Score answer is the expected level: a number in [0, numLevels).
            ASSERT_TRUE(answer["answer"].IsNumber());
            const double scoreValue = answer["answer"].GetDouble();
            ASSERT_GE(scoreValue, -1e-9);
            ASSERT_LE(scoreValue, static_cast<double>(expectedProbKeys[i].size() - 1) + 1e-9);
            ASSERT_TRUE(answer.HasMember("legend"));
            ASSERT_TRUE(answer["legend"].IsObject());
            ASSERT_EQ(answer["legend"].MemberCount(), expectedProbKeys[i].size());
        } else {
            // choice/predicate answer is the winning option value / "yes" / "no".
            ASSERT_TRUE(answer["answer"].IsString());
            const std::string answerText = answer["answer"].GetString();
            bool answerIsKey = false;
            for (const char* key : expectedProbKeys[i])
                if (answerText == key)
                    answerIsKey = true;
            ASSERT_TRUE(answerIsKey) << "answer '" << answerText << "' is not one of the option values of question " << i;
        }

        ASSERT_TRUE(answer.HasMember("probabilities"));
        const auto& probabilities = answer["probabilities"];
        ASSERT_TRUE(probabilities.IsObject());
        ASSERT_EQ(probabilities.MemberCount(), expectedProbKeys[i].size());
        double probabilitySum = 0.0;
        for (auto member = probabilities.MemberBegin(); member != probabilities.MemberEnd(); ++member) {
            bool keyKnown = false;
            for (const char* key : expectedProbKeys[i])
                if (std::string(member->name.GetString()) == key)
                    keyKnown = true;
            ASSERT_TRUE(keyKnown) << "unexpected probability key '" << member->name.GetString() << "'";
            ASSERT_TRUE(member->value.IsNumber());
            ASSERT_GE(member->value.GetDouble(), 0.0);
            probabilitySum += member->value.GetDouble();
        }
        ASSERT_NEAR(probabilitySum, 1.0, 1e-3);

        ASSERT_TRUE(answer.HasMember("confidence"));
        const double confidence = answer["confidence"].GetDouble();
        ASSERT_GT(confidence, 0.0);
        ASSERT_LE(confidence, 1.0 + 1e-6);

        const auto& diagnostic = diagnostics[i];
        ASSERT_TRUE(diagnostic.IsObject());
        const double labelMass = diagnostic["label_mass"].GetDouble();
        ASSERT_GT(labelMass, 0.0);
        ASSERT_LE(labelMass, 1.0 + 1e-6);
        // The winner's raw probability is one of the terms of the label mass sum
        ASSERT_LE(confidence, labelMass + 1e-6);
        ASSERT_TRUE(diagnostic["argmax_is_label"].IsBool());
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
    ASSERT_STREQ(parsedResponse["object"].GetString(), "decision");
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 1);
    ASSERT_STREQ(answers[0]["type"].GetString(), "predicate");
    ASSERT_TRUE(answers[0]["probabilities"].IsObject());
    ASSERT_EQ(answers[0]["probabilities"].MemberCount(), 2);

    // A history that ends on an assistant turn: the question block cannot be folded into the
    // assistant message, so a fresh user message is started. The request must still be answered.
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
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointDecisions, trailingAssistantBody, &trailingAssistantResponse, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK)
        << trailingAssistantResponse;
    rapidjson::Document trailingAssistant;
    trailingAssistant.Parse(trailingAssistantResponse.c_str());
    ASSERT_FALSE(trailingAssistant.HasParseError());
    const auto& trailingAssistantAnswers = trailingAssistant["answers"];
    ASSERT_TRUE(trailingAssistantAnswers.IsArray());
    ASSERT_EQ(trailingAssistantAnswers.Size(), 1);
    ASSERT_STREQ(trailingAssistantAnswers[0]["type"].GetString(), "predicate");
    ASSERT_TRUE(trailingAssistantAnswers[0]["probabilities"].IsObject());
    ASSERT_EQ(trailingAssistantAnswers[0]["probabilities"].MemberCount(), 2);
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
        // The score answer is a number, not a string, so compare values (operator== handles
        // both); confidence stays a strict double comparison.
        ASSERT_TRUE(first["answers"][i]["answer"] == second["answers"][i]["answer"]);
        ASSERT_DOUBLE_EQ(first["answers"][i]["confidence"].GetDouble(), second["answers"][i]["confidence"].GetDouble());
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
    ASSERT_TRUE(answer["answer"].IsNumber());
    ASSERT_TRUE(answer["probabilities"].IsObject());
    ASSERT_EQ(answer["probabilities"].MemberCount(), 10);
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
    ASSERT_STREQ(parsedResponse["object"].GetString(), "decision");
    const auto& answers = parsedResponse["answers"];
    ASSERT_TRUE(answers.IsArray());
    ASSERT_EQ(answers.Size(), 1);
    ASSERT_STREQ(answers[0]["type"].GetString(), "predicate");
    ASSERT_TRUE(answers[0]["probabilities"].IsObject());
    ASSERT_EQ(answers[0]["probabilities"].MemberCount(), 2);
    // VLM prompt tokens come from the pipeline metrics (expanded visual tokens), not from
    // the text-only inputIds placeholders.
    ASSERT_GT(parsedResponse["usage"]["prompt_tokens"].GetInt(), 0);
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
        ASSERT_TRUE(answers[i]["answer"].IsString());
        ASSERT_TRUE(answers[i]["confidence"].IsNumber());
        ASSERT_TRUE(answers[i]["probabilities"].IsObject());
        ASSERT_EQ(answers[i]["probabilities"].MemberCount(), 2);
        ASSERT_STREQ(answers[i]["name"].GetString(), i == 0 ? "is_animal" : "is_vehicle");
    }
    // completion tokens = number of questions (each generates exactly one token)
    ASSERT_EQ(parsedResponse["usage"]["completion_tokens"].GetInt(), 2);
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

// Streaming is not supported
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
    ASSERT_NE(status.string().find("streaming is not supported"), std::string::npos);
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

// /v1/decisions requires continuous batching; legacy servables reject it
TEST_F(LLMHttpDecisionsTest, decisionsLegacyServableRejected) {
    std::string requestBody = R"({
        "model": "lm_legacy_regular",
        "input": "The sky is",
        "questions": [
            {"type": "predicate", "instructions": "Is the statement true?"}
        ]
    })";

    auto status = handler->dispatchToProcessor(endpointDecisions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    ASSERT_NE(status.string().find("UNIMPLEMENTED"), std::string::npos);
    ASSERT_NE(status.string().find("requires a continuous batching servable"), std::string::npos);
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
