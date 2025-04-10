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
#include <atomic>
#include <string>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../profiler.hpp"
#include "apis/openai_completions.hpp"
#include "servable.hpp"

using namespace ovms;

namespace mediapipe {

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

class HttpLLMCalculator : public CalculatorBase {
    std::shared_ptr<GenAiServable> servable;
    std::shared_ptr<GenAiServableExecutionContext> executionContext;

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp iterationBeginTimestamp{0};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::GenAiServableMap>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
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
        ovms::GenAiServableMap servableMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::GenAiServableMap>();
        auto it = servableMap.find(cc->NodeName());
        RET_CHECK(it != servableMap.end()) << "Could not find initialized LLM node named: " << cc->NodeName();
        servable = it->second;
        executionContext = servable->createExecutionContext();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        //SPDLOG_INFO("Fail in Calculator::Process()");
        //return absl::InvalidArgumentError("Response generation failed");

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Process start", cc->NodeName());
        OVMS_PROFILE_FUNCTION();
        RET_CHECK(this->servable != nullptr);

        // For cases where MediaPipe decides to trigger Process() when there are no inputs
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }

        executionContext->response = "";  // always enter new process with initialized, empty response
        try {
            // First iteration of Process()
            if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
                auto status = servable->loadRequest(executionContext, cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>());
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Request loaded successfully", cc->NodeName());

                // Creates internal API handler in executionContext with data from the payload and parses the request
                status = servable->parseRequest(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Request parsed successfully", cc->NodeName());

                // Runs necessary preprocessing on the input data like chat template application, tokenization or operations on visual data etc.
                // Depending on the pipeline type, after calling this method executionContext should contain all necessary data to start the generation.
                status = servable->prepareInputs(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Input for the pipeline prepared successfully", cc->NodeName());

                status = servable->scheduleExecution(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Pipeline execution scheduled successfully", cc->NodeName());
            }

            if (!executionContext->apiHandler->isStream()) {  // Unary scenario
                OVMS_PROFILE_SCOPE("Unary generation cycle");
                auto status = servable->readCompleteExecutionResults(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Received complete execution results, preparing response", cc->NodeName());

                status = servable->prepareCompleteResponse(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Response prepared, sending it down the graph", cc->NodeName());

                std::string& response = executionContext->response;
                cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{std::move(response)}, iterationBeginTimestamp);
            } else {  // Streaming scenario
                OVMS_PROFILE_SCOPE("Stream generation cycle");
                auto status = servable->readPartialExecutionResults(executionContext);
                if (status != absl::OkStatus())
                    return status;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Received partial execution results", cc->NodeName());

                status = servable->preparePartialResponse(executionContext);
                if (status != absl::OkStatus())
                    return status;
                std::string& response = executionContext->response;
                if (!response.empty()) {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Response prepared, sending it down the graph", cc->NodeName());
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{std::move(response)}, iterationBeginTimestamp);
                }
                if (executionContext->sendLoopbackSignal)
                    cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, iterationBeginTimestamp);
            }
        } catch (ov::AssertFailure& e) {
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            return absl::InvalidArgumentError("Response generation failed");
        }
        auto now = std::chrono::system_clock::now();
        iterationBeginTimestamp = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }
};

const std::string HttpLLMCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string HttpLLMCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string HttpLLMCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(HttpLLMCalculator);

}  // namespace mediapipe
