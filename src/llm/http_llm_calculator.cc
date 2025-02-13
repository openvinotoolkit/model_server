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
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../profiler.hpp"
#include "llmnoderesources.hpp"
#include "apis/openai_completions.hpp"

using namespace ovms;

namespace mediapipe {

#define IGNORE_EOS_MAX_TOKENS_LIMIT 4000

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

class HttpLLMCalculator : public CalculatorBase {
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    ov::AnyMap executionContext;
    mediapipe::Timestamp timestamp{0};

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
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
                auto status = nodeResources->loadRequest(executionContext, cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>());
                if (status != absl::OkStatus())
                return status;

                status = nodeResources->createApiHandler(executionContext);
                if (status != absl::OkStatus())
                    return status;
                
                // Fills api_handler field of executionContext with data from the payload , this might be common for all pipelines
                status = nodeResources->parseRequest(executionContext);
                if (status != absl::OkStatus())
                    return status;
                
                // Runs necessary preprocessing on the input data like chat template application, tokenization or operations on visual data etc.
                // Depending on the pipeline type, after calling this method executionContext should contain all necessary data to start the generation.
                status = nodeResources->preparePipelineInput(executionContext);
                if (status != absl::OkStatus())
                    return status;

                status = nodeResources->schedulePipelineExecution(executionContext);
                if (status != absl::OkStatus())
                    return status;
            }
            auto& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();
            if (!apiHandler->isStream()) { // Unary scenario
                OVMS_PROFILE_SCOPE("Unary generation cycle");
                auto status = nodeResources->readCompleteExecutionResults(executionContext);
                if (status != absl::OkStatus())
                    return status;
                //status = nodeResources->processCompleteExecutionResults(executionContext);
                //if (status != absl::OkStatus())
                //    return status;
                std::string& response = executionContext.at("response").as<std::string>();
                cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{std::move(response)}, timestamp);
            } else { // Streaming scenario
                OVMS_PROFILE_SCOPE("Stream generation cycle");
                auto status = nodeResources->readPartialExecutionResults(executionContext);
                if (status != absl::OkStatus())
                    return status;
                //status = nodeResources->processPartialExecutionResults(executionContext);
                //if (status != absl::OkStatus())
                //    return status;
                std::string& response = executionContext.at("response").as<std::string>();
                cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{std::move(response)}, timestamp);
                if (executionContext.at("send_loopback_signal").as<bool>())
                    cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);
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

const std::string HttpLLMCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string HttpLLMCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string HttpLLMCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(HttpLLMCalculator);

}  // namespace mediapipe
