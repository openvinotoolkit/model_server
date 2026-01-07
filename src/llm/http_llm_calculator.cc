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
#include "../logging.hpp"
#include "../profiler.hpp"
#include "apis/openai_completions.hpp"
#include "builtin_tool_executor.hpp"
#include "io_processing/base_output_parser.hpp"
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
        this->servable = it->second;
        this->executionContext = servable->createExecutionContext();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator [Node: {}] Open end", cc->NodeName());
        return absl::OkStatus();
    }
    absl::Status Process(CalculatorContext* cc) final {
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

                // Tokenize endpoint doesn't require full servable path and it ends workflow after tokenization, it does not need additional processing
                if (executionContext->endpoint == Endpoint::TOKENIZE) {
                    OVMS_PROFILE_SCOPE("Tokenize generation cycle");
                    status = servable->processTokenizeRequest(executionContext);
                    if (status != absl::OkStatus())
                        return status;
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Tokenization response prepared, sending it down the graph", cc->NodeName());

                    std::string& response = executionContext->response;
                    cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new std::string{std::move(response)}, iterationBeginTimestamp);
                    return absl::OkStatus();
                }

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

                // Built-in tool execution loop
                // This loop continues inference when built-in tools are detected, executes them,
                // appends results to chat history, and re-runs inference
                while (true) {
                    auto status = servable->readCompleteExecutionResults(executionContext);
                    if (status != absl::OkStatus())
                        return status;
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Received complete execution results", cc->NodeName());

                    // Reset parsed output state before preparing response
                    executionContext->hasLastParsedOutput = false;

                    // Prepare response - this parses the output and stores it in executionContext->lastParsedOutput
                    status = servable->prepareCompleteResponse(executionContext);
                    if (status != absl::OkStatus())
                        return status;
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Response prepared, checking for built-in tools", cc->NodeName());

                    // Check if there are built-in tool calls to execute (uses executionContext->lastParsedOutput)
                    if (ovms::GenAiServable::hasBuiltInToolCalls(executionContext)) {
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Built-in tool calls detected: {}", 
                            cc->NodeName(), executionContext->lastParsedOutput.builtInToolCalls.size());
                        
                        // Check iteration limit to prevent infinite loops
                        if (executionContext->builtInToolExecutionIteration >= ovms::GenAiServableExecutionContext::MAX_BUILTIN_TOOL_ITERATIONS) {
                            SPDLOG_LOGGER_WARN(llm_calculator_logger, "LLMCalculator  [Node: {}] Max built-in tool execution iterations ({}) reached, stopping", 
                                cc->NodeName(), ovms::GenAiServableExecutionContext::MAX_BUILTIN_TOOL_ITERATIONS);
                            break;
                        }

                        executionContext->builtInToolExecutionIteration++;
                        SPDLOG_LOGGER_INFO(llm_calculator_logger, "LLMCalculator  [Node: {}] Executing built-in tools (iteration {})", 
                            cc->NodeName(), executionContext->builtInToolExecutionIteration);

                        // Execute built-in tools using the parsed output from executionContext
                        const auto& parsedOutput = executionContext->lastParsedOutput;
                        ovms::BuiltInToolResults_t toolResults = servable->executeBuiltInTools(parsedOutput.builtInToolCalls);
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Built-in tools executed, got {} results", 
                            cc->NodeName(), toolResults.size());

                        // Append assistant message and tool results to chat history
                        servable->appendToolResultsToChatHistory(executionContext, parsedOutput.content, parsedOutput.builtInToolCalls, toolResults);
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Tool results appended to chat history", cc->NodeName());

                        // Re-prepare inputs with updated chat history
                        status = servable->prepareInputs(executionContext);
                        if (status != absl::OkStatus())
                            return status;
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Re-prepared inputs for continued inference", cc->NodeName());

                        // Schedule new execution
                        status = servable->scheduleExecution(executionContext);
                        if (status != absl::OkStatus())
                            return status;
                        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Re-scheduled execution for built-in tool continuation", cc->NodeName());

                        // Continue the loop to read the next generation result
                        continue;
                    }

                    // No built-in tools to execute, break out of the loop
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] No built-in tools detected, proceeding with response", cc->NodeName());
                    break;
                }

                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLMCalculator  [Node: {}] Sending final response down the graph", cc->NodeName());

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
