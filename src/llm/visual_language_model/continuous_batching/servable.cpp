//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "servable.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../config.hpp"
#include "../../../logging.hpp"
#include "../../../tokenize/tokenize_parser.hpp"
#include "../../text_utils.hpp"
#if (PYTHON_DISABLE == 0)
#endif

namespace ovms {
namespace {

template <typename Pipeline, typename = void>
struct SupportsAudioAddRequest : std::false_type {};

template <typename Pipeline>
struct SupportsAudioAddRequest<Pipeline, std::void_t<decltype(std::declval<Pipeline&>().add_request(
                                           std::declval<uint64_t>(),
                                           std::declval<const std::string&>(),
                                           std::declval<const std::vector<ov::Tensor>&>(),
                                           std::declval<const std::vector<ov::Tensor>&>(),
                                           std::declval<const std::vector<ov::Tensor>&>(),
                                           std::declval<const ov::genai::GenerationConfig&>()))>> : std::true_type {};

template <typename Pipeline, typename GenerationHandle>
absl::Status addVlmRequest(Pipeline& pipeline,
    GenerationHandle& generationHandle,
    uint64_t requestId,
    const InputRequest& inputRequest) {
    if constexpr (SupportsAudioAddRequest<Pipeline>::value) {
        generationHandle = pipeline.add_request(
            requestId,
            inputRequest.promptText,
            inputRequest.inputImages,
            inputRequest.inputVideos,
            inputRequest.inputAudios,
            inputRequest.generationConfig);
        return absl::OkStatus();
    }

    if (!inputRequest.inputAudios.empty()) {
        return absl::UnimplementedError(
            "Audio input with VLM continuous batching requires an OpenVINO GenAI version that supports audio in add_request().");
    }

    generationHandle = pipeline.add_request(
        requestId,
        inputRequest.promptText,
        inputRequest.inputImages,
        inputRequest.inputVideos,
        inputRequest.generationConfig);
    return absl::OkStatus();
}

}  // namespace

void VisualLanguageModelServable::logPerfMetrics(ov::genai::VLMPerfMetrics& perfMetrics) {
    const size_t inputTokenCount = perfMetrics.get_num_input_tokens();
    const size_t outputTokenCount = perfMetrics.get_num_generated_tokens();
    const double prepareEmbeddingsTimeMs = perfMetrics.get_prepare_embeddings_duration().mean;
    // Continuous batching starts request timing after embeddings preparation.
    const double llmTtftMs = perfMetrics.get_ttft().mean;
    const double ttftMs = llmTtftMs + prepareEmbeddingsTimeMs;
    const double prefillSpeedTps = calculatePrefillSpeed(inputTokenCount, llmTtftMs);

    SPDLOG_LOGGER_DEBUG(
        llm_calculator_logger,
        "Request processing metrics | input_token_count: {} | output_token_count: {} | total_token_count: {} | prepare_embeddings_time_ms: {:.3f} | llm_ttft_ms: {:.3f} | ttft_ms: {:.3f} | prefill_speed_tps: {:.3f} | image_slice_count: {}",
        inputTokenCount,
        outputTokenCount,
        inputTokenCount + outputTokenCount,
        prepareEmbeddingsTimeMs,
        llmTtftMs,
        ttftMs,
        prefillSpeedTps,
        perfMetrics.get_total_image_slice_count());
}

absl::Status VisualLanguageModelServable::addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    return addVlmRequest(
        *properties->pipeline,
        vlmExecutionContext->generationHandle,
        currentRequestId++,  // to be removed from API?
        vlmExecutionContext->inputRequest);
}

absl::Status VisualLanguageModelServable::validateEndpoint(Endpoint endpoint) const {
    if (endpoint == Endpoint::COMPLETIONS) {
        return absl::InvalidArgumentError("VLM Servable does not support the /completions endpoint. Use /chat/completions or /responses.");
    }
    return absl::OkStatus();
}

std::shared_ptr<GenAiServableExecutionContext> VisualLanguageModelServable::createExecutionContext() {
    return std::make_shared<VisualLanguageModelServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> VisualLanguageModelServable::getProperties() {
    return properties;
}

absl::Status VisualLanguageModelServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto status = GenAiServable::prepareCompleteResponse(executionContext);
    if (status.ok() && llm_calculator_logger->should_log(spdlog::level::debug)) {
        auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
        auto perfMetrics = vlmExecutionContext->generationHandle->get_vlm_perf_metrics();
        logPerfMetrics(perfMetrics);
    }
    return status;
}

absl::Status VisualLanguageModelServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto status = GenAiServable::preparePartialResponse(executionContext);
    if (!status.ok()) {
        stopGenerationOnResponseError(std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext));
        return status;
    }
    if (status.ok() &&
        !executionContext->sendLoopbackSignal &&
        llm_calculator_logger->should_log(spdlog::level::debug)) {
        auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
        auto perfMetrics = vlmExecutionContext->generationHandle->get_vlm_perf_metrics();
        logPerfMetrics(perfMetrics);
    }
    return status;
}

}  // namespace ovms
