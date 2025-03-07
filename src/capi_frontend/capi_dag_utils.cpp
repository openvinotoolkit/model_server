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
#include "capi_dag_utils.hpp"

#include <memory>
#include <string>
#include <utility>

#include "capi_utils.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "buffer.hpp"
#include "inferencetensor.hpp"
#include "inferenceresponse.hpp"

namespace ovms {
OVMS_ServableState convertToServableState(ovms::PipelineDefinitionStateCode code) {
    switch (code) {
    case ovms::PipelineDefinitionStateCode::BEGIN:
        return OVMS_ServableState::OVMS_STATE_BEGIN;
    case ovms::PipelineDefinitionStateCode::RELOADING:
        return OVMS_ServableState::OVMS_STATE_LOADING;
    case ovms::PipelineDefinitionStateCode::AVAILABLE:
    case ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION:
        return OVMS_ServableState::OVMS_STATE_AVAILABLE;
    case ovms::PipelineDefinitionStateCode::RETIRED:
        return OVMS_ServableState::OVMS_STATE_RETIRED;
    case ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED:
    case ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION:
        return OVMS_ServableState::OVMS_STATE_LOADING_FAILED;
    }
    throw new std::exception();
}

Status prepareConsolidatedTensorImpl(InferenceResponse* response, const std::string& name, ov::element::Type_t precision, const ov::Shape& shape, char*& bufferOut, size_t size) {
    InferenceTensor* outputTensor{nullptr};
    Status status = response->addOutput(
        name,
        getPrecisionAsOVMSDataType(ovElementTypeToOvmsPrecision(precision)),
        reinterpret_cast<const int64_t*>(shape.data()),
        shape.size());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, servable: {}; tensor with name: {}", response->getServableName(), name);
        return StatusCode::INTERNAL_ERROR;
    }
    const std::string* outputNameFromCapiTensor = nullptr;
    size_t outputId = 0;
    auto outputCount = response->getOutputCount();
    while (outputId < outputCount) {
        status = response->getOutput(outputId, &outputNameFromCapiTensor, &outputTensor);
        if (status.ok() &&
            (nullptr != outputNameFromCapiTensor) &&
            (name == *outputNameFromCapiTensor)) {
            if (precision == ov::element::string) {
                std::string msg{"String format is not supported in DAG in demultiplexing scenarios as of now"};
                SPDLOG_LOGGER_DEBUG(dag_executor_logger, msg);
                return Status(StatusCode::NOT_IMPLEMENTED, std::move(msg));
            }

            auto consolidatedBuffer = std::make_unique<Buffer>(size, OVMS_BUFFERTYPE_CPU, std::nullopt);
            // const cast is ok here since we own the buffer
            bufferOut = reinterpret_cast<char*>(const_cast<void*>(consolidatedBuffer->data()));
            outputTensor->setBuffer(std::move(consolidatedBuffer));
            return StatusCode::OK;
        }
        ++outputId;
    }
    SPDLOG_LOGGER_ERROR(dag_executor_logger, "Cannot serialize output with name:{} for servable name:{}; version:{}; error: cannot find output",
        name, response->getServableName(), response->getServableVersion());
    return StatusCode::INTERNAL_ERROR;
}
}  // namespace ovms
