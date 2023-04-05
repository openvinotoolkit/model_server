//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "customnodesession.hpp"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>

#include "../custom_node_interface.h"  // NOLINT
#include "../logging.hpp"
#include "../profiler.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#include "custom_node_output_allocator.hpp"
#include "node.hpp"
#include "node_library.hpp"
#include "node_library_utils.hpp"
#include "nodeinputhandler.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(metadata, nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(std::move(metadata), nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::~CustomNodeSession() = default;

static std::unordered_map<std::string, shape_t> createOwnedShapesCopy(const TensorMap& tensorMap) {
    std::unordered_map<std::string, shape_t> tensorsDims;
    for (auto& [name, tensor] : tensorMap) {
        shape_t tensorDims = tensor.get_shape();
        tensorsDims.emplace(name, std::move(tensorDims));
    }
    return tensorsDims;
}

Status CustomNodeSession::execute(PipelineEventQueue& notifyEndQueue, Node& node, const NodeLibrary& library, std::unique_ptr<struct CustomNodeParam[]>& parameters, int parametersCount, void* customNodeLibraryInternalManager) {
    OVMS_PROFILE_FUNCTION();
    const auto& tensorMap = this->inputHandler->getInputs();
    auto inputTensorsCount = tensorMap.size();
    // this is a hack to overcome OV 1.0 -> 2.0 API change where we do not get reference to
    // tensor shape now but a copy. Hence we have to extend the lifetime of dims vector
    auto tensorsDims = createOwnedShapesCopy(tensorMap);
    auto inputTensors = createCustomNodeTensorArray(tensorMap, tensorsDims);
    struct CustomNodeTensor* outputTensors = nullptr;
    int outputTensorsCount = 0;
    this->timer->start(EXECUTE);
    OVMS_PROFILE_SYNC_BEGIN("Custom Node Library execute()");
    int result = library.execute(
        inputTensors.get(),
        inputTensorsCount,
        &outputTensors,
        &outputTensorsCount,
        parameters.get(),
        parametersCount,
        customNodeLibraryInternalManager);
    OVMS_PROFILE_SYNC_END("Custom Node Library execute()");
    this->timer->stop(EXECUTE);
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Custom node execution processing time for node {}; session: {} - {} ms",
        this->getName(),
        this->getSessionKey(),
        this->timer->elapsed<std::chrono::microseconds>(EXECUTE) / 1000);

    // If result is not 0, it means execution has failed.
    // In this case shared library is responsible for cleaning up resources (memory).
    if (result != 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has failed custom node execution with return code: {}", getName(), getSessionKey(), result);
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_EXECUTION_FAILED;
    }
    // In other cases we are responsible of cleaning whatever is possible.
    if (outputTensors == nullptr) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted outputs handle", getName(), getSessionKey());
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED;
    }

    if (outputTensorsCount <= 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted number of outputs", getName(), getSessionKey());
        library.release(outputTensors, customNodeLibraryInternalManager);
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT;
    }

    // At this point this is important we do not exit before finishing the loop.
    // There will be memory leak if any tensor is not converted into ov::Tensor.
    // ov::Tensor destructor is responsible for cleaning up resources.
    Status status = StatusCode::OK;
    for (int i = 0; i < outputTensorsCount; i++) {
        ov::Tensor resultTensor;
        auto result = this->createTensor(&outputTensors[i], resultTensor, library, customNodeLibraryInternalManager);
        if (outputTensors[i].name == nullptr) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed tensor conversion - missing output name", getName(), getSessionKey());
            status = StatusCode::NODE_LIBRARY_OUTPUT_MISSING_NAME;
            continue;
        }
        if (!result.ok()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed to convert {}: to tensor", getName(), getSessionKey(), outputTensors[i].name);
            if (status.ok()) {
                status = result;
            }
            continue;
        }
        this->resultTensors.emplace(std::string(outputTensors[i].name), std::move(resultTensor));
    }

    library.release(outputTensors, customNodeLibraryInternalManager);
    notifyEndQueue.push({node, getSessionKey()});
    return status;
}

Status CustomNodeSession::fetchResult(const std::string& name, ov::Tensor& resultTensor) {
    auto it = resultTensors.find(name);
    if (it == resultTensors.end()) {
        return StatusCode::NODE_LIBRARY_MISSING_OUTPUT;
    }
    resultTensor = it->second;
    return StatusCode::OK;
}

void CustomNodeSession::releaseTensorResources(const struct CustomNodeTensor* tensor, const NodeLibrary& library, void* customNodeLibraryInternalManager) {
    if (tensor->data) {
        library.release(tensor->data, customNodeLibraryInternalManager);
    }
    if (tensor->dims) {
        library.release(tensor->dims, customNodeLibraryInternalManager);
    }
}

class TensorResourcesGuard {
    const struct CustomNodeTensor* tensor;
    const NodeLibrary& library;
    bool persistData = false;
    void* customNodeLibraryInternalManager;

public:
    TensorResourcesGuard(const struct CustomNodeTensor* tensor, const NodeLibrary& library, void* customNodeLibraryInternalManager) :
        tensor(tensor),
        library(library),
        customNodeLibraryInternalManager(customNodeLibraryInternalManager) {}
    ~TensorResourcesGuard() {
        if (tensor->data && !persistData) {
            library.release(tensor->data, customNodeLibraryInternalManager);
        }
        if (tensor->dims) {
            library.release(tensor->dims, customNodeLibraryInternalManager);
        }
    }
    void setPersistData() {
        this->persistData = true;
    }
};

Status CustomNodeSession::createTensor(const struct CustomNodeTensor* tensor, ov::Tensor& resultTensor, const NodeLibrary& library, void* customNodeLibraryInternalManager) {
    TensorResourcesGuard tensorResourcesGuard(tensor, library, customNodeLibraryInternalManager);

    auto precision = ovmsPrecisionToIE2Precision(toInferenceEnginePrecision(tensor->precision));
    if (precision == ov::element::Type_t::undefined) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; Unspecified output precision:{} from custom node tensor: {}",
            this->getName(),
            this->getSessionKey(),
            precision,
            tensor->name);
        return StatusCode::NODE_LIBRARY_INVALID_PRECISION;
    }

    if (tensor->dims == nullptr || tensor->dimsCount == 0) {
        std::string error;
        if (tensor->dims == nullptr) {
            error = "shape handle is null";
        } else if (tensor->dimsCount == 0) {
            error = "shape dimensions number is equal to 0";
        }
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; error: {}",
            this->getName(),
            this->getSessionKey(),
            error);
        return StatusCode::NODE_LIBRARY_INVALID_SHAPE;
    }
    shape_t shape(tensor->dims, tensor->dims + tensor->dimsCount);

    size_t expectedElementsCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    size_t expectedDataLength = expectedElementsCount *= ov::element::Type(precision).size();
    if (tensor->data == nullptr || tensor->dataBytes != expectedDataLength) {
        std::stringstream error;
        if (tensor->data == nullptr) {
            error << "data handle is null";
        } else if (tensor->dataBytes != expectedDataLength) {
            error << "not expected data length: expected: " << expectedDataLength << " vs " << tensor->dataBytes;
        }
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; error: {}",
            this->getName(),
            this->getSessionKey(),
            error.str());
        return StatusCode::NODE_LIBRARY_INVALID_CONTENT_SIZE;
    }
    auto allocator = CustomNodeOutputAllocator(*tensor, library, customNodeLibraryInternalManager);
    try {
        switch (tensor->precision) {
        case CustomNodeTensorPrecision::FP32:
        case CustomNodeTensorPrecision::I32:
        case CustomNodeTensorPrecision::I8:
        case CustomNodeTensorPrecision::U8:
        case CustomNodeTensorPrecision::FP16:
        case CustomNodeTensorPrecision::I16:
        case CustomNodeTensorPrecision::U16:
        case CustomNodeTensorPrecision::FP64:
        case CustomNodeTensorPrecision::I64:
            resultTensor = ov::Tensor(ov::element::Type(ovmsPrecisionToIE2Precision(toInferenceEnginePrecision(tensor->precision))), ov::Shape(shape), allocator);
            break;
        case CustomNodeTensorPrecision::UNSPECIFIED:
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    }
    tensorResourcesGuard.setPersistData();
    return StatusCode::OK;
}

void CustomNodeSession::clearInputs() {
    this->inputHandler->clearInputs();
}

void CustomNodeSession::release() {
}

}  // namespace ovms
