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
#include <utility>

#include "custom_node_output_allocator.hpp"
#include "logging.hpp"
#include "node.hpp"
#include "node_library.hpp"
#include "node_library_utils.hpp"
#include "nodeinputhandler.hpp"
#include "pipelineeventqueue.hpp"
#include "timer.hpp"

namespace ovms {

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(metadata, nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(std::move(metadata), nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::~CustomNodeSession() = default;

Status CustomNodeSession::execute(PipelineEventQueue& notifyEndQueue, Node& node, const NodeLibrary& library, std::unique_ptr<struct CustomNodeParam[]>& parameters, int parametersCount, void* customNodeLibraryInternalManager) {
    const auto& blobMap = this->inputHandler->getInputs();
    auto inputTensorsCount = blobMap.size();
    auto inputTensors = createCustomNodeTensorArray(blobMap);
    struct CustomNodeTensor* outputTensors = nullptr;
    int outputTensorsCount = 0;

    this->timer->start("execution");
    int result = library.execute(
        inputTensors.get(),
        inputTensorsCount,
        &outputTensors,
        &outputTensorsCount,
        parameters.get(),
        parametersCount,
        customNodeLibraryInternalManager);
    this->timer->stop("execution");
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Custom node execution processing time for node {}; session: {} - {} ms",
        this->getName(),
        this->getSessionKey(),
        this->timer->elapsed<std::chrono::microseconds>("execution") / 1000);

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
    // There will be memory leak if any tensor is not converted into blob.
    // Blob destructor is responsible for cleaning up resources.
    Status status = StatusCode::OK;
    for (int i = 0; i < outputTensorsCount; i++) {
        InferenceEngine::Blob::Ptr resultBlob;
        auto result = this->createBlob(&outputTensors[i], resultBlob, library, customNodeLibraryInternalManager);
        if (outputTensors[i].name == nullptr) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed blob conversion - missing output name", getName(), getSessionKey());
            status = StatusCode::NODE_LIBRARY_OUTPUT_MISSING_NAME;
            continue;
        }
        if (!result.ok()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed to convert {}: to blob", getName(), getSessionKey(), outputTensors[i].name);
            if (status.ok()) {
                status = result;
            }
            continue;
        }
        this->resultBlobs.emplace(std::string(outputTensors[i].name), std::move(resultBlob));
    }

    library.release(outputTensors, customNodeLibraryInternalManager);
    notifyEndQueue.push({node, getSessionKey()});
    return status;
}

Status CustomNodeSession::fetchResult(const std::string& name, InferenceEngine::Blob::Ptr& resultBlob) {
    auto it = resultBlobs.find(name);
    if (it == resultBlobs.end()) {
        return StatusCode::NODE_LIBRARY_MISSING_OUTPUT;
    }
    resultBlob = it->second;
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

Status CustomNodeSession::createBlob(const struct CustomNodeTensor* tensor, InferenceEngine::Blob::Ptr& resultBlob, const NodeLibrary& library, void* customNodeLibraryInternalManager) {
    TensorResourcesGuard tensorResourcesGuard(tensor, library, customNodeLibraryInternalManager);
    InferenceEngine::TensorDesc desc;

    auto precision = toInferenceEnginePrecision(tensor->precision);
    if (precision == Precision::UNDEFINED) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; Unspecified output precision:{} from custom node tensor: {}",
            this->getName(),
            this->getSessionKey(),
            precision,
            tensor->name);
        return StatusCode::NODE_LIBRARY_INVALID_PRECISION;
    }
    desc.setPrecision(ovmsPrecisionToIE1Precision(precision));

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
    InferenceEngine::SizeVector shape(tensor->dims, tensor->dims + tensor->dimsCount);
    desc.setDims(shape);

    size_t expectedElementsCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    size_t expectedDataLength = expectedElementsCount *= ovmsPrecisionToIE1Precision(precision).size();
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
    auto allocator = std::make_shared<CustomNodeOutputAllocator>(*tensor, library, customNodeLibraryInternalManager);
    //auto allocator = std::make_shared<CustomNodeOutputAllocator_2>(*tensor, library, customNodeLibraryInternalManager);
    try {
        switch (tensor->precision) {
        case CustomNodeTensorPrecision::FP32:
            resultBlob = InferenceEngine::make_shared_blob<float>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I32:
            resultBlob = InferenceEngine::make_shared_blob<int32_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I8:
            resultBlob = InferenceEngine::make_shared_blob<int8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::U8:
            resultBlob = InferenceEngine::make_shared_blob<uint8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::FP16:
            resultBlob = InferenceEngine::make_shared_blob<uint16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I16:
            resultBlob = InferenceEngine::make_shared_blob<int16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::U16:
            resultBlob = InferenceEngine::make_shared_blob<uint16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::UNSPECIFIED:
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (const InferenceEngine::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    }

    resultBlob->allocate();
    tensorResourcesGuard.setPersistData();
    return StatusCode::OK;
}

void CustomNodeSession::clearInputs() {
    this->inputHandler->clearInputs();
}

void CustomNodeSession::release() {
}

}  // namespace ovms
