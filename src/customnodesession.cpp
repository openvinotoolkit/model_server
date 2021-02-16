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

namespace ovms {

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(metadata, nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    NodeSession(std::move(metadata), nodeName, inputsCount, collapsingDetails) {}

CustomNodeSession::~CustomNodeSession() = default;

Status CustomNodeSession::execute(PipelineEventQueue& notifyEndQueue, Node& node, const NodeLibrary& library, std::unique_ptr<struct CustomNodeParam[]>& parameters, int parametersLength) {
    const auto& blobMap = this->inputHandler->getInputs();
    auto inputTensorsLength = blobMap.size();
    auto inputTensors = createCustomNodeTensorArray(blobMap);
    struct CustomNodeTensor* outputTensors = nullptr;
    int outputTensorsLength = 0;

    int result = library.execute(
        inputTensors.get(),
        inputTensorsLength,
        &outputTensors,
        &outputTensorsLength,
        parameters.get(),
        parametersLength);

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

    if (outputTensorsLength <= 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted number of outputs", getName(), getSessionKey());
        library.release(outputTensors);
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT;
    }

    // At this point this is important we do not exit before finishing the loop.
    // There will be memory leak if any tensor is not converted into blob.
    // Blob destructor is responsible for cleaning up resources.
    Status status = StatusCode::OK;
    for (int i = 0; i < outputTensorsLength; i++) {
        InferenceEngine::Blob::Ptr resultBlob;
        auto result = this->createBlob(&outputTensors[i], resultBlob, library);
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

    library.release(outputTensors);

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

void CustomNodeSession::releaseTensorResources(const struct CustomNodeTensor* tensor, const NodeLibrary& library) {
    if (tensor->data) {
        library.release(tensor->data);
    }
    if (tensor->dims) {
        library.release(tensor->dims);
    }
}

class TensorResourcesGuard {
    const struct CustomNodeTensor* tensor;
    const NodeLibrary& library;
    bool persistData = false;

public:
    TensorResourcesGuard(const struct CustomNodeTensor* tensor, const NodeLibrary& library) :
        tensor(tensor),
        library(library) {}
    ~TensorResourcesGuard() {
        if (tensor->data && !persistData) {
            library.release(tensor->data);
        }
        if (tensor->dims) {
            library.release(tensor->dims);
        }
    }
    void setPersistData() {
        this->persistData = true;
    }
};

Status CustomNodeSession::createBlob(const struct CustomNodeTensor* tensor, InferenceEngine::Blob::Ptr& resultBlob, const NodeLibrary& library) {
    TensorResourcesGuard tensorResourcesGuard(tensor, library);
    InferenceEngine::TensorDesc desc;

    InferenceEngine::Precision precision = toInferenceEnginePrecision(tensor->precision);
    if (precision == InferenceEngine::Precision::UNSPECIFIED) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; Unspecified output precision from custom node",
            this->getName(),
            this->getSessionKey());
        return StatusCode::NODE_LIBRARY_INVALID_PRECISION;
    }
    desc.setPrecision(precision);

    if (tensor->dims == nullptr || tensor->dimsLength == 0) {
        std::string error;
        if (tensor->dims == nullptr) {
            error = "shape handle is null";
        } else if (tensor->dimsLength == 0) {
            error = "shape dimensions number is equal to 0";
        }
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; error: {}",
            this->getName(),
            this->getSessionKey(),
            error);
        return StatusCode::NODE_LIBRARY_INVALID_SHAPE;
    }
    InferenceEngine::SizeVector shape(tensor->dims, tensor->dims + tensor->dimsLength);
    desc.setDims(shape);

    size_t expectedElementsCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    size_t expectedDataLength = expectedElementsCount *= precision.size();
    if (tensor->data == nullptr || tensor->dataLength != expectedDataLength) {
        std::stringstream error;
        if (tensor->data == nullptr) {
            error << "data handle is null";
        } else if (tensor->dataLength != expectedDataLength) {
            error << "not expected data length: expected: " << expectedDataLength << " vs " << tensor->dataLength;
        }
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; error: {}",
            this->getName(),
            this->getSessionKey(),
            error.str());
        return StatusCode::NODE_LIBRARY_INVALID_CONTENT_SIZE;
    }
    auto allocator = std::make_shared<CustomNodeOutputAllocator>(*tensor, library);
    try {
        switch (tensor->precision) {
        case CustomNodeTensorPrecision::FP32:
            resultBlob = InferenceEngine::make_shared_blob<float>(desc, std::move(allocator));
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
        case CustomNodeTensorPrecision::I32:
            resultBlob = InferenceEngine::make_shared_blob<int32_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I8:
            resultBlob = InferenceEngine::make_shared_blob<int8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::U8:
            resultBlob = InferenceEngine::make_shared_blob<uint8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::UNSPECIFIED:
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
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
