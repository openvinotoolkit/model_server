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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <openvino/openvino.hpp>

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
#include "nodesession.hpp"
#include "pipelineeventqueue.hpp"
#include "tensormap.hpp"

struct CustomNodeTensor;
struct CustomNodeParam;

namespace ovms {

class Node;
struct NodeLibrary;
class Status;

class CustomNodeSession : public NodeSession {
    TensorMap resultTensors;

public:
    CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails);
    CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails);
    virtual ~CustomNodeSession();

    Status execute(PipelineEventQueue& notifyEndQueue, Node& node, const NodeLibrary& library, std::unique_ptr<struct CustomNodeParam[]>& parameters, int parametersCount, void* customNodeLibraryInternalManager) {
        OVMS_PROFILE_FUNCTION();
        const auto& tensorMap = this->inputHandler->getInputs();
        auto inputTensorsCount = tensorMap.size();
        // this is a hack to overcome OV 1.0 -> 2.0 API change where we do not get reference to
        // tensor shape now but a copy. Hence we have to extend the lifetime of dims vector
        auto tensorsDims = CustomNodeSession::createOwnedShapesCopy(tensorMap);
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
            auto creationResult = this->createTensor(&outputTensors[i], resultTensor, library, customNodeLibraryInternalManager);
            if (outputTensors[i].name == nullptr) {
                SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed tensor conversion - missing output name", getName(), getSessionKey());
                status = StatusCode::NODE_LIBRARY_OUTPUT_MISSING_NAME;
                continue;
            }
            if (!creationResult.ok()) {
                SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; failed to convert {}: to tensor", getName(), getSessionKey(), outputTensors[i].name);
                if (status.ok()) {
                    status = std::move(creationResult);
                }
                continue;
            }
            this->resultTensors.emplace(std::string(outputTensors[i].name), std::move(resultTensor));
        }

        library.release(outputTensors, customNodeLibraryInternalManager);
        notifyEndQueue.push({node, getSessionKey()});
        return status;
    }

    Status fetchResult(const std::string& name, ov::Tensor& resultTensor);

    void clearInputs();
    void release() override;

private:
    static std::unordered_map<std::string, shape_t> createOwnedShapesCopy(const TensorMap& tensorMap);
    static void releaseTensorResources(const struct CustomNodeTensor* tensor, const NodeLibrary& library, void* customNodeLibraryInternalManager);
    Status createTensor(const struct CustomNodeTensor* tensor, ov::Tensor& resultTensor, const NodeLibrary& library, void* customNodeLibraryInternalManager);
};
}  // namespace ovms
