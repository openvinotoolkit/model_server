//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "exit_node.hpp"

#include <string>
#include <utility>

#include "../capi_frontend/inferenceresponse.hpp"
#include "../logging.hpp"
#include "../ov_utils.hpp"
#include "../serialization.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

#include "exitnodesession.hpp"

namespace ovms {

const std::string EXIT_NODE_NAME = "response";

template <typename ResponseType>
Status ExitNode<ResponseType>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    OVMS_PROFILE_FUNCTION();
    auto& exitNodeSession = static_cast<ExitNodeSession<ResponseType>&>(nodeSession);
    return this->fetchResults(exitNodeSession.getInputTensors());
}

template <typename ResponseType>
Status ExitNode<ResponseType>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) {
    OVMS_PROFILE_FUNCTION();
    notifyEndQueue.push(NodeSessionKeyPair(*this, sessionId));
    return StatusCode::OK;
}

template <>
Status OutputGetter<const TensorMap&>::get(const std::string& name, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    auto it = outputSource.find(name);
    if (it == outputSource.end()) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Failed to find expected pipeline output when serializing response: {}", name);
        return StatusCode::INTERNAL_ERROR;
    }
    tensor = it->second;
    return StatusCode::OK;
}

template <typename ResponseType>
Status ExitNode<ResponseType>::fetchResults(const TensorMap& inputTensors) {
    OutputGetter<const TensorMap&> outputGetter(inputTensors);
    static const model_version_t version{1};
    return serializePredictResponse(outputGetter, pipelineName, version, this->outputsInfo, this->response, getOutputMapKeyName, useSharedOutputContent);
}

template <typename ResponseType>
std::unique_ptr<NodeSession> ExitNode<ResponseType>::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) {
    return std::make_unique<ExitNodeSession<ResponseType>>(metadata, getName(), previous.size(), collapsingDetails, response);
}

template class ExitNode<InferenceResponse>;
template Status ExitNode<::KFSResponse>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status ExitNode<tensorflow::serving::PredictResponse>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status ExitNode<::KFSResponse>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status ExitNode<tensorflow::serving::PredictResponse>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status ExitNode<::KFSResponse>::fetchResults(const TensorMap& inputTensors);
template Status ExitNode<tensorflow::serving::PredictResponse>::fetchResults(const TensorMap& inputTensors);
template std::unique_ptr<NodeSession> ExitNode<::KFSResponse>::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails);
template std::unique_ptr<NodeSession> ExitNode<tensorflow::serving::PredictResponse>::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails);
}  // namespace ovms
