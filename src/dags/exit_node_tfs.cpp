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
#include "../tfs_frontend/serialization.hpp"

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "exit_node_impl.hpp"

namespace ovms {

template Status ExitNode<tensorflow::serving::PredictResponse>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status ExitNode<tensorflow::serving::PredictResponse>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status ExitNode<tensorflow::serving::PredictResponse>::fetchResults(const TensorMap& inputTensors);
template std::unique_ptr<NodeSession> ExitNode<tensorflow::serving::PredictResponse>::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails);

}  // namespace ovms
