//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "../kfs_frontend/serialization.hpp"

#include "exit_node_impl.hpp"

namespace ovms {

template Status ExitNode<::KFSResponse>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status ExitNode<::KFSResponse>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status ExitNode<::KFSResponse>::fetchResults(const TensorMap& inputTensors);
template std::unique_ptr<NodeSession> ExitNode<::KFSResponse>::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails);

}  // namespace ovms
