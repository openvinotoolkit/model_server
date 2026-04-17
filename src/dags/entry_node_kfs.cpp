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
#include "../kfs_frontend/kfs_utils.hpp"
#include "../kfs_frontend/deserialization.hpp"

#include "entry_node_impl.hpp"

namespace ovms {

template Status EntryNode<::KFSRequest>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status EntryNode<::KFSRequest>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status EntryNode<::KFSRequest>::fetchResults(TensorWithSourceMap& outputs);
template Status EntryNode<::KFSRequest>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName);
template const Status EntryNode<::KFSRequest>::validate();

}  //  namespace ovms
