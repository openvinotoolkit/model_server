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
#include "../tfs_frontend/tfs_utils.hpp"
#include "../tfs_frontend/deserialization.hpp"

#pragma warning(push)
#pragma warning(disable : 4624 6001 6385 6386 6326 6011 4457 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "entry_node_impl.hpp"

namespace ovms {

template Status EntryNode<tensorflow::serving::PredictRequest>::execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue);
template Status EntryNode<tensorflow::serving::PredictRequest>::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs);
template Status EntryNode<tensorflow::serving::PredictRequest>::createShardedTensor(ov::Tensor& dividedTensor, Precision precision, const shape_t& shape, const ov::Tensor& tensor, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string tensorName);
template const Status EntryNode<tensorflow::serving::PredictRequest>::validate();

}  //  namespace ovms
