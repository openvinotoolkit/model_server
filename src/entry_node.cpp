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
#include "entry_node.hpp"

#include <spdlog/spdlog.h>

namespace ovms {

Status EntryNode::fetchResults(BlobMap& outputs) {
    for (auto& kv : this->request->inputs()) {
        const auto& name = kv.first;
        const auto& tensor_proto = kv.second;

        // Retrieve shape in OV format
        InferenceEngine::SizeVector shape;
        for (int i = 0; i < tensor_proto.tensor_shape().dim_size(); i++) {
            shape.emplace_back(tensor_proto.tensor_shape().dim(i).size());
        }

        // Make shared blob
        // - hardcoded precision for now
        // - not validated for buffer overflow and precision
        InferenceEngine::TensorDesc description{InferenceEngine::Precision::I8, shape, InferenceEngine::Layout::ANY};
        outputs[name] = InferenceEngine::make_shared_blob<int8_t>(description, (int8_t*)(tensor_proto.tensor_content().data()));

        SPDLOG_INFO("EntryNode::fetchResults (deserialization) (Node name: {}): blob with name [{}] has been prepared", getName(), name);
    }

    return StatusCode::OK;
}

}  // namespace ovms
