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

#include <spdlog/spdlog.h>

namespace ovms {

Status ExitNode::fetchResults(BlobMap&) {
    // Serialize results to proto
    for (const auto& kv : this->input_blobs) {
        const auto& output_name = kv.first;
        auto& blob = kv.second;

        // Hardcoded precision for now
        tensorflow::TensorProto& proto = (*this->response->mutable_outputs())[output_name];
        proto.set_dtype(tensorflow::DataType::DT_INT8);

        auto description = blob->getTensorDesc();
        for (int dim : description.getDims()) {
            proto.mutable_tensor_shape()->add_dim()->set_size(dim);
        }
        // proto.mutable_tensor_content()->assign((char*)blob->buffer(), blob->byteSize());

        SPDLOG_INFO("ExitNode::fetchResults (Node name {}): serialize blob to proto: blob name [{}]", getName(), output_name);
    }

    return StatusCode::OK;
}

}  // namespace ovms
