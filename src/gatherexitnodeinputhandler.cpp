//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "gatherexitnodeinputhandler.hpp"

#include <utility>

#include "logging.hpp"
#include "status.hpp"

namespace ovms {

Status prepareConsolidatedTensorImpl(tensorflow::serving::PredictResponse* response, char*& bufferOut, const std::string& name, size_t size) {
    OVMS_PROFILE_FUNCTION();
    tensorflow::TensorProto tensorProto;
    auto [it, isInserted] = response->mutable_outputs()->insert(google::protobuf::MapPair<std::string, tensorflow::TensorProto>(name, std::move(tensorProto)));
    if (!isInserted) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, tensor with name {} already prepared", name);
        return StatusCode::INTERNAL_ERROR;
    }
    it->second.mutable_tensor_content()->resize(size);
    bufferOut = it->second.mutable_tensor_content()->data();
    return StatusCode::OK;
}

Status prepareConsolidatedTensorImpl(::inference::ModelInferResponse* response, char*& bufferOut, const std::string& name, size_t size) {
    OVMS_PROFILE_FUNCTION();
    for (int i = 0; i < response->outputs_size(); i++) {
        if (response->mutable_outputs(i)->name() == name) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, tensor with name {} already prepared", name);
            return StatusCode::INTERNAL_ERROR;
        }
    }
    auto* proto = response->add_outputs();
    proto->set_name(name);
    auto* content = response->add_raw_output_contents();
    content->resize(size);
    bufferOut = content->data();
    return StatusCode::OK;
}

}  // namespace ovms
