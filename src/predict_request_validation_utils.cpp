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
#include "predict_request_validation_utils.hpp"

#include <string>
#include <sstream>

#include <spdlog/spdlog.h>

namespace ovms {

Status validateNumberOfInputs_New(const tensorflow::serving::PredictRequest& request, const size_t expectedNumberOfInputs) {
    if (request.inputs_size() > 0 && expectedNumberOfInputs == static_cast<size_t>(request.inputs_size())) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request.inputs_size();
    const std::string details = ss.str();
    SPDLOG_DEBUG("[requested endpoint:{} version:{}] Invalid number of inputs - {}", request.model_spec().name(), request.model_spec().version().value(), details);
    return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
}

Status validateAndGetInput_New(const tensorflow::serving::PredictRequest& request, const std::string& name, google::protobuf::Map<std::string, tensorflow::TensorProto>::const_iterator& it) {
    it = request.inputs().find(name);
    if (it != request.inputs().end()) {
        return StatusCode::OK;
    }
    std::stringstream ss;
    ss << "Required input: " << name;
    const std::string details = ss.str();
    SPDLOG_DEBUG("[requested endpoint:{} version:{}] Missing input with specific name - {}", request.model_spec().name(), request.model_spec().version().value(), details);
    return Status(StatusCode::INVALID_MISSING_INPUT, details);
}

Status checkIfShapeValuesNegative_New(const tensorflow::TensorProto& proto) {
    for (size_t i = 0; i < proto.tensor_shape().dim_size(); i++) {
        if (proto.tensor_shape().dim(i).size() <= 0) {
            const std::string details = "Negative or zero dimension size is not acceptable: " + TensorInfo::tensorShapeToString(proto.tensor_shape());
            SPDLOG_DEBUG("Invalid shape - {}", details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms
