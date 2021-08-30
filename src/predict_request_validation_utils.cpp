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

Status validateAndGetInput_New(const tensorflow::serving::PredictRequest& request, const std::string& name, tensorflow::TensorProto& proto) {
    auto& it = request.inputs().find(name);

    if (it != request.inputs().end()) {
        proto = it->second;
        return StatusCode::OK;
    }
    {
        std::stringstream ss;
        ss << "Required input: " << name;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model: {} version: {}] Missing input with specific name - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_MISSING_INPUT, details);
    }
}

Status checkIfShapeValuesNegative_New(const tensorflow::serving::PredictRequest& request) {

}

}  // namespace ovms
