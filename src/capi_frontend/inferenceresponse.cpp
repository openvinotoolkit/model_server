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
#include "inferenceresponse.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

#include "../logging.hpp"
#include "../modelversion.hpp"
#include "../status.hpp"
#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"

namespace ovms {

const std::string RESPONSE_SERVABLE_NAME_USED_ONLY_IN_PREDICTION_TESTS = "CONSTRUCTOR_USED_ONLY_IN_PREDICTION_TESTS";

// this constructor can be removed with prediction tests overhaul
InferenceResponse::InferenceResponse() :
    InferenceResponse(RESPONSE_SERVABLE_NAME_USED_ONLY_IN_PREDICTION_TESTS, 42) {}
InferenceResponse::InferenceResponse(const std::string& servableName, model_version_t servableVersion) :
    servableName(servableName),
    servableVersion(servableVersion) {}
const std::string& InferenceResponse::getServableName() const {
    return this->servableName;
}

model_version_t InferenceResponse::getServableVersion() const {
    return this->servableVersion;
}

Status InferenceResponse::addOutput(const std::string& name, OVMS_DataType datatype, const int64_t* shape, size_t dimCount) {
    auto it = std::find_if(outputs.begin(),
        outputs.end(),
        [&name](const std::pair<std::string, InferenceTensor>& pair) {
            return name == pair.first;
        });
    if (outputs.end() != it) {
        return StatusCode::DOUBLE_TENSOR_INSERT;
    }

    auto pair = std::pair<std::string, InferenceTensor>(name, InferenceTensor{datatype, shape, dimCount});
    outputs.push_back(std::move(pair));
    SPDLOG_LOGGER_DEBUG(capi_logger, "Successfully added tensor: {}; to servable:{} version: {} response", name, getServableName(), getServableVersion());
    return StatusCode::OK;
}

Status InferenceResponse::getOutput(uint32_t id, const std::string** name, const InferenceTensor** tensor) const {
    if (outputs.size() <= id) {
        *tensor = nullptr;
        SPDLOG_LOGGER_DEBUG(capi_logger, "Could not find tensor: {}; in servable:{} version: {} response", id, getServableName(), getServableVersion());
        return StatusCode::NONEXISTENT_TENSOR;
    }
    *name = &(outputs[id].first);
    *tensor = &(outputs[id].second);
    return StatusCode::OK;
}

Status InferenceResponse::getOutput(uint32_t id, const std::string** name, InferenceTensor** tensor) {
    return const_cast<const InferenceResponse*>(this)->getOutput(id, name, const_cast<const InferenceTensor**>(tensor));
}

Status InferenceResponse::addParameter(const char* parameterName, OVMS_DataType datatype, const void* data) {
    auto it = std::find_if(parameters.begin(),
        parameters.end(),
        [&parameterName](const InferenceParameter& parameter) {
            return parameterName == parameter.getName();
        });
    if (parameters.end() != it) {
        return StatusCode::DOUBLE_PARAMETER_INSERT;
    }
    parameters.emplace_back(parameterName, datatype, data);
    return StatusCode::OK;
}

const InferenceParameter* InferenceResponse::getParameter(uint32_t id) const {
    if (id >= parameters.size()) {
        return nullptr;
    }
    return &parameters[id];
}

uint32_t InferenceResponse::getOutputCount() const {
    return this->outputs.size();
}

uint32_t InferenceResponse::getParameterCount() const {
    return this->parameters.size();
}

void InferenceResponse::Clear() {
    outputs.clear();
    parameters.clear();
}
}  // namespace ovms
