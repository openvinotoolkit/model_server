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

#include <string>
#include <unordered_map>

#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"
#include "modelversion.hpp"
#include "status.hpp"

namespace ovms {
InferenceResponse::InferenceResponse(const std::string& servableName, uint64_t servableVersion) :
    servableName(servableName),
    servableVersion(servableVersion) {}
const std::string& InferenceResponse::getServableName() const {
    return this->servableName;
}
uint64_t InferenceResponse::getServableVersion() const {
    return this->servableVersion;
}
Status InferenceResponse::addOutput(const std::string& name, DataType datatype, const size_t* shape, size_t dimCount) {
    // TODO insert tensor with wrong shape/datatype/name/dimcount
    // TODO reuse infer response/request
    auto [it, emplaced] = outputs.emplace(name, InferenceTensor{datatype, shape, dimCount});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_TENSOR_INSERT;
}
Status InferenceResponse::getOutput(const char* name, InferenceTensor** tensor) {
    auto it = outputs.find(name);
    if (outputs.end() == it) {
        *tensor = nullptr;
        return StatusCode::NONEXISTENT_TENSOR;
    }
    *tensor = &it->second;
    return StatusCode::OK;
}
Status InferenceResponse::addParameter(const char* parameterName, DataType datatype, const void* data) {
    auto [it, emplaced] = parameters.emplace(parameterName, InferenceParameter{parameterName, datatype, data});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_PARAMETER_INSERT;
}
const InferenceParameter* InferenceResponse::getParameter(const char* name) const {
    auto it = parameters.find(name);
    if (it != parameters.end())
        return &it->second;
    return nullptr;
}
}  // namespace ovms
