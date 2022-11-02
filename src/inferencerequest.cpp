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
#include "inferencerequest.hpp"

#include "status.hpp"
namespace ovms {

InferenceRequest::InferenceRequest(const char* servableName, uint64_t servableVersion) :
    servableName(servableName),
    servableVersion(servableVersion) {
}

const std::string& InferenceRequest::getServableName() const {
    return this->servableName;
}
uint64_t InferenceRequest::getServableVersion() const {
    return this->servableVersion;
}
Status InferenceRequest::addInput(const char* name, DataType datatype, const size_t* shape, size_t dimCount) {
    auto [it, emplaced] = inputs.emplace(name, InferenceTensor{datatype, shape, dimCount});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_INPUT_INSERT;
}
Status InferenceRequest::setInputBuffer(const char* name, const void* addr, size_t byteSize, BufferType bufferType, std::optional<uint32_t> deviceId) {
    auto it = inputs.find(name);
    if (it == inputs.end()) {
        return StatusCode::NONEXISTENT_INPUT_FOR_SET_BUFFER;
    }
    return it->second.setBuffer(addr, byteSize, bufferType, deviceId);
}
Status InferenceRequest::removeInputBuffer(const char* name) {
    auto it = inputs.find(name);
    if (it == inputs.end()) {
        return StatusCode::NONEXISTENT_INPUT_FOR_REMOVE_BUFFER;
    }
    return it->second.removeBuffer();
}
Status InferenceRequest::removeAllInputs() {
    inputs.clear();
    return StatusCode::OK;
}
Status InferenceRequest::getInput(const char* name, const InferenceTensor** tensor) const {
    auto it = inputs.find(name);
    if (it == inputs.end()) {
        *tensor = nullptr;
        return StatusCode::NONEXISTENT_INPUT;
    }
    *tensor = &it->second;
    return StatusCode::OK;
}
Status InferenceRequest::removeInput(const char* name) {
    auto count = inputs.erase(name);
    if (count) {
        return StatusCode::OK;
    }
    return StatusCode::NONEXISTENT_INPUT_FOR_REMOVAL;
}
Status InferenceRequest::addParameter(const char* parameterName, DataType datatype, const void* data) {
    auto [it, emplaced] = parameters.emplace(parameterName, InferenceParameter{parameterName, datatype, data});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_PARAMETER_INSERT;
}
Status InferenceRequest::removeParameter(const char* name) {
    auto count = parameters.erase(name);
    if (count) {
        return StatusCode::OK;
    }
    return StatusCode::NONEXISTENT_PARAMETER_FOR_REMOVAL;
}
const InferenceParameter* InferenceRequest::getParameter(const char* name) const {
    auto it = parameters.find(name);
    if (it != parameters.end())
        return &it->second;
    return nullptr;
}
}  // namespace ovms
