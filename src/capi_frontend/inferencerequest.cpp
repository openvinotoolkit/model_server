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

#include "../status.hpp"

namespace ovms {
// this constructor can be removed with prediction tests overhaul
InferenceRequest::InferenceRequest() :
    InferenceRequest("CONSTRUCTOR_USED_ONLY_IN_PREDICTION_TESTS", 42) {}
InferenceRequest::InferenceRequest(const char* servableName, model_version_t servableVersion) :
    servableName(servableName),
    servableVersion(servableVersion) {
}

const std::string& InferenceRequest::getServableName() const {
    return this->servableName;
}
model_version_t InferenceRequest::getServableVersion() const {
    return this->servableVersion;
}
Status InferenceRequest::addInput(const char* name, OVMS_DataType datatype, const int64_t* shape, size_t dimCount) {
    auto [it, emplaced] = inputs.emplace(name, InferenceTensor{datatype, shape, dimCount});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_TENSOR_INSERT;
}
Status InferenceRequest::setInputBuffer(const char* name, const void* addr, size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId) {
    auto it = inputs.find(name);
    if (it == inputs.end()) {
        return StatusCode::NONEXISTENT_TENSOR_FOR_SET_BUFFER;
    }
    return it->second.setBuffer(addr, byteSize, bufferType, deviceId);
}
Status InferenceRequest::removeInputBuffer(const char* name) {
    auto it = inputs.find(name);
    if (it == inputs.end()) {
        return StatusCode::NONEXISTENT_TENSOR_FOR_REMOVE_BUFFER;
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
        return StatusCode::NONEXISTENT_TENSOR;
    }
    *tensor = &it->second;
    return StatusCode::OK;
}
uint64_t InferenceRequest::getInputsSize() const {
    return inputs.size();
}
Status InferenceRequest::removeInput(const char* name) {
    auto count = inputs.erase(name);
    if (count) {
        return StatusCode::OK;
    }
    return StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL;
}
Status InferenceRequest::addParameter(const char* parameterName, OVMS_DataType datatype, const void* data) {
    auto [it, emplaced] = parameters.emplace(parameterName, InferenceParameter{parameterName, datatype, data});
    return emplaced ? StatusCode::OK : StatusCode::DOUBLE_PARAMETER_INSERT;
}
Status InferenceRequest::removeParameter(const char* name) {
    auto count = parameters.erase(name);
    if (count) {
        return StatusCode::OK;
    }
    return StatusCode::NONEXISTENT_PARAMETER;
}
const InferenceParameter* InferenceRequest::getParameter(const char* name) const {
    auto it = parameters.find(name);
    if (it != parameters.end())
        return &it->second;
    return nullptr;
}

// Assuming the request is already validated, therefore no need to check for negative values or zeros
Status InferenceRequest::getBatchSize(size_t& batchSize, size_t batchSizeIndex) const {
    if (inputs.size() == 0) {
        return StatusCode::INTERNAL_ERROR;
    }
    // we make here the same assumption as with bs=auto in TFS/KFS API
    const InferenceTensor& tensor = inputs.begin()->second;
    const auto& shape = tensor.getShape();
    if (batchSizeIndex >= shape.size()) {
        return StatusCode::INTERNAL_ERROR;
    }
    batchSize = shape[batchSizeIndex];
    return StatusCode::OK;
}

// Assuming the request is already validated, therefore no need to check for negative values or zeros
std::map<std::string, shape_t> InferenceRequest::getRequestShapes() const {
    std::map<std::string, shape_t> result;
    for (auto& [name, tensor] : inputs) {
        result.emplace(name, shape_t(
                                 reinterpret_cast<shape_t::const_pointer>(tensor.getShape().data()),
                                 reinterpret_cast<shape_t::const_pointer>(tensor.getShape().data() + tensor.getShape().size())));
    }
    return result;
}
}  // namespace ovms
