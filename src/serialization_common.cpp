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
#include "serialization_common.hpp"

#include "logging.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {
void serializeContent(std::string* content, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    // We only fill if the content is not already filled.
    // It can be filled in gather exit node handler.
    if (content->size() == 0) {
        content->assign((char*)tensor.data(), tensor.get_byte_size());
    }
}

void serializeStringContent(std::string* content, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    if (content->size()) {
        return;
    }

    std::string* strings = tensor.data<std::string>();
    for (size_t i = 0; i < tensor.get_shape()[0]; i++) {
        uint32_t strLen = strings[i].size();
        content->append(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
        content->append((char*)strings[i].data(), strLen);
    }
}

void serializeStringContentFrom2DU8(std::string* content, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    // We only fill if the content is not already filled.
    // It can be filled in gather exit node handler.
    if (!content->empty()) {
        return;
    }

    size_t batchSize = tensor.get_shape()[0];
    size_t maxStringLen = tensor.get_shape()[1];
    for (size_t i = 0; i < batchSize; i++) {
        uint32_t strLen = strnlen((char*)tensor.data() + i * maxStringLen, maxStringLen);
        content->append(reinterpret_cast<const char*>(&strLen), sizeof(strLen));
        content->append((char*)tensor.data() + i * maxStringLen, strLen);
    }
}
template <>
Status OutputGetter<ov::InferRequest&>::get(const std::string& name, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    try {
        OV_LOGGER("ov::InferRequest: {}, outputSource.get_tensor({})", reinterpret_cast<void*>(&outputSource), name);
        tensor = outputSource.get_tensor(name);
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_SERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}
const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo) {
    return tensorInfo.getName();
}

const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo) {
    return first;
}
// force instantiation
template class OutputGetter<ov::InferRequest&>;
}  // namespace ovms
