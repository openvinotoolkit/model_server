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
#include "ov_utils.hpp"

#include <functional>
#include <memory>
#include <sstream>

#include <spdlog/spdlog.h>

#include "tensorinfo.hpp"

namespace ovms {
ov::runtime::Tensor createSharedTensor(ov::element::Type_t precision, const shape_t& shape, void* data) {
    auto tensor = ov::runtime::Tensor(precision, shape);
    std::memcpy(tensor.data(), data, std::accumulate(shape.begin(), shape.end(), ov::element::Type(precision).size(), std::multiplies<size_t>()));
    return tensor;
}

Status createSharedTensor(ov::runtime::Tensor& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape) {
    destinationTensor = ov::runtime::Tensor(precision, shape);
    return StatusCode::OK;
}

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<TensorInfo>>& inputsInfo) {
    std::stringstream stringStream;
    for (const auto& pair : inputsInfo) {
        const auto& name = pair.first;
        auto inputInfo = pair.second;
        auto precision = inputInfo->getPrecision();
        auto layout = inputInfo->getLayout();
        auto shape = inputInfo->getShape();

        stringStream << "\nname: " << name
                     << "; mapping: " << inputInfo->getMappedName()
                     << "; shape: " << shape.toString()
                     << "; precision: " << TensorInfo::getPrecisionAsString(precision)
                     << "; layout: " << TensorInfo::getStringFromLayout(layout);
    }
    return stringStream.str();
}

Status tensorClone(ov::runtime::Tensor& destinationTensor, const ov::runtime::Tensor& sourceTensor) {
    destinationTensor = ov::runtime::Tensor(sourceTensor.get_element_type(), sourceTensor.get_shape());

    if (destinationTensor.get_byte_size() != sourceTensor.get_byte_size()) {
        SPDLOG_ERROR("tensorClone byte size mismatch destination:{}; source:{}",
            destinationTensor.get_byte_size(),
            sourceTensor.get_byte_size());
        return StatusCode::OV_CLONE_TENSOR_ERROR;
    }
    std::memcpy(destinationTensor.data(), sourceTensor.data(), sourceTensor.get_byte_size());
    return StatusCode::OK;
}

}  // namespace ovms
