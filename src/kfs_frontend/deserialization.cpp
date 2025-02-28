//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "deserialization.hpp"

#include "../itensorfactory.hpp"
#include "../logging.hpp"

namespace ovms {
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo,
    const std::string& buffer) {
    OVMS_PROFILE_FUNCTION();
    OV_LOGGER("ov::Shape()");
    ov::Shape shape;
    for (int i = 0; i < requestInput.shape_size(); i++) {
        OV_LOGGER("ov::Shape::push_back({})", requestInput.shape().at(i));
        shape.push_back(requestInput.shape().at(i));
    }
    ov::element::Type precision = tensorInfo->getOvPrecision();
    if (!buffer.size()) {
        OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
        return ov::Tensor(precision, shape);
    }
    OV_LOGGER("ov::Tensor({}, shape, data)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
    return ov::Tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(buffer.data())));
}
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    OV_LOGGER("ov::Shape()");
    ov::Shape shape;
    for (int i = 0; i < requestInput.shape_size(); i++) {
        OV_LOGGER("ov::Shape::push_back({})", requestInput.shape().at(i));
        shape.push_back(requestInput.shape().at(i));
    }

    ov::element::Type precision = tensorInfo->getOvPrecision();
    OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
    ov::Tensor tensor(precision, shape);
    return tensor;
}
}  // namespace ovms
