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

#include "buffer.hpp"
#include "../itensorfactory.hpp"
#include "../logging.hpp"
#include "../tensor_conversion.hpp"

namespace ovms {
ov::Tensor makeTensor(const InferenceTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo, const std::unordered_map<int, std::shared_ptr<IOVTensorFactory>>& factories) {
    OVMS_PROFILE_FUNCTION();
    ov::Shape shape;
    OV_LOGGER("ov::Shape(): {}", (void*)&shape);
    for (const auto& dim : requestInput.getShape()) {
        OV_LOGGER("ov::Shape::push_back({})", dim);
        shape.push_back(dim);
    }
    ov::element::Type_t precision = tensorInfo->getOvPrecision();
    if (!requestInput.getBuffer()->getByteSize()) {
        OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
        return ov::Tensor(precision, shape);
    }
    // TODO validation shouldn't allow setting unsupported memory types
    // in inputs/outputs for particular device
    // validation shouldn't allow unsupporeted buffer types
    // write test
    auto it = factories.find(requestInput.getBuffer()->getBufferType());
    if (it == factories.end()) {
        SPDLOG_ERROR("Could not find appropriate tensor factory for buffer type:{}", (unsigned int)requestInput.getBuffer()->getBufferType());
        throw std::runtime_error("Could not find appropriate tensor factory");
    }
    IOVTensorFactory* factory = it->second.get();
    return factory->create(precision, shape, requestInput.getBuffer()->data());
}

template <>
Status getTensor(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor) {
    return request.getInput(name.c_str(), tensor);
}
}  // namespace ovms
