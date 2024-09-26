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
#include "regularovtensorfactory.hpp"

#include <openvino/openvino.hpp>

#include "logging.hpp"
namespace ovms {
ov::Tensor RegularOVTensorFactory::create(ov::element::Type_t type, const ov::Shape& shape, const void* data) {
    SPDLOG_TRACE("create regular ov::Tensor buffer: {}", data);
    OV_LOGGER("ov::Tensor({}, shape:{}, data:{})", type, (void*)&shape, data);
    return ov::Tensor(type, shape, (void*)data);  // TODO cast
}
}  // namespace ovms
