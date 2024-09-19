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
#pragma once

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#include "./ovms.h"
#include "itensorfactory.hpp"
namespace ov::intel_gpu::ocl {
class VAContext;
}
namespace ovms {

class VAAPITensorFactory : public IOVTensorFactory {
    ov::intel_gpu::ocl::VAContext& vaContext;
    uint32_t vaPlaneId;

public:
    VAAPITensorFactory(ov::intel_gpu::ocl::VAContext& vaContext, OVMS_BufferType type);

    /**
     * Create tensor and interpret data ptr appropriately depending on the
     * factory type.
     */
    ov::Tensor create(ov::element::Type_t type, const ov::Shape& shape, const void* data) override;
};
}  // namespace ovms
