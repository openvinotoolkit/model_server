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

#include "itensorfactory.hpp"
#include "logging.hpp"
namespace ovms {

class OpenCLTensorFactory : public IOVTensorFactory {
    ov::intel_gpu::ocl::ClContext& ovOclContext;

public:
    OpenCLTensorFactory(ov::intel_gpu::ocl::ClContext& ovOclContext) :
        ovOclContext(ovOclContext) {
    }

    /**
     * Create tensor and intepret data ptr appropiately depending on the
     * factory type.
     */
    virtual ov::Tensor create(ov::element::Type_t type, const ov::Shape& shape, const void* data) override {
        SPDLOG_TRACE("create ov::Tensor from context with buffer: {}", data);  // TODO OVTRACING
        return ovOclContext.create_tensor(type, shape, *(reinterpret_cast<const cl::Buffer*>(data)));
    }
    virtual ov::Tensor create(ov::element::Type_t type, const ov::Shape& shape, Buffer* buffer) override {
        // FIXME TODO
        return ov::Tensor(type, shape);
    }
};
}  // namespace ovms
