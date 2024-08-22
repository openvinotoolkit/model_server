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
#include "opencltensorfactory.hpp"

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#include "logging.hpp"
namespace ovms {

OpenCLTensorFactory::OpenCLTensorFactory(ov::intel_gpu::ocl::ClContext& ovOclContext) :
    ovOclContext(ovOclContext) {
}

ov::Tensor OpenCLTensorFactory::create(ov::element::Type_t type, const ov::Shape& shape, const void* data) {
    SPDLOG_TRACE("create ov::Tensor from context with buffer: {}", data);
    OV_LOGGER("ov::intel_gpu::ocl::ClContext: {}, ov::intel_gpu::ocl::ClContext::create_tensor(type:{}, shape:{}, data:{})", (void*)&ovOclContext, type, (void*)&shape, data);
    return ovOclContext.create_tensor(type, shape, *(reinterpret_cast<const cl::Buffer*>(data)));
}
}  // namespace ovms
