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
#include "vaapitensorfactory.hpp"

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>

#include "logging.hpp"
namespace ovms {

static uint32_t getVaPlaneId(OVMS_BufferType bufferType) {
    if (bufferType == OVMS_BUFFERTYPE_VASURFACE_Y)
        return 0;
    if (bufferType == OVMS_BUFFERTYPE_VASURFACE_UV)
        return 1;
    throw std::runtime_error("Unsupported buffer type in VAAPITensorFactory");
    return -1;
}

VAAPITensorFactory::VAAPITensorFactory(ov::intel_gpu::ocl::VAContext& vaContext, OVMS_BufferType type) :
    vaContext(vaContext),
    vaPlaneId(getVaPlaneId(type)) {
}

ov::Tensor VAAPITensorFactory::create(ov::element::Type_t type, const ov::Shape& shape, const void* data) {
    SPDLOG_TRACE("create ov::Tensor from context with buffer: {}", data);
    OV_LOGGER("ov::AnyMap() {{{{{}, {}}}, {{{}, {}}}, {{{}, {}}}}}",
        ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE,
        ov::intel_gpu::dev_object_handle.name(), static_cast<cl_uint>(reinterpret_cast<intptr_t>(data)),
        ov::intel_gpu::va_plane.name(), this->vaPlaneId);
    ov::AnyMap tensor_params = {
        {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
        {ov::intel_gpu::dev_object_handle.name(), static_cast<cl_uint>(reinterpret_cast<intptr_t>(data))},
        {ov::intel_gpu::va_plane.name(), this->vaPlaneId}};
    OV_LOGGER("ov::intel_gpu::ocl::VAContext: {}, ov::intel_gpu::ocl::VAContext::create_tensor(element::type: {}, shape: {}, data: {})", (void*)&vaContext, type, (void*)&shape, data);
    return vaContext.create_tensor(type, shape, tensor_params);
}
}  // namespace ovms
