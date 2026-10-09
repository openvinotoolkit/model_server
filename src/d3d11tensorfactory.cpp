//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "d3d11tensorfactory.hpp"

#include <stdexcept>

#include <openvino/openvino.hpp>

#include "logging.hpp"
namespace ovms {

static uint32_t getD3D11PlaneId(OVMS_BufferType bufferType) {
    if (bufferType == OVMS_BUFFERTYPE_D3D11_TEXTURE_Y)
        return 0;
    if (bufferType == OVMS_BUFFERTYPE_D3D11_TEXTURE_UV)
        return 1;
    throw std::runtime_error("Unsupported buffer type in D3D11TensorFactory");
}

D3D11TensorFactory::D3D11TensorFactory(ov::intel_gpu::ocl::D3DContext& d3dContext, OVMS_BufferType type) :
    d3dContext(d3dContext),
    planeId(getD3D11PlaneId(type)) {
}

ov::Tensor D3D11TensorFactory::create(ov::element::Type_t type, const ov::Shape& shape, const void* data) {
    SPDLOG_TRACE("create ov::Tensor from D3DContext with texture: {}", data);
    ID3D11Texture2D* surface = reinterpret_cast<ID3D11Texture2D*>(const_cast<void*>(data));
    return this->d3dContext.create_tensor(type, shape, surface, this->planeId);
}
}  // namespace ovms
