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
#pragma once

#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>
// dx.hpp transitively includes the OpenCL C++ headers (CL/opencl.hpp), which
// trip MSVC /analyze warnings (e.g. C6011) that this build treats as errors.
// Suppress them only around the third-party include.
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 6011 6101 6386 6387 6385 6001 28182 26495)
#endif
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "./ovms.h"
#include "itensorfactory.hpp"

namespace ovms {

// Windows D3D11 analog of VAAPITensorFactory: imports a decoded GStreamer NV12
// ID3D11Texture2D plane into the model's already-compiled D3DContext, so no
// per-graph compile or private context is needed on the inference path.
class D3D11TensorFactory : public IOVTensorFactory {
    ov::intel_gpu::ocl::D3DContext& d3dContext;
    uint32_t planeId;

public:
    D3D11TensorFactory(ov::intel_gpu::ocl::D3DContext& d3dContext, OVMS_BufferType type);

    // data is the ID3D11Texture2D* for the decoded NV12 surface.
    ov::Tensor create(ov::element::Type_t type, const ov::Shape& shape, const void* data) override;
};
}  // namespace ovms
