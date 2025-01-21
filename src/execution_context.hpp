//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <cstdint>

namespace ovms {

struct ExecutionContext {
    enum class Interface : uint8_t {
        GRPC,
        REST,
    };
    enum class Method : uint8_t {
        // TensorflowServing
        Predict,
        GetModelMetadata,
        GetModelStatus,

        // Model Control API
        ConfigReload,
        ConfigStatus,

        // KServe
        ModelInfer,
        ModelInferStream,
        ModelReady,
        ModelMetadata,

        // V3
        V3Unary,
        V3Stream,
    };

    Interface interface;
    Method method;

    ExecutionContext(Interface interface, Method method) :
        interface(interface),
        method(method) {}
};

}  // namespace ovms
