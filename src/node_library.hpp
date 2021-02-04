//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <inference_engine.hpp>

#include "custom_node_interface.h"  // NOLINT

namespace ovms {

CustomNodeTensorPrecision toCustomNodeTensorPrecision(InferenceEngine::Precision precision);
InferenceEngine::Precision toInferenceEnginePrecision(CustomNodeTensorPrecision precision);

typedef int (*execute_fn)(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int);
typedef int (*release_fn)(struct CustomNodeTensor*);

struct NodeLibrary {
    execute_fn execute = nullptr;
    release_fn releaseBuffer = nullptr;
    release_fn releaseTensors = nullptr;

    bool isValid() const {
        return execute != nullptr && releaseBuffer != nullptr && releaseTensors != nullptr;
    }
};

}  // namespace ovms
