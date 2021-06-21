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
#include "node_library.hpp"

namespace ovms {

class CustomNodeOutputAllocator : public InferenceEngine::IAllocator {
    struct CustomNodeTensor tensor;
    NodeLibrary nodeLibrary;

public:
    CustomNodeOutputAllocator(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary) :
        tensor(tensor),
        nodeLibrary(nodeLibrary) {}

    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* a) noexcept override {}

    void* alloc(size_t size) noexcept override {
        return (void*)tensor.data;
    }

    bool free(void* handle) noexcept override {
        return nodeLibrary.release(tensor.data) == 0;
    }
};

}  // namespace ovms
