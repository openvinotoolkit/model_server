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

#include <openvino/openvino.hpp>

#include "../custom_node_interface.h"  // NOLINT
#include "node_library.hpp"

namespace ovms {

bool operator==(const CustomNodeTensor& t1, const CustomNodeTensor& t2);

class CustomNodeOutputAllocator {
    struct ::CustomNodeTensor tensor;
    NodeLibrary nodeLibrary;
    void* customNodeLibraryInternalManager;

public:
    CustomNodeOutputAllocator(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary, void* customNodeLibraryInternalManager);
    void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t));
    void deallocate(void* handle, const size_t bytes, size_t alignment = alignof(max_align_t));
    bool is_equal(const CustomNodeOutputAllocator& other) const;
};

}  // namespace ovms
