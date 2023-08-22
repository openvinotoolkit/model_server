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
#include "custom_node_output_allocator.hpp"

#include "../custom_node_interface.h"  // NOLINT
#include "../logging.hpp"

namespace ovms {

bool operator==(const CustomNodeTensor& t1, const CustomNodeTensor& t2) {
    return (t1.name == t2.name) &&
           (t1.data == t2.data) &&
           (t1.dataBytes == t2.dataBytes) &&
           (t1.dims == t2.dims) &&
           (t1.dimsCount == t2.dimsCount) &&
           (t1.precision == t2.precision);
}
CustomNodeOutputAllocator::CustomNodeOutputAllocator(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary, void* customNodeLibraryInternalManager) :
    tensor(tensor),
    nodeLibrary(nodeLibrary),
    customNodeLibraryInternalManager(customNodeLibraryInternalManager) {}
void* CustomNodeOutputAllocator::allocate(const size_t bytes, const size_t alignment) {
    return (void*)tensor.data;
}
void CustomNodeOutputAllocator::deallocate(void* handle, const size_t bytes, size_t alignment) {
    bool succeeded = nodeLibrary.release(tensor.data, customNodeLibraryInternalManager) == 0;
    if (false == succeeded) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to release custom node tensor:{} buffer using library:{}", tensor.name, nodeLibrary.basePath);
    }
}
bool CustomNodeOutputAllocator::is_equal(const CustomNodeOutputAllocator& other) const {
    return (customNodeLibraryInternalManager == other.customNodeLibraryInternalManager) &&
           (nodeLibrary == other.nodeLibrary) &&
           (tensor == other.tensor);
}

}  // namespace ovms
