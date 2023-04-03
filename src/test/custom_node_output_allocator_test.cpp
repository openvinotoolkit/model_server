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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../dags/custom_node_output_allocator.hpp"
#include "../precision.hpp"
#include "../shape.hpp"

using namespace ovms;

class NodeLibraryCheckingReleaseCalled {
public:
    static bool releaseBufferCalled;
    static int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
    static int deinitialize(void* customNodeLibraryInternalManager);
    static int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    static int getInputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    static int getOutputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    static int release(void* ptr, void* customNodeLibraryInternalManager);
};

int NodeLibraryCheckingReleaseCalled::initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 5;
}

int NodeLibraryCheckingReleaseCalled::deinitialize(void* customNodeLibraryInternalManager) {
    return 6;
}

int NodeLibraryCheckingReleaseCalled::execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 1;
}

int NodeLibraryCheckingReleaseCalled::getInputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 2;
}

int NodeLibraryCheckingReleaseCalled::getOutputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 3;
}

int NodeLibraryCheckingReleaseCalled::release(void* ptr, void* customNodeLibraryInternalManager) {
    releaseBufferCalled = true;
    return 0;
}

bool NodeLibraryCheckingReleaseCalled::releaseBufferCalled = false;

class CustomNodeOutputAllocatorCheckingFreeCalled {
    // Since test is copying/moving the allocator to ov::Tensor class
    // we need to use counter to ensure how many times deallocate was called
    std::shared_ptr<int> freeCallCount{std::make_shared<int>(0)};
    std::shared_ptr<CustomNodeOutputAllocator> allocImpl;

public:
    CustomNodeOutputAllocatorCheckingFreeCalled(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary, void* customNodeLibraryInternalManager) :
        allocImpl(std::make_shared<CustomNodeOutputAllocator>(tensor, nodeLibrary, customNodeLibraryInternalManager)) {
    }

    ~CustomNodeOutputAllocatorCheckingFreeCalled() {
        EXPECT_EQ(*freeCallCount, 1);
    }

    void* allocate(const size_t bytes, const size_t alignment) {
        return allocImpl->allocate(bytes, alignment);
    }

    void deallocate(void* handle, const size_t bytes, size_t alignment) {
        allocImpl->deallocate(handle, bytes, alignment);
        (*freeCallCount)++;
    }

    bool is_equal(const CustomNodeOutputAllocatorCheckingFreeCalled& other) const {
        return allocImpl->is_equal(*other.allocImpl);
    }
};

TEST(CustomNodeOutputAllocator, TensorDeallocationCallsReleaseBuffer) {
    unsigned int elementsCount = 10;
    std::vector<float> data(elementsCount);
    CustomNodeTensor tensor{
        "name",
        reinterpret_cast<uint8_t*>(data.data()),
        sizeof(float) * elementsCount,
        reinterpret_cast<uint64_t*>(&elementsCount),
        1,
        CustomNodeTensorPrecision::FP32};
    NodeLibrary library{
        NodeLibraryCheckingReleaseCalled::initialize,
        NodeLibraryCheckingReleaseCalled::deinitialize,
        NodeLibraryCheckingReleaseCalled::execute,
        NodeLibraryCheckingReleaseCalled::getInputsInfo,
        NodeLibraryCheckingReleaseCalled::getOutputsInfo,
        NodeLibraryCheckingReleaseCalled::release};
    void* customNodeLibraryInternalManager = nullptr;
    CustomNodeOutputAllocatorCheckingFreeCalled alloc(tensor, library, customNodeLibraryInternalManager);
    EXPECT_FALSE(NodeLibraryCheckingReleaseCalled::releaseBufferCalled);
    {
        auto elemType = ovmsPrecisionToIE2Precision(Precision::FP32);
        shape_t shape{data.size()};
        auto tensorIE2 = std::make_shared<ov::Tensor>(elemType, shape, alloc);
    }
    EXPECT_TRUE(NodeLibraryCheckingReleaseCalled::releaseBufferCalled);
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 5;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 6;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 1;
}

int getInputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 2;
}

int getOutputsInfo(struct CustomNodeTensorInfo** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 3;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    return 0;
}

TEST(CustomNodeOutputAllocator, TensorReturnsCorrectPointer) {
    unsigned int elementsCount = 10;
    std::vector<float> data(elementsCount);
    CustomNodeTensor tensor{
        "name",
        reinterpret_cast<uint8_t*>(data.data()),
        sizeof(float) * elementsCount,
        reinterpret_cast<uint64_t*>(&elementsCount),
        1,
        CustomNodeTensorPrecision::FP32};
    NodeLibrary library{
        initialize,
        deinitialize,
        execute,
        getInputsInfo,
        getOutputsInfo,
        release};
    void* customNodeLibraryInternalManager = nullptr;
    CustomNodeOutputAllocator alloc(tensor, library, customNodeLibraryInternalManager);
    auto elemType = ovmsPrecisionToIE2Precision(Precision::FP32);
    shape_t shape{10};
    auto tensorIE2 = std::make_shared<ov::Tensor>(elemType, shape, alloc);
    EXPECT_EQ(tensorIE2->data(), tensor.data);
}
