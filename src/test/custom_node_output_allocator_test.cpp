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

#include "../custom_node_output_allocator.hpp"

using namespace ovms;

class NodeLibraryTest : public NodeLibrary {
    static bool releaseBufferCalled;

public:
    ~NodeLibraryTest();
    static int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength);
    static int releaseBuffer(struct CustomNodeTensor* output);
    static int releaseTensors(struct CustomNodeTensor* outputs);
};

NodeLibraryTest::~NodeLibraryTest() {
    EXPECT_TRUE(releaseBufferCalled);
}

int NodeLibraryTest::execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    return 1;
}

int NodeLibraryTest::releaseBuffer(struct CustomNodeTensor* output) {
    releaseBufferCalled = true;
    return 2;
}

int NodeLibraryTest::releaseTensors(struct CustomNodeTensor* outputs) {
    return 3;
}

bool NodeLibraryTest::releaseBufferCalled = false;

class CustomNodeOutputAllocatorTest : public CustomNodeOutputAllocator {
    bool freeCalled = false;

public:
    CustomNodeOutputAllocatorTest(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary) :
        CustomNodeOutputAllocator(tensor, nodeLibrary) {
    }

    ~CustomNodeOutputAllocatorTest() {
        EXPECT_TRUE(freeCalled);
    }

    bool free(void* handle) noexcept override {
        bool tmp = CustomNodeOutputAllocator::free(handle);
        freeCalled = true;
        return tmp;
    }
};

TEST(CustomNodeOutputAllocator, RemoveBlob) {
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::C};
    int* dims = new int(1);
    dims[0] = 10;
    const InferenceEngine::TensorDesc desc{precision, {10}, layout};
    std::vector<float> data(dims[0]);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    CustomNodeTensor tensor{"name", reinterpret_cast<unsigned char*>(data.data()), static_cast<int>(dims[0]), dims, 1, static_cast<int>(precision.size())};
    NodeLibraryTest library{NodeLibraryTest::execute, NodeLibraryTest::releaseBuffer, NodeLibraryTest::releaseTensors};
    std::shared_ptr<CustomNodeOutputAllocator> customNodeOutputAllocator = std::make_shared<CustomNodeOutputAllocatorTest>(tensor, library);
    {
        InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(desc, customNodeOutputAllocator);
        blob->allocate();
    }
#pragma GCC diagnostic pop
}

TEST(CustomNodeOutputAllocator, CheckIfBlobReturnsCorrectPointer) {
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::C};
    int* dims = new int(1);
    dims[0] = 10;
    const InferenceEngine::TensorDesc desc{precision, {10}, layout};
    std::vector<float> data(dims[0]);
    CustomNodeTensor tensor{"name", reinterpret_cast<unsigned char*>(data.data()), static_cast<int>(dims[0]), dims, 1, static_cast<int>(precision.size())};
    NodeLibrary library{NodeLibraryTest::execute, NodeLibraryTest::releaseBuffer, NodeLibraryTest::releaseTensors};
    std::shared_ptr<CustomNodeOutputAllocator> customNodeOutputAllocator = std::make_shared<CustomNodeOutputAllocator>(tensor, library);
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(desc, customNodeOutputAllocator);
    blob->allocate();
    EXPECT_EQ(static_cast<unsigned char*>(blob->buffer()), tensor.data);
}
