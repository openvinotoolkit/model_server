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
#include <algorithm>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../custom_nodes/common/buffersqueue.hpp"

using namespace ovms;
using custom_nodes_common::BuffersQueue;

TEST(CustomNodeBuffersQueue, GetAllBuffers) {
    const std::string content{"abc"};
    size_t buffersCount = 12;
    BuffersQueue buffersQueue(content.size(), buffersCount);
    std::vector<void*> buffers(buffersCount);
    for (size_t i = 0; i < buffersCount; ++i) {
        buffers[i] = buffersQueue.getBuffer();
        ASSERT_NE(nullptr, buffers[i]) << "Failed to get: " << i;
        std::memcpy(buffers[i], content.c_str(), content.size());
    }
    for (size_t i = 0; i < buffersCount; ++i) {
        EXPECT_EQ(0, std::memcmp(buffers[i], content.c_str(), content.size()))
            << "Buffer: " << i << " has different content";
    }
    std::sort(buffers.begin(), buffers.end());
    char* ptr = static_cast<char*>(buffers[0]);
    for (size_t i = 1; i < buffersCount; ++i) {
        EXPECT_GE(static_cast<char*>(buffers[i]) - ptr, content.size())
            << "distance between two buffers is to small. Between: " << i - 1 << " and " << i;
        ptr = static_cast<char*>(buffers[i]);
    }
    for (size_t i = 0; i < buffersCount; ++i) {
        EXPECT_TRUE(buffersQueue.returnBuffer(buffers[i])) << "failed to release buffer: " << i;
    }
}

TEST(CustomNodeBuffersQueue, GetAllBuffersThenNullptrForNextRequest) {
    const std::string content{"abc"};
    size_t buffersCount = 1;
    BuffersQueue buffersQueue(content.size(), buffersCount);
    std::vector<void*> buffers(buffersCount);
    for (size_t i = 0; i < buffersCount; ++i) {
        buffers[i] = buffersQueue.getBuffer();
        ASSERT_NE(nullptr, buffers[i]) << "Failed to get: " << i;
    }
    void* buffer = buffersQueue.getBuffer();
    EXPECT_EQ(nullptr, buffer) << "Failed to get buffer";
    // just to make sure that getBufferAndReturn function had
    // time to call getBuffer
    for (size_t i = 0; i < buffersCount; ++i) {
        EXPECT_TRUE(buffersQueue.returnBuffer(buffers[i])) << "failed to release buffer: " << i;
    }
}
TEST(CustomNodeBuffersQueue, ForbidReturningNonConformingAddressesSizeGreaterThan1) {
    const std::string content{"abc"};
    size_t buffersCount = 4;
    BuffersQueue buffersQueue(content.size(), buffersCount);
    std::vector<void*> buffers(buffersCount);
    // get first address
    for (size_t i = 0; i < buffersCount; ++i) {
        buffers[i] = static_cast<char*>(buffersQueue.getBuffer());
        ASSERT_NE(nullptr, buffers[i]) << "Failed to get: " << i;
    }
    char* start = static_cast<char*>(*(std::min_element(buffers.begin(), buffers.end())));
    char* end = static_cast<char*>(*(std::max_element(buffers.begin(), buffers.end())));
    auto misalignedOffset = content.size() - 1;
    ASSERT_NE(0, misalignedOffset);
    EXPECT_FALSE(buffersQueue.returnBuffer(start - content.size()));
    EXPECT_FALSE(buffersQueue.returnBuffer(start - 1));
    EXPECT_FALSE(buffersQueue.returnBuffer(start + misalignedOffset));
    EXPECT_FALSE(buffersQueue.returnBuffer(end + content.size()));
    EXPECT_FALSE(buffersQueue.returnBuffer(end + 1));
    EXPECT_FALSE(buffersQueue.returnBuffer(end - misalignedOffset));
}
TEST(CustomNodeBuffersQueue, ForbidReturningNonConformingAddressesSizeEqual1) {
    const std::string content{"a"};
    size_t buffersCount = 4;
    BuffersQueue buffersQueue(content.size(), buffersCount);
    std::vector<void*> buffers(buffersCount);
    // get first address
    for (size_t i = 0; i < buffersCount; ++i) {
        buffers[i] = static_cast<char*>(buffersQueue.getBuffer());
        ASSERT_NE(nullptr, buffers[i]) << "Failed to get: " << i;
    }
    char* start = static_cast<char*>(*(std::min_element(buffers.begin(), buffers.end())));
    char* end = static_cast<char*>(*(std::max_element(buffers.begin(), buffers.end())));
    EXPECT_FALSE(buffersQueue.returnBuffer(start - content.size()));
    EXPECT_FALSE(buffersQueue.returnBuffer(start - 1));
    EXPECT_FALSE(buffersQueue.returnBuffer(end + content.size()));
    EXPECT_FALSE(buffersQueue.returnBuffer(end + 1));
}
TEST(CustomNodeBuffersQueue, GetAndReturnBuffersSeveralTimes) {
    const std::vector<std::string> contents{{"abc"}, {"dce"}};
    size_t buffersCount = 42;
    int iterations = 121;
    BuffersQueue buffersQueue(contents[0].size(), buffersCount);
    for (auto j = 0; j < iterations; ++j) {
        const std::string& content = contents[j % contents.size()];
        std::vector<void*> buffers(buffersCount);
        for (size_t i = 0; i < buffersCount; ++i) {
            buffers[i] = buffersQueue.getBuffer();
            ASSERT_NE(nullptr, buffers[i]) << "Failed to get: " << i << " iteration:" << j;
            std::memcpy(buffers[i], content.c_str(), content.size());
        }
        for (size_t i = 0; i < buffersCount; ++i) {
            EXPECT_EQ(0, std::memcmp(buffers[i], content.c_str(), content.size()))
                << "Buffer: " << i << " has different content iteration:" << j;
        }
        for (size_t i = 0; i < buffersCount; ++i) {
            EXPECT_TRUE(buffersQueue.returnBuffer(buffers[i])) << "failed to release buffer: " << i << " iteration:" << j;
        }
    }
}
