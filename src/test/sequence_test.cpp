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
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../ov_utils.hpp"
#include "../sequence.hpp"
#include "stateful_test_utils.hpp"

TEST(Sequence, SequenceDisabled) {
    uint64_t sequenceId = 3;
    ovms::Sequence sequence(sequenceId);
    ASSERT_FALSE(sequence.isTerminated());
    sequence.setTerminated();
    ASSERT_TRUE(sequence.isTerminated());
}

TEST(Sequence, UpdateLastActivityTime) {
    // last activity time update is private method and it's called inside updateMemoryState
    // so updateMemoryState method is triggered to test last activity time update
    ovms::model_memory_state_t newState;
    std::vector<size_t> shape1{1, 10};
    size_t elementsCount1 = std::accumulate(shape1.begin(), shape1.end(), 1, std::multiplies<size_t>());
    std::vector<float> state1(elementsCount1);
    std::iota(state1.begin(), state1.end(), 0);
    addState(newState, "state1", shape1, state1);

    uint64_t sequenceId = 3;
    ovms::Sequence sequence(sequenceId);
    sequence.updateMemoryState(newState);
    auto time1 = sequence.getLastActivityTime();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    sequence.updateMemoryState(newState);
    auto time2 = sequence.getLastActivityTime();
    ASSERT_TRUE(std::chrono::duration_cast<std::chrono::milliseconds>(time2.time_since_epoch()).count() >
                std::chrono::duration_cast<std::chrono::milliseconds>(time1.time_since_epoch()).count());
}

TEST(Sequence, UpdateSequenceState) {
    ovms::model_memory_state_t newState;

    std::vector<size_t> shape1{1, 10};
    size_t elementsCount1 = std::accumulate(shape1.begin(), shape1.end(), 1, std::multiplies<size_t>());
    std::vector<float> state1(elementsCount1);
    std::iota(state1.begin(), state1.end(), 0);
    addState(newState, "state1", shape1, state1);

    std::vector<size_t> shape2{1, 20};
    size_t elementsCount2 = std::accumulate(shape2.begin(), shape2.end(), 1, std::multiplies<size_t>());
    std::vector<float> state2(elementsCount2);
    std::iota(state2.begin(), state2.end(), 10);
    addState(newState, "state2", shape2, state2);

    uint64_t sequenceId = 3;
    ovms::Sequence sequence(sequenceId);
    sequence.updateMemoryState(newState);

    const ovms::sequence_memory_state_t& sequenceMemoryState = sequence.getMemoryState();
    ASSERT_TRUE(sequenceMemoryState.count("state1"));
    ASSERT_TRUE(sequenceMemoryState.count("state2"));

    std::vector<float> state1BlobSequenceData;
    state1BlobSequenceData.assign((float*)sequenceMemoryState.at("state1")->buffer(), ((float*)sequenceMemoryState.at("state1")->buffer()) + elementsCount1);
    EXPECT_EQ(state1BlobSequenceData, state1);

    std::vector<float> state2BlobSequenceData;
    state2BlobSequenceData.assign((float*)sequenceMemoryState.at("state2")->buffer(), ((float*)sequenceMemoryState.at("state2")->buffer()) + elementsCount2);
    EXPECT_EQ(state2BlobSequenceData, state2);
}
