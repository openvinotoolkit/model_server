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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>

#include "../ov_utils.hpp"
#include "../sequence.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using namespace InferenceEngine;

class MockIVariableState : public IVariableState {
public:
    MOCK_METHOD(StatusCode, GetName, (char* name, size_t len, ResponseDesc* resp), (const, noexcept, override));
    MOCK_METHOD(StatusCode, Reset, (ResponseDesc * resp), (noexcept, override));
    MOCK_METHOD(StatusCode, SetState, (Blob::Ptr newState, ResponseDesc* resp), (noexcept, override));
    MOCK_METHOD(StatusCode, GetState, (Blob::CPtr & state, ResponseDesc* resp), (const, noexcept, override));
};

class MockIVariableStateWithData : public MockIVariableState {
public:
    std::string stateName;
    Blob::Ptr stateBlob;

    MockIVariableStateWithData(std::string name, Blob::Ptr blob) {
        stateName = name;
        stateBlob = blob;
    }

    StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept override {
        snprintf(name, sizeof(stateName), stateName.c_str());
        return StatusCode::OK;
    }

    StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept override {
        state = stateBlob;
        return StatusCode::OK;
    }
};

void addState(ovms::model_memory_state_t& states, std::string name, std::vector<size_t>& shape, std::vector<float>& values) {
    const Precision precision{Precision::FP32};
    const Layout layout{Layout::NC};
    const TensorDesc desc{precision, shape, layout};

    Blob::Ptr stateBlob = make_shared_blob<float>(desc, values.data());
    std::shared_ptr<IVariableState> ivarPtr = std::make_shared<MockIVariableStateWithData>(name, stateBlob);
    states.push_back(VariableState(ivarPtr));
}

TEST(Sequence, MovedMutexNullified) {
    ovms::Sequence sequence;
    const std::unique_ptr<std::mutex>& mutexRef = sequence.getMutexRef();
    ASSERT_FALSE(mutexRef == nullptr);
    std::unique_ptr<std::mutex> mutex = sequence.moveMutex();
    // Pointer in Sequence object should be now null
    ASSERT_TRUE(mutexRef == nullptr);
    // Local ptr variable contains valid pointer to mutex now
    ASSERT_FALSE(mutex == nullptr);
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

    ovms::Sequence sequence;
    auto time1 = sequence.getLastActivityTime();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    sequence.updateMemoryState(newState);
    auto time2 = sequence.getLastActivityTime();
    EXPECT_NE(time1, time2);
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

    ovms::Sequence sequence;
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
