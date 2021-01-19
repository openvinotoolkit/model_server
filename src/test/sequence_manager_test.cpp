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
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../sequence_manager.hpp"
#include "../status.hpp"
#include "stateful_test_utils.hpp"

TEST(SequenceManager, AddSequenceOK) {
    ovms::SequenceManager sequenceManager(120, 24);
    ASSERT_FALSE(sequenceManager.sequenceExists(42));
    auto status = sequenceManager.addSequence(42);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceManager.sequenceExists(42));
}

TEST(SequenceManager, AddSequenceConflict) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    auto status = sequenceManager.addSequence(42);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
    ASSERT_TRUE(sequenceManager.sequenceExists(42));
}

TEST(SequenceManager, RemoveSequenceOK) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    auto status = sequenceManager.removeSequence(42);
    ASSERT_TRUE(status.ok());
    ASSERT_FALSE(sequenceManager.sequenceExists(42));
}

TEST(SequenceManager, RemoveSequenceNotExists) {
    ovms::SequenceManager sequenceManager(120, 24);
    auto status = sequenceManager.removeSequence(42);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, HasSequenceOK) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.hasSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);
}

TEST(SequenceManager, HasSequenceNotExist) {
    ovms::SequenceManager sequenceManager(120, 24);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.hasSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, HasSequenceTerminated) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.terminateSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);

    sequenceMutexPtr = nullptr;
    status = sequenceManager.hasSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_TERMINATED);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, CreateSequenceOK) {
    ovms::SequenceManager sequenceManager(120, 24);
    ASSERT_FALSE(sequenceManager.sequenceExists(42));
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.createSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);
    ASSERT_TRUE(sequenceManager.sequenceExists(42));
}

TEST(SequenceManager, CreateSequenceConflict) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.createSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, TerminateSequenceOK) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.terminateSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);
}

TEST(SequenceManager, TerminateSequenceMissing) {
    ovms::SequenceManager sequenceManager(120, 24);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.terminateSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, TerminateSequenceAlreadyTerminated) {
    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    auto status = sequenceManager.terminateSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);

    sequenceMutexPtr = nullptr;
    status = sequenceManager.terminateSequence(42, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_TERMINATED);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, GetSequencePtrNoControlInput) {
    ovms::SequenceManager sequenceManager(120, 24);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    ovms::SequenceProcessingSpec spec(ovms::NO_CONTROL_INPUT, 42);
    auto status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);

    sequenceManager.addSequence(42);
    status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);

    sequenceManager.terminateSequence(42, sequenceMutexPtr);

    sequenceMutexPtr = nullptr;
    status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_TERMINATED);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, GetSequencePtrSequenceStart) {
    ovms::SequenceManager sequenceManager(120, 24);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 42);
    auto status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);

    sequenceMutexPtr = nullptr;
    status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, GetSequencePtrSequenceEnd) {
    ovms::SequenceManager sequenceManager(120, 24);
    ovms::MutexPtr sequenceMutexPtr = nullptr;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_END, 42);
    auto status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);

    sequenceManager.addSequence(42);
    status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceMutexPtr != nullptr);

    sequenceMutexPtr = nullptr;
    status = sequenceManager.getSequenceMutexPtr(spec, sequenceMutexPtr);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_TERMINATED);
    ASSERT_TRUE(sequenceMutexPtr == nullptr);
}

TEST(SequenceManager, UpdateSequenceState) {
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

    ovms::SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    sequenceManager.updateSequenceMemoryState(42, newState);

    const ovms::sequence_memory_state_t& sequenceMemoryState = sequenceManager.getSequenceMemoryState(42);
    ASSERT_TRUE(sequenceMemoryState.count("state1"));
    ASSERT_TRUE(sequenceMemoryState.count("state2"));

    std::vector<float> state1BlobSequenceData;
    state1BlobSequenceData.assign((float*)sequenceMemoryState.at("state1")->buffer(), ((float*)sequenceMemoryState.at("state1")->buffer()) + elementsCount1);
    EXPECT_EQ(state1BlobSequenceData, state1);

    std::vector<float> state2BlobSequenceData;
    state2BlobSequenceData.assign((float*)sequenceMemoryState.at("state2")->buffer(), ((float*)sequenceMemoryState.at("state2")->buffer()) + elementsCount2);
    EXPECT_EQ(state2BlobSequenceData, state2);
}

TEST(SequenceManager, RemoveTimedOutSequences) {
    ovms::model_memory_state_t newState;
    std::vector<size_t> shape1{1, 10};
    size_t elementsCount1 = std::accumulate(shape1.begin(), shape1.end(), 1, std::multiplies<size_t>());
    std::vector<float> state1(elementsCount1);
    std::iota(state1.begin(), state1.end(), 0);
    addState(newState, "state1", shape1, state1);

    ovms::SequenceManager sequenceManager(5, 24);
    sequenceManager.addSequence(42);
    sequenceManager.addSequence(314);

    ASSERT_TRUE(sequenceManager.sequenceExists(42));
    ASSERT_TRUE(sequenceManager.sequenceExists(314));
    std::this_thread::sleep_for(std::chrono::seconds(3));

    sequenceManager.removeTimedOutSequences(std::chrono::steady_clock::now());
    ASSERT_TRUE(sequenceManager.sequenceExists(42));
    ASSERT_TRUE(sequenceManager.sequenceExists(314));

    sequenceManager.updateSequenceMemoryState(42, newState);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    sequenceManager.removeTimedOutSequences(std::chrono::steady_clock::now());
    ASSERT_TRUE(sequenceManager.sequenceExists(42));
    ASSERT_FALSE(sequenceManager.sequenceExists(314));
}
