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
#include <limits>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../sequence_manager.hpp"
#include "../status.hpp"
#include "stateful_test_utils.hpp"

TEST(SequenceManager, GetUniqueSequenceIdFirstOK) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 1;

    EXPECT_EQ(sequenceManager.mockGetUniqueSequenceId(), sequenceId);
    ovms::SequenceProcessingSpec spec1(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec1);

    EXPECT_EQ(sequenceManager.mockGetUniqueSequenceId(), sequenceId + 1);
    ovms::SequenceProcessingSpec spec2(ovms::SEQUENCE_START, sequenceId + 1);
    sequenceManager.mockCreateSequence(spec2);

    EXPECT_EQ(sequenceManager.mockGetUniqueSequenceId(), sequenceId + 2);
}

TEST(SequenceManager, GetUniqueSequenceIdFirstTaken) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 1;

    for (auto i = sequenceId; i < 5; i++) {
        ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, i);
        sequenceManager.mockCreateSequence(spec);
    }
    EXPECT_EQ(sequenceManager.mockGetUniqueSequenceId(), sequenceId + 4);
}

TEST(SequenceManager, GetUniqueSequenceIdExceedRange) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = std::numeric_limits<uint64_t>::max();
    sequenceManager.setSequenceIdCounter(sequenceId);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    EXPECT_EQ(sequenceManager.mockGetUniqueSequenceId(), 1);
}

TEST(SequenceManager, CreateSequenceOK) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    ASSERT_FALSE(sequenceManager.sequenceExists(sequenceId));
    auto status = sequenceManager.mockCreateSequence(spec);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId));
}

TEST(SequenceManager, CreateSequenceConflict) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockCreateSequence(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId));
}

TEST(SequenceManager, CreateSequenceNoIdProvided) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    ovms::SequenceProcessingSpec spec1(ovms::SEQUENCE_START, 1);
    sequenceManager.mockCreateSequence(spec1);
    ovms::SequenceProcessingSpec spec2(ovms::SEQUENCE_START, 0);
    ASSERT_TRUE(sequenceManager.sequenceExists(1));
    auto status = sequenceManager.mockCreateSequence(spec2);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(spec2.getSequenceId(), 2);
    ASSERT_TRUE(sequenceManager.sequenceExists(spec2.getSequenceId()));
}

TEST(SequenceManager, CreateSequenceNoIdProvidedExceedingRange) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = std::numeric_limits<uint64_t>::max();
    sequenceManager.setSequenceIdCounter(sequenceId);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId));
    ovms::SequenceProcessingSpec spec2(ovms::SEQUENCE_START, 0);
    auto status = sequenceManager.mockCreateSequence(spec2);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(spec2.getSequenceId(), 1);
    ASSERT_TRUE(sequenceManager.sequenceExists(spec2.getSequenceId()));
}

TEST(SequenceManager, RemoveSequenceOK) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.removeSequence(sequenceId);
    ASSERT_TRUE(status.ok());
    ASSERT_FALSE(sequenceManager.sequenceExists(sequenceId));
}

TEST(SequenceManager, RemoveSequenceNotExists) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    auto status = sequenceManager.removeSequence(42);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, HasSequenceOK) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockHasSequence(sequenceId);
    ASSERT_TRUE(status.ok());
}

TEST(SequenceManager, HasSequenceNotExist) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    auto status = sequenceManager.mockHasSequence(42);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, HasSequenceTerminated) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockTerminateSequence(sequenceId);
    ASSERT_TRUE(status.ok());

    status = sequenceManager.mockHasSequence(sequenceId);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, TerminateSequenceOK) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockTerminateSequence(sequenceId);
    ASSERT_TRUE(status.ok());
}

TEST(SequenceManager, TerminateSequenceMissing) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    auto status = sequenceManager.mockTerminateSequence(42);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, TerminateSequenceAlreadyTerminated) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockTerminateSequence(sequenceId);
    ASSERT_TRUE(status.ok());

    status = sequenceManager.mockTerminateSequence(sequenceId);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, CreateSequenceAlreadyTerminated) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(spec);
    auto status = sequenceManager.mockTerminateSequence(sequenceId);
    ASSERT_TRUE(status.ok());

    status = sequenceManager.mockCreateSequence(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_TERMINATED);
}

TEST(SequenceManager, ProcessSpecNoControlInput) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::NO_CONTROL_INPUT, sequenceId);
    auto status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);

    ovms::SequenceProcessingSpec creation_spec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(creation_spec);
    status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status.ok());

    sequenceManager.mockTerminateSequence(sequenceId);

    status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, ProcessSpecSequenceStart) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 42);
    auto status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status.ok());

    status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
}

TEST(SequenceManager, ProcessSpecSequenceEnd) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_END, sequenceId);
    auto status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);

    ovms::SequenceProcessingSpec creationSpec(ovms::SEQUENCE_START, sequenceId);
    sequenceManager.mockCreateSequence(creationSpec);
    status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status.ok());

    status = sequenceManager.processRequestedSpec(spec);
    ASSERT_TRUE(status == ovms::StatusCode::SEQUENCE_MISSING);
}

TEST(SequenceManager, RemoveOneIdleSequence) {
    ovms::model_memory_state_t_2 newState;
    DummyStatefulModel realModel;
    std::vector<float> state{10};

    ov::runtime::InferRequest auxInferRequest = realModel.createInferRequest_2();
    realModel.setVariableState_2(auxInferRequest, state);
    ov::runtime::VariableState memoryState = realModel.getVariableState_2(auxInferRequest);
    newState.push_back(memoryState);

    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId1 = 42;
    ovms::SequenceProcessingSpec spec1(ovms::SEQUENCE_START, sequenceId1);
    uint64_t sequenceId2 = 314;
    ovms::SequenceProcessingSpec spec2(ovms::SEQUENCE_START, sequenceId2);
    sequenceManager.mockCreateSequence(spec1);
    sequenceManager.mockCreateSequence(spec2);

    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId1));
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId2));

    sequenceManager.removeIdleSequences();
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId1));
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId2));

    sequenceManager.getSequence(sequenceId1).updateMemoryState_2(newState);

    sequenceManager.removeIdleSequences();
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId1));
    ASSERT_FALSE(sequenceManager.sequenceExists(sequenceId2));
}

TEST(SequenceManager, RemoveAllIdleSequences) {
    MockedSequenceManager sequenceManager(24, "dummy", 1);
    uint64_t sequenceId1 = 42;
    ovms::SequenceProcessingSpec spec1(ovms::SEQUENCE_START, sequenceId1);
    uint64_t sequenceId2 = 314;
    ovms::SequenceProcessingSpec spec2(ovms::SEQUENCE_START, sequenceId2);
    sequenceManager.mockCreateSequence(spec1);
    sequenceManager.mockCreateSequence(spec2);

    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId1));
    ASSERT_TRUE(sequenceManager.sequenceExists(sequenceId2));
    sequenceManager.getSequence(sequenceId1).setIdle();
    sequenceManager.getSequence(sequenceId2).setIdle();
    sequenceManager.removeIdleSequences();
    ASSERT_FALSE(sequenceManager.sequenceExists(sequenceId1));
    ASSERT_FALSE(sequenceManager.sequenceExists(sequenceId2));
}

TEST(SequenceManager, MultiManagersAllIdleSequences) {
    std::vector<MockedSequenceManager*> managers;
    for (int i = 0; i < 10; i++) {
        MockedSequenceManager* sequenceManager = new MockedSequenceManager(10, std::to_string(i), 1);
        ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, i + 1);
        sequenceManager->mockCreateSequence(spec);
        managers.push_back(sequenceManager);
    }

    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(managers[i]->sequenceExists(i + 1));
    }

    for (int i = 0; i < 10; i++) {
        managers[i]->getSequence(i + 1).setIdle();
        ASSERT_EQ(managers[i]->removeIdleSequences(), ovms::StatusCode::OK);
    }

    for (int i = 0; i < 10; i++) {
        ASSERT_FALSE(managers[i]->sequenceExists(i + 1));
    }

    for (int i = 0; i < 10; i++) {
        delete managers[i];
    }
}

TEST(SequenceManager, ExceedMaxSequenceNumber) {
    MockedSequenceManager sequenceManager(5, "dummy", 1);
    uint64_t sequenceId = 1;
    auto i = sequenceId;
    for (; i < 6; i++) {
        ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, i);
        ASSERT_EQ(sequenceManager.mockCreateSequence(spec), ovms::StatusCode::OK);
    }

    for (; i < 3; i++) {
        ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, i);
        ASSERT_EQ(sequenceManager.mockCreateSequence(spec), ovms::StatusCode::MAX_SEQUENCE_NUMBER_REACHED);
    }
}
