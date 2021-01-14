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

#include "../sequence_manager.hpp"
#include "../status.hpp"

using namespace ovms;

TEST(SequenceManager, AddSequenceOK) {
    SequenceManager sequenceManager(120, 24);
    ASSERT_FALSE(sequenceManager.hasSequence(42));
    auto status = sequenceManager.addSequence(42);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(sequenceManager.hasSequence(42));
}

TEST(SequenceManager, AddSequenceConflict) {
    SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    auto status = sequenceManager.addSequence(42);
    ASSERT_TRUE(status == StatusCode::SEQUENCE_ALREADY_EXISTS);
    ASSERT_TRUE(sequenceManager.hasSequence(42));
}

TEST(SequenceManager, RemoveSequenceOK) {
    SequenceManager sequenceManager(120, 24);
    sequenceManager.addSequence(42);
    auto status = sequenceManager.removeSequence(42);
    ASSERT_TRUE(status.ok());
    ASSERT_FALSE(sequenceManager.hasSequence(42));
}

TEST(SequenceManager, RemoveSequenceNotExists) {
    SequenceManager sequenceManager(120, 24);
    auto status = sequenceManager.removeSequence(42);
    ASSERT_TRUE(status == StatusCode::SEQUENCE_MISSING);
}