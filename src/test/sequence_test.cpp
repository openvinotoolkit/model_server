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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../sequence.hpp"

TEST(Sequence, MovedMutexNullified) {
    ovms::Sequence sequence;
    std::unique_ptr<std::mutex>& mutexRef = sequence.getMutexRef();
    ASSERT_FALSE(mutexRef == nullptr);
    std::unique_ptr<std::mutex> mutex = sequence.moveMutex();
    // Pointer in Sequence object should be now null
    ASSERT_TRUE(mutexRef == nullptr);
    // Local ptr variable contains valid pointer to mutex now
    ASSERT_FALSE(mutex == nullptr);
}
