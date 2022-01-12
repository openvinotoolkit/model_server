//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../layout.hpp"

using namespace ovms;

TEST(Layout, BatchPositionValid) {
    EXPECT_EQ(Layout{"NHWC"}.getBatchIndex(), 0);
    EXPECT_EQ(Layout{"HWCN"}.getBatchIndex(), 3);
    EXPECT_EQ(Layout{"NC"}.getBatchIndex(), 0);
    EXPECT_EQ(Layout{"NCHW"}.getBatchIndex(), 0);
    EXPECT_EQ(Layout{"CHW"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N..."}.getBatchIndex(), 0);
    EXPECT_EQ(Layout{"N...CH"}.getBatchIndex(), 0);
    EXPECT_EQ(Layout{"??N..."}.getBatchIndex(), 2);
    EXPECT_EQ(Layout{"?C???N..."}.getBatchIndex(), 5);
}

TEST(Layout, BatchPositionInvalid) {
    EXPECT_EQ(Layout{"NCHWN"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N...N"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N...C...H"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N???N"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"C??H"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{""}.getBatchIndex(), std::nullopt);
}
