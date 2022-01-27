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
    EXPECT_EQ(Layout{"...NC"}.getBatchIndex(), std::nullopt);
}

TEST(Layout, BatchPositionInvalid) {
    EXPECT_EQ(Layout{"NCHWN"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N.C.H.W"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N..H.W"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N.H..W"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"NH.W.."}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"NH."}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N.H.W"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"NHW.."}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N..CH"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N.CH"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"..NHW."}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N...N"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N...C...H"}.getBatchIndex(), std::nullopt);  // TODO validate as incorrect
    EXPECT_EQ(Layout{"N???N"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"C??H"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{""}.getBatchIndex(), std::nullopt);
}
TEST(Layout, Validate) {
    EXPECT_EQ(Layout{"N.CH"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"..NHW."}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"N...N"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"N...C...H"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
}
TEST(Layout, CreateIntersection) {
    // handling variety of possible layoyts can be cumbersome
    // we should at least assume that there is only one ETC as then it should be feasible to validate such layouts. For now we assume that we just return N...
    auto intersect = Layout("NCHW").createIntersection(Layout("NCHW")).value();
    EXPECT_EQ(intersect, Layout("NCHW")) << intersect;
    EXPECT_EQ(Layout("NCHWD").createIntersection(
                  Layout("NCHW")),
        std::nullopt);
    EXPECT_EQ(Layout("NCHW").createIntersection(
                  Layout("NCHWD")),
        std::nullopt);
    EXPECT_EQ(Layout("NCHW").createIntersection(
                  Layout("N...")),
        Layout("NCHW"));
    EXPECT_EQ(Layout("N...").createIntersection(
                  Layout("NCHW")),
        Layout("NCHW"));
}
TEST(Layout, DISABLED_CreateIntersection2) {
    // handling variety of possible layoyts can be cumbersome
    // we should at least assume that there is only one ETC as then it should be feasible to validate such layouts. For now we assume that we just return N...
    EXPECT_EQ(Layout("NCHWD").createIntersection(
                  Layout("NCHW?")),
        Layout("NCHWD"));
    EXPECT_EQ(Layout("NCHW?").createIntersection(  // test symmetry
                  Layout("NCHWD")),
        Layout("NCHWD"));
    EXPECT_EQ(Layout("NCHWD").createIntersection(
                  Layout("NCHW")),
        std::nullopt);
    EXPECT_EQ(Layout("NCHW").createIntersection(
                  Layout("NCHWD")),
        std::nullopt);
    EXPECT_EQ(Layout("NC??").createIntersection(
                                Layout("??DH"))
                  .value(),
        Layout("NCDH"));
    EXPECT_EQ(Layout("NC...").createIntersection(
                                 Layout("??DH"))
                  .value(),
        Layout("NCDH"));
    EXPECT_EQ(Layout("N...").createIntersection(
                                Layout("...D"))
                  .value(),
        Layout("N...D"));
    EXPECT_EQ(Layout("N...").createIntersection(
                                Layout("??D"))
                  .value(),
        Layout("N?D"));
}
