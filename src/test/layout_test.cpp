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
#include "../status.hpp"

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
    EXPECT_EQ(Layout{"..."}.getBatchIndex(), std::nullopt);
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
    EXPECT_EQ(Layout{"N...C...H"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"N???N"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{"C??H"}.getBatchIndex(), std::nullopt);
    EXPECT_EQ(Layout{""}.getBatchIndex(), std::nullopt);
}

TEST(Layout, Validate) {
    EXPECT_EQ(Layout{"..."}.validate(), StatusCode::OK);   // unspecified layout used in DAG
    EXPECT_EQ(Layout{"N..."}.validate(), StatusCode::OK);  // default model layout
    EXPECT_EQ(Layout{"N.CH"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"..NHW."}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"N...N"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
    EXPECT_EQ(Layout{"N...C...H"}.validate(), StatusCode::LAYOUT_WRONG_FORMAT);
}

TEST(Layout, CreateIntersectionPositive) {
    EXPECT_EQ(Layout("NCHW").createIntersection(Layout("NCHW"), 4), Layout("NCHW"));
    EXPECT_EQ(Layout("NCHW").createIntersection(Layout("N..."), 4), Layout("NCHW"));
    EXPECT_EQ(Layout("N...").createIntersection(Layout("NCHW"), 4), Layout("NCHW"));
    EXPECT_EQ(Layout("NCHWD").createIntersection(Layout("NCHW?"), 5), Layout("NCHWD"));
    EXPECT_EQ(Layout("NCHW?").createIntersection(Layout("NCHWD"), 5), Layout("NCHWD"));  // test symmetry
    EXPECT_EQ(Layout("NC??").createIntersection(Layout("??DH"), 4), Layout("NCDH"));
    EXPECT_EQ(Layout("NC...").createIntersection(Layout("??DH"), 4), Layout("NCDH"));
    EXPECT_EQ(Layout("N...").createIntersection(Layout("...D"), 4), Layout("N??D"));
    EXPECT_EQ(Layout("N...").createIntersection(Layout("??D"), 3), Layout("N?D"));
    EXPECT_EQ(Layout("N?H...W?C").createIntersection(Layout("...HWD?"), 6), Layout("N?HWDC"));
    EXPECT_EQ(Layout("...N").createIntersection(Layout("...??N"), 5), Layout("????N"));
}

TEST(Layout, CreateIntersectionNegative) {
    EXPECT_EQ(Layout("NCHWD").createIntersection(Layout("NCHW"), 5), std::nullopt);
    EXPECT_EQ(Layout("NCHW").createIntersection(Layout("NCHWD"), 5), std::nullopt);
    EXPECT_EQ(Layout("N...C").createIntersection(Layout("N...W"), 4), std::nullopt);
    EXPECT_EQ(Layout("NC??...").createIntersection(Layout("...C"), 5), std::nullopt);
    EXPECT_EQ(Layout("...N").createIntersection(Layout("N..."), 2), std::nullopt);
    EXPECT_EQ(Layout("...N").createIntersection(Layout("N..."), 10), std::nullopt);
    EXPECT_EQ(Layout("...N").createIntersection(Layout("...N?"), 4), std::nullopt);
}

TEST(Layout, ConversionBetweenOvLayout) {
    for (const auto& layoutStr : std::vector<std::string>{
             {"NHWC"},
             {"HWCN"},
             {"NC"},
             {"NCHW"},
             {"CHW"},
             {"N..."},
             {"N...CH"},
             {"??N..."},
             {"?C???N..."},
             {"...NC"},
             {"..."},
         }) {
        EXPECT_EQ(
            ovms::Layout::fromOvLayout(
                ov::Layout(
                    ovms::Layout{layoutStr})),
            ovms::Layout{layoutStr})
            << "error converting layout: " << layoutStr;
    }
}

TEST(Layout, IsCompatibleWithShape) {
    EXPECT_TRUE(Layout("NCHW").isCompatible(Shape{10, 3, 224, 224}));
    EXPECT_TRUE(Layout("NCHW...").isCompatible(Shape{1, 3, 224, 224}));
    EXPECT_TRUE(Layout("N?HW...").isCompatible(Shape{1, 3, 224, 224}));
    EXPECT_TRUE(Layout("N...").isCompatible(Shape{1}));
    EXPECT_TRUE(Layout("N...").isCompatible(Shape{1, 5, 9, 100}));
    EXPECT_TRUE(Layout("...").isCompatible(Shape{1, 5, 9, 100}));
    EXPECT_TRUE(Layout("NC...H").isCompatible(Shape{1, 5, 9}));
    EXPECT_TRUE(Layout("NC...H").isCompatible(Shape{1, 5, 9, 100}));
}

TEST(Layout, IsIncompatibleWithShape) {
    EXPECT_FALSE(Layout("NCHW").isCompatible(Shape{1, 3, 224, 224, 10}));  // to many dims in shape
    EXPECT_FALSE(Layout("NCHW").isCompatible(Shape{1, 224, 224}));         // too few dims in shape
    EXPECT_FALSE(Layout("N...H").isCompatible(Shape{1}));                  // too few dims in shape
    EXPECT_FALSE(Layout("N?HW").isCompatible(Shape{1, 224, 224}));         // too few dims in shape
    EXPECT_FALSE(Layout("N?HW").isCompatible(Shape{1, 224, 224, 3, 1}));   // too many dims in shape
}
