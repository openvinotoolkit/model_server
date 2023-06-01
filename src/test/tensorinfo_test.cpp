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

#include "../tensorinfo.hpp"

using namespace ovms;

TEST(TensorInfo, Intersection) {
    //  match
    auto first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"});
    auto second = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, Dimension::any(), {220, 225}, {200, 300}}), Layout{"NCHW"});
    auto intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    EXPECT_TRUE(intersect->isTensorSpecEqual(TensorInfo("a", "b", Precision::FP32, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"}))) << intersect->asString();
    // precision UNDEFINED
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"});
    second = std::make_shared<const TensorInfo>("a", "b", Precision::UNDEFINED, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"});
    intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    EXPECT_TRUE(intersect->isTensorSpecEqual(TensorInfo("a", "b", Precision::FP32, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"}))) << intersect->asString();
    // Unspecified intersection should succeed with any & itself
    first = TensorInfo::getUnspecifiedTensorInfo();
    second = TensorInfo::getUnspecifiedTensorInfo();
    intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    EXPECT_TRUE(intersect->isTensorSpecEqual(*first)) << intersect->asString();
    // uspecified should succeed with any
    first = TensorInfo::getUnspecifiedTensorInfo();
    second = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, Dimension::any(), {220, 225}, {200, 300}}), Layout{"NCHW"});
    intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    EXPECT_TRUE(intersect->isTensorSpecEqual(*second)) << intersect->asString();
    // default layout should match any
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, Dimension::any(), {220, 225}, {200, 300}}), Layout::getDefaultLayout());
    second = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, Dimension::any(), {220, 225}, {200, 300}}), Layout{"NCHW"});
    intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    EXPECT_TRUE(intersect->isTensorSpecEqual(*second)) << intersect->asString();
    // precision mismatch
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    second = std::make_shared<const TensorInfo>("a", "b", Precision::I32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    EXPECT_EQ(first->createIntersection(*second), nullptr);
    // layout order mismatch
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    second = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NHWC"});
    EXPECT_EQ(first->createIntersection(*second), nullptr);
    // name mismatch
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    second = std::make_shared<const TensorInfo>("a2", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    EXPECT_EQ(first->createIntersection(*second), nullptr);
    // mapped name mismatch
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    second = std::make_shared<const TensorInfo>("a", "b2", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCHW"});
    EXPECT_EQ(first->createIntersection(*second), nullptr);
    // intersect with demultiplexer
    first = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"N...H?"});
    second = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, 224}), Layout{"NCH..."});
    intersect = first->createIntersection(*second);
    ASSERT_NE(intersect, nullptr);
    intersect = intersect->createCopyWithDemultiplexerDimensionPrefix(Dimension::any());
    EXPECT_EQ(intersect->getLayout(), "N?CH?");
}

TEST(TensorInfo, LayoutWithAppliedDemultiplexer) {
    auto info = std::make_shared<const TensorInfo>("a", "b", Precision::FP32, Shape({1, 3, 224, {220, 230}}), Layout{"NCHW"});
    info = info->createCopyWithDemultiplexerDimensionPrefix(100);
    EXPECT_TRUE(info->isInfluencedByDemultiplexer());
    EXPECT_EQ(info->getShape(), (Shape{100, 1, 3, 224, Dimension{220, 230}})) << info->getShape().toString();
    EXPECT_EQ(info->getLayout(), Layout("N?CHW")) << info->getLayout();
}
