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
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../shape.hpp"
#include "test_utils.hpp"

using ovms::Dimension;
using ovms::Shape;

TEST(Dimension, Match) {
    EXPECT_TRUE(Dimension(1, 1).match(1));
    EXPECT_TRUE(Dimension(1, 2).match(1));
    EXPECT_TRUE(Dimension(1, 2).match(2));
    EXPECT_TRUE(Dimension::any().match(-1));
    EXPECT_TRUE(Dimension::any().match(1));
    EXPECT_TRUE(Dimension::any().match(42));
    EXPECT_TRUE(Dimension::any().match(-42));  // TODO
    EXPECT_FALSE(Dimension(10, 20).match(2));
    EXPECT_FALSE(Dimension(10, 20).match(22));
    EXPECT_FALSE(Dimension(10, 20).match(-12));
    EXPECT_FALSE(Dimension(10, 20).match(-1));
}

TEST(Dimension, PartiallyFitsInto) {
    EXPECT_TRUE(Dimension(1, 2).partiallyFitsInto(Dimension::any()));
    EXPECT_TRUE(Dimension(1, 1).partiallyFitsInto(Dimension::any()));
    EXPECT_TRUE(Dimension::any().partiallyFitsInto(Dimension::any()));
    EXPECT_TRUE(Dimension(1, 2).partiallyFitsInto(Dimension(2, 3)));
    EXPECT_TRUE(Dimension(1, 2).partiallyFitsInto(Dimension(0, 1)));
    EXPECT_TRUE(Dimension(15, 25).partiallyFitsInto(Dimension(10, 20)));
    EXPECT_TRUE(Dimension(15, 19).partiallyFitsInto(Dimension(10, 20)));
    EXPECT_TRUE(Dimension::any().partiallyFitsInto(Dimension(10, 20)));
    EXPECT_FALSE(Dimension(1, 2).partiallyFitsInto(Dimension(3, 3)));
    EXPECT_FALSE(Dimension(1, 2).partiallyFitsInto(Dimension(3, 4)));
}

TEST(Dimension, CreateIntersection) {
    EXPECT_EQ(Dimension(1, 2).createIntersection(Dimension::any()), Dimension(1, 2));
    EXPECT_EQ(Dimension(1, 1).createIntersection(Dimension::any()), Dimension(1, 1));
    EXPECT_EQ(Dimension::any().createIntersection(Dimension::any()), Dimension::any());
    EXPECT_EQ(Dimension(1, 2).createIntersection(Dimension(2, 3)), Dimension(2, 2));
    EXPECT_EQ(Dimension(1, 2).createIntersection(Dimension(0, 1)), Dimension(1, 1));
    EXPECT_EQ(Dimension(15, 25).createIntersection(Dimension(10, 20)), Dimension(15, 20));
    EXPECT_EQ(Dimension(15, 19).createIntersection(Dimension(10, 20)), Dimension(15, 19));
    EXPECT_EQ(Dimension::any().createIntersection(Dimension(10, 20)), Dimension(10, 20));
    EXPECT_EQ(Dimension(1, 2).createIntersection(Dimension(3, 3)), std::nullopt);
    EXPECT_EQ(Dimension(1, 2).createIntersection(Dimension(3, 4)), std::nullopt);
}
TEST(Shape, CreateIntersection) {
    EXPECT_EQ(Shape({1, 6, 8}).createIntersection(Shape({1, 6})), std::nullopt);
    EXPECT_EQ(Shape({1, 6, 8}).createIntersection(Shape({1, 6, 8})), Shape({1, 6, 8}));
    EXPECT_EQ(Shape({{1, 2}, {6, 12}, Dimension::any()}).createIntersection(Shape({1, 8, 100})), Shape({1, 8, 100}));
    EXPECT_EQ(Shape({{3, 5}, {7, 10}, {11, 19}}).createIntersection(Shape({Dimension(4), Dimension(8, 13), Dimension(3, 13)})),
        ovms::Shape({Dimension(4), Dimension(8, 10), Dimension(11, 13)}));  // TODO
    EXPECT_EQ(Shape({Dimension::any(), {1, 5}, {1, 19}}).createIntersection(Shape({Dimension::any(), {1, 10}, 3})), Shape({Dimension::any(), {1, 5}, 3}));
}

TEST(Shape, OvShapeMatch) {
    EXPECT_FALSE(ovms::Shape({1, 6, 8}).match(ov::Shape({1, 6})));
    EXPECT_TRUE(ovms::Shape({1, 6, 8}).match(ov::Shape({1, 6, 8})));
    EXPECT_TRUE(ovms::Shape({{1, 2}, {6, 12}, Dimension::any()}).match(ov::Shape({1, 8, 100})));
    size_t skipPosition = 1;
    EXPECT_TRUE(ovms::Shape({{3, 5}, {7, 10}, {11, 19}}).match(ov::Shape({4, 1000, 12}), skipPosition));
    skipPosition = 2;
    EXPECT_TRUE(ovms::Shape({{3, 5}, {7, 10}, {11, 19}}).match(ov::Shape({4, 8, 12000}), skipPosition));
}
