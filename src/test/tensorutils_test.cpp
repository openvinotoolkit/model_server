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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../tensor_utils.hpp"

using namespace ovms;

TEST(TensorWithSource, CanCreateWithoutSource) {
    auto actual = ov::Tensor(ov::element::f32, {2});
    TensorWithSource tensor(actual);
    EXPECT_FALSE(tensor.hasSource());
    EXPECT_EQ(tensor.getActualTensor().data(), actual.data());
}

TEST(TensorWithSource, CanCreateWithSource) {
    auto actual = ov::Tensor(ov::element::f32, {2});
    auto source = ov::Tensor(ov::element::f32, {2});
    TensorWithSource tensor(actual, source);
    EXPECT_TRUE(tensor.hasSource());
    EXPECT_EQ(tensor.getActualTensor().data(), actual.data());
    EXPECT_EQ(tensor.getSourceTensor().data(), source.data());
}
