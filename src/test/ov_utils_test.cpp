//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../ov_utils.hpp"

using testing::ElementsAre;

TEST(OVUtils, CopyBlob) {
    const std::vector<size_t> shape{2, 3, 4, 5};
    const auto elementType = ov::element::Type(ov::element::Type_t::f32);
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * elementType.size();

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    ov::runtime::Tensor originalTensor(elementType, shape, data.data());
    std::shared_ptr<ov::runtime::Tensor> copyTensor = nullptr;

    ASSERT_EQ(ovms::tensorClone(copyTensor, originalTensor), ovms::StatusCode::OK);

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor->get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor->get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), totalByteSize);
    ASSERT_EQ(copyTensor->get_byte_size(), totalByteSize);

    ASSERT_EQ(copyTensor->get_strides(), originalTensor.get_strides());

    std::vector<float> originalTensorActualData;
    originalTensorActualData.assign(static_cast<float*>(originalTensor.data()), static_cast<float*>(originalTensor.data()) + elementsCount);

    std::vector<float> copyTensorActualData;
    copyTensorActualData.assign(static_cast<float*>(copyTensor->data()), static_cast<float*>(copyTensor->data()) + elementsCount);

    EXPECT_EQ(originalTensorActualData, data);
    EXPECT_EQ(copyTensorActualData, data);

    // Expect memory addresses to differ since cloning should allocate new memory space for the cloned blob
    EXPECT_NE(originalTensor.data(), copyTensor->data());
}

TEST(OVUtils, ConstCopyBlob) {
    const std::vector<size_t> shape{2, 3, 4, 5};
    const auto elementType = ov::element::Type(ov::element::Type_t::f32);
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * elementType.size();

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    ov::runtime::Tensor originalTensor(elementType, shape, data.data());
    std::shared_ptr<ov::runtime::Tensor> copyTensor = nullptr;

    ASSERT_EQ(ovms::tensorClone(copyTensor, originalTensor), ovms::StatusCode::OK);

    ASSERT_EQ(originalTensor.get_shape(), shape);
    ASSERT_EQ(copyTensor->get_shape(), shape);

    ASSERT_EQ(originalTensor.get_element_type(), elementType);
    ASSERT_EQ(copyTensor->get_element_type(), elementType);

    ASSERT_EQ(originalTensor.get_byte_size(), totalByteSize);
    ASSERT_EQ(copyTensor->get_byte_size(), totalByteSize);

    ASSERT_EQ(copyTensor->get_strides(), originalTensor.get_strides());

    std::vector<float> originalTensorActualData;
    const void* start = (const void*)(originalTensor.data());
    originalTensorActualData.assign((float*)start, (float*)start + elementsCount);

    std::vector<float> copyTensorActualData;
    copyTensorActualData.assign(static_cast<float*>(copyTensor->data()), static_cast<float*>(copyTensor->data()) + elementsCount);

    EXPECT_EQ(originalTensorActualData, data);
    EXPECT_EQ(copyTensorActualData, data);

    // Expect memory addresses to differ since cloning should allocate new memory space for the cloned blob
    EXPECT_NE(originalTensor.data(), copyTensor->data());
}
