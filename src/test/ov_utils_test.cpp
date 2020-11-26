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
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::NCHW};
    const size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const size_t totalByteSize = elementsCount * precision.size();

    const InferenceEngine::TensorDesc desc{precision, shape, layout};

    std::vector<float> data(elementsCount);
    std::iota(data.begin(), data.end(), 0);

    InferenceEngine::Blob::Ptr originalBlob = InferenceEngine::make_shared_blob<float>(desc, data.data());
    InferenceEngine::Blob::Ptr copyBlob = nullptr;
    ASSERT_EQ(ovms::blobClone(copyBlob, originalBlob), ovms::StatusCode::OK);

    ASSERT_EQ(originalBlob->getTensorDesc().getDims(), shape);
    ASSERT_EQ(copyBlob->getTensorDesc().getDims(), shape);

    ASSERT_EQ(originalBlob->getTensorDesc().getLayout(), layout);
    ASSERT_EQ(copyBlob->getTensorDesc().getLayout(), layout);

    ASSERT_EQ(originalBlob->getTensorDesc().getPrecision(), precision);
    ASSERT_EQ(copyBlob->getTensorDesc().getPrecision(), precision);

    ASSERT_EQ(originalBlob->byteSize(), totalByteSize);
    ASSERT_EQ(copyBlob->byteSize(), totalByteSize);

    std::vector<float> originalBlobActualData;
    originalBlobActualData.assign((float*)originalBlob->buffer(), ((float*)originalBlob->buffer()) + elementsCount);

    std::vector<float> copyBlobActualData;
    copyBlobActualData.assign((float*)copyBlob->buffer(), ((float*)copyBlob->buffer()) + elementsCount);

    EXPECT_EQ(originalBlobActualData, data);
    EXPECT_EQ(copyBlobActualData, data);

    // Expect memory addresses to differ since cloning should allocate new memory space for the cloned blob
    EXPECT_NE((float*)copyBlob->buffer(), (float*)originalBlob->buffer());
}
