//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>

namespace ovms {
// Internal graph builder implemented in embeddings_servable.cpp.
std::shared_ptr<ov::op::Op> get_mean_pooling_op(
    const ov::Output<ov::Node>& last_hidden_state_node,
    const ov::Output<ov::Node>& attention_mask);
}  // namespace ovms

namespace {

std::shared_ptr<ov::Model> makeMeanPoolingModel(const ov::PartialShape& shape) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
        ov::PartialShape{shape[0], shape[1]});
    return std::make_shared<ov::Model>(ov::OutputVector{ovms::get_mean_pooling_op(hidden, mask)},
        ov::ParameterVector{hidden, mask});
}

// Model an FP16 boundary after each reduction independently of the CPU plugin's
// mixed-precision decisions. The former sum exceeds FP16's finite range here.
void roundReductionOutputsToFp16(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::ReduceSum>(node) && !ov::is_type<ov::op::v1::ReduceMean>(node)) {
            continue;
        }
        const auto consumers = node->output(0).get_target_inputs();
        auto narrowed = std::make_shared<ov::op::v0::Convert>(node, ov::element::f16);
        auto widened = std::make_shared<ov::op::v0::Convert>(narrowed, ov::element::f32);
        for (const auto& consumer : consumers) {
            consumer.replace_source_output(widened);
        }
    }
    model->validate_nodes_and_infer_types();
}

class EmbeddingsMeanPoolingTest : public ::testing::TestWithParam<size_t> {};

TEST_P(EmbeddingsMeanPoolingTest, HandlesPaddingAndFullyMaskedRowsWithFp16Intermediates) {
    const size_t sequenceLength = GetParam();
    constexpr size_t batchSize = 4;
    constexpr size_t hiddenSize = 4;
    auto model = makeMeanPoolingModel(ov::PartialShape{batchSize, -1, hiddenSize});
    roundReductionOutputsToFp16(model);
    ov::Tensor hidden(ov::element::f32, ov::Shape{batchSize, sequenceLength, hiddenSize});
    ov::Tensor mask(ov::element::i64, ov::Shape{batchSize, sequenceLength});
    const std::vector<float> values{80.0f, -96.0f, 3.0f, 0.125f};
    // No padding, right padding, left padding, and a fully masked row.
    const size_t validLength = std::max<size_t>(1, sequenceLength / 2);
    for (size_t batch = 0; batch < batchSize; ++batch) {
        for (size_t token = 0; token < sequenceLength; ++token) {
            const bool valid = batch == 0 || (batch == 1 && token < validLength) ||
                               (batch == 2 && token >= sequenceLength - validLength);
            mask.data<int64_t>()[batch * sequenceLength + token] = valid;
            for (size_t dim = 0; dim < hiddenSize; ++dim) {
                hidden.data<float>()[(batch * sequenceLength + token) * hiddenSize + dim] =
                    valid ? values[dim] : 3000.0f;
            }
        }
    }
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{batchSize, hiddenSize})};
    ASSERT_TRUE(model->evaluate(outputs, ov::TensorVector{hidden, mask}));
    for (size_t batch = 0; batch < batchSize; ++batch) {
        for (size_t dim = 0; dim < hiddenSize; ++dim) {
            const float actual = outputs[0].data<float>()[batch * hiddenSize + dim];
            EXPECT_TRUE(std::isfinite(actual));
            EXPECT_NEAR(actual, batch == 3 ? 0.0f : values[dim], 0.1f);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SequenceLengths, EmbeddingsMeanPoolingTest,
    ::testing::Values(1, 7, 128, 8192));

TEST(EmbeddingsMeanPooling, MatchesDoublePrecisionMaskedReference) {
    constexpr size_t batchSize = 3;
    constexpr size_t sequenceLength = 257;
    constexpr size_t hiddenSize = 7;
    auto model = makeMeanPoolingModel(ov::PartialShape{batchSize, sequenceLength, hiddenSize});
    ov::Tensor hidden(ov::element::f32, ov::Shape{batchSize, sequenceLength, hiddenSize});
    ov::Tensor mask(ov::element::i64, ov::Shape{batchSize, sequenceLength});
    for (size_t i = 0; i < hidden.get_size(); ++i) {
        hidden.data<float>()[i] = static_cast<float>(std::sin(static_cast<double>(i)) * 100.0);
    }
    for (size_t i = 0; i < mask.get_size(); ++i) {
        mask.data<int64_t>()[i] = (i % 5 != 0);
    }
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{batchSize, hiddenSize})};
    ASSERT_TRUE(model->evaluate(outputs, ov::TensorVector{hidden, mask}));
    for (size_t batch = 0; batch < batchSize; ++batch) {
        for (size_t dim = 0; dim < hiddenSize; ++dim) {
            double sum = 0.0;
            size_t count = 0;
            for (size_t token = 0; token < sequenceLength; ++token) {
                if (mask.data<int64_t>()[batch * sequenceLength + token]) {
                    sum += hidden.data<float>()[(batch * sequenceLength + token) * hiddenSize + dim];
                    ++count;
                }
            }
            EXPECT_NEAR(outputs[0].data<float>()[batch * hiddenSize + dim], sum / count, 1e-5);
        }
    }
}

}  // namespace
