//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "embeddings_servable.hpp"

#include <vector>

#include "../logging.hpp"

#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov::genai;
using namespace ov;

namespace ovms {

// From OpenVINO GenAI repository
static std::shared_ptr<op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
    auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

    auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
}

// From OpenVINO GenAI repository
static std::shared_ptr<op::Op> get_mean_pooling_op(std::shared_ptr<Model> model,
    const ov::Output<ov::Node>& last_hidden_state_node) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

    auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];

    auto unsqueze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto unsqueze = std::make_shared<op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

    auto input_mask_expanded = std::make_shared<op::v3::Broadcast>(unsqueze, shape_of);

    auto input_mask_expanded_convert =
        std::make_shared<op::v0::Convert>(input_mask_expanded, last_hidden_state_node.get_element_type());

    auto last_hidden_node_with_applied_attention_mask =
        std::make_shared<op::v1::Multiply>(last_hidden_state_node, input_mask_expanded_convert->outputs()[0]);

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto sum_hidden_state = std::make_shared<op::v1::ReduceSum>(last_hidden_node_with_applied_attention_mask, axis_1);

    // f32 overflow possible
    // ReduceMean might help with overflow but its precision diverges from LlamaIndex
    auto sum_expanded_mask = std::make_shared<op::v1::ReduceSum>(input_mask_expanded_convert, axis_1);

    auto nearest_to_zero =
        std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1e-12});
    auto max_expanded_mask = std::make_shared<op::v1::Maximum>(sum_expanded_mask, nearest_to_zero);

    // shape: [batch_size, hidden_state_size]
    return std::make_shared<op::v1::Divide>(sum_hidden_state, max_expanded_mask);
}

// From OpenVINO GenAI repository
static std::shared_ptr<op::Op> get_last_token_pooling_op(std::shared_ptr<Model> model,
    const ov::Output<ov::Node>& last_hidden_state_node) {
    auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(attention_mask, axis_1);
    auto subtract_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto subtract = std::make_shared<op::v1::Subtract>(reduce_sum, subtract_1);

    return std::make_shared<op::v8::Gather>(last_hidden_state_node, subtract, axis_1, 1);
}

std::shared_ptr<ov::Model> EmbeddingsServable::applyPrePostProcessing(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor processor(model);

    // Find the output with 3 dimensions (batch_size, sequence_length, hidden_size)
    this->targetOutputIndex = -1;
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        if (model->outputs()[i].get_partial_shape().rank() == 3) {
            this->targetOutputIndex = static_cast<int>(i);
            break;
        }
    }

    if (this->targetOutputIndex == -1) {
        OPENVINO_THROW("No output with 3 dimensions found");
    }

    processor.output(this->targetOutputIndex).postprocess().custom([this, model](const ov::Output<ov::Node>& node) {
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Applying {} pooling to embeddings output",
            mediapipe::EmbeddingsCalculatorOVOptions_Pooling_Name(this->pooling));

        switch (this->pooling) {
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_CLS: {
            return get_cls_pooling_op(node);
        }
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_LAST: {
            return get_last_token_pooling_op(model, node);
        }
        case mediapipe::EmbeddingsCalculatorOVOptions_Pooling_MEAN: {
            return get_mean_pooling_op(model, node);
        }
        }
        OPENVINO_THROW("Pooling type is not supported");
    });

    if (this->normalizeEmbeddings) {
        SPDLOG_LOGGER_DEBUG(embeddings_calculator_logger, "Applying L2 normalization to embeddings output");
        processor.output(this->targetOutputIndex).postprocess().custom([](const ov::Output<ov::Node>& node) {
            auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
            return std::make_shared<op::v0::NormalizeL2>(node, axis_const, 1e-12, op::EpsMode::MAX);
        });
    }

    model = processor.build();
    return model;
}

}  // namespace ovms
