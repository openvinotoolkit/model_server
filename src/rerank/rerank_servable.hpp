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
#pragma once

#include "../sidepacket_servable.hpp"
#include "src/filesystem/filesystem.hpp"
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#pragma warning(pop)
#include <memory>
#include <string>
#include <unordered_map>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace ovms {

struct RerankServable : SidepacketServable {
    bool addBosToken = true;
    bool isQwen3 = false;
    bool hasPositionIds = false;
    bool hasBeamIdx = false;

    RerankServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) :
        SidepacketServable(modelDir, targetDevice, pluginConfig, graphPath) {
        std::filesystem::path tokenizerConfigPath = (parsedModelsPath / "tokenizer_config.json");
        if (!std::filesystem::exists(tokenizerConfigPath)) {
            return;
        }
        std::ifstream ifs(tokenizerConfigPath.string());
        if (!ifs.is_open()) {
            return;
        }
        rapidjson::Document tokenizerConfig;
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::ParseResult parseResult = tokenizerConfig.ParseStream(isw);
        if (parseResult.Code()) {
            SPDLOG_ERROR("Parsing tokenizer_config.json failed: {}", rapidjson::GetParseError_En(parseResult.Code()));
            return;
        }
        if (tokenizerConfig.HasMember("add_bos_token") && tokenizerConfig["add_bos_token"].IsBool() && tokenizerConfig["add_bos_token"].IsFalse()) {
            SPDLOG_DEBUG("Rerank model add_bos_token set to false");
            addBosToken = false;
        }
    }

protected:
    std::shared_ptr<ov::Model> applyPrePostProcessing(ov::Core& core, std::shared_ptr<ov::Model> model, ov::AnyMap& properties) override {
        // Detect Qwen3 model type from config.json
        std::filesystem::path configPath = parsedModelsPath / "config.json";
        if (std::filesystem::exists(configPath)) {
            std::ifstream ifs(configPath.string());
            if (ifs.is_open()) {
                rapidjson::Document modelConfig;
                rapidjson::IStreamWrapper isw(ifs);
                rapidjson::ParseResult parseResult = modelConfig.ParseStream(isw);
                if (!parseResult.Code()) {
                    if (modelConfig.HasMember("model_type") && modelConfig["model_type"].IsString()) {
                        std::string modelType = modelConfig["model_type"].GetString();
                        if (modelType == "qwen3") {
                            SPDLOG_INFO("Detected Qwen3 reranker model, applying specialized postprocessing");
                            isQwen3 = true;
                        }
                    }
                }
            }
        }

        if (!isQwen3) {
            return model;
        }

        // Check model inputs for position_ids and beam_idx
        for (const auto& input : model->inputs()) {
            if (input.get_any_name() == "position_ids") {
                hasPositionIds = true;
                SPDLOG_DEBUG("Qwen3 reranker model has position_ids input");
            }
            if (input.get_any_name() == "beam_idx") {
                hasBeamIdx = true;
                SPDLOG_DEBUG("Qwen3 reranker model has beam_idx input");
            }
        }

        // Check output shape — only apply postprocessing for CausalLM models (3D output)
        ov::PartialShape outputShape = model->get_output_partial_shape(0);
        if (outputShape.rank().get_length() == 2) {
            // Already a 2D output (text-classification export) — postprocessing won't help
            // because the classification head has random weights
            SPDLOG_WARN("Qwen3 reranker has 2D output shape (text-classification export). "
                        "Re-export with --task text-generation for correct scoring.");
            return model;
        }

        // Look up yes/no token IDs
        int64_t yesTokenId = -1;
        int64_t noTokenId = -1;
        {
            auto yesTokens = tokenizer->encode("yes");
            if (yesTokens.input_ids.get_size() == 1 && yesTokens.input_ids.get_element_type() == ov::element::i64) {
                yesTokenId = reinterpret_cast<int64_t*>(yesTokens.input_ids.data())[0];
            }
            auto noTokens = tokenizer->encode("no");
            if (noTokens.input_ids.get_size() == 1 && noTokens.input_ids.get_element_type() == ov::element::i64) {
                noTokenId = reinterpret_cast<int64_t*>(noTokens.input_ids.data())[0];
            }
        }

        if (yesTokenId < 0 || noTokenId < 0) {
            SPDLOG_ERROR("Failed to look up yes/no token IDs for Qwen3 reranker");
            return model;
        }
        SPDLOG_INFO("Qwen3 reranker token IDs: yes={}, no={}", yesTokenId, noTokenId);

        // Apply Qwen3 postprocessing to model graph
        // Ported from openvino-2026.0-genai text_rerank_pipeline.cpp apply_qwen3_postprocessing()
        //
        // Input: model output logits [batch, seq_len, vocab_size]
        // Output: [batch, 1] tensor containing (yes_logit - no_logit)
        //         sigmoid of this equals softmax P(yes), so OVMS's existing sigmoid scoring works.
        ov::preprocess::PrePostProcessor processor(model);

        processor.output().postprocess().custom(
            [yesTokenId, noTokenId](const ov::Output<ov::Node>& node) -> std::shared_ptr<ov::Node> {
                // Step 1: Slice last token — [batch, seq_len, vocab] → [batch, 1, vocab]
                auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
                auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
                auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                auto axis1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

                auto lastTokenSlice = std::make_shared<ov::op::v8::Slice>(node, start, stop, step, axis1);

                // Step 2: Squeeze seq_len dim — [batch, 1, vocab] → [batch, vocab]
                auto squeezed = std::make_shared<ov::op::v0::Squeeze>(lastTokenSlice, axis1);

                // Step 3: Gather yes and no logits — [batch, vocab] → [batch, 2]
                auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2},
                    std::vector<int64_t>{noTokenId, yesTokenId});
                auto gatherAxis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                auto gathered = std::make_shared<ov::op::v8::Gather>(squeezed, indices, gatherAxis);

                // Step 4: Compute yes_logit - no_logit → [batch, 1]
                // gathered[:, 0] = no_logit, gathered[:, 1] = yes_logit
                auto yesStart = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                auto yesStop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
                auto yesSlice = std::make_shared<ov::op::v8::Slice>(gathered, yesStart, yesStop, step, gatherAxis);

                auto noStart = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
                auto noStop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                auto noSlice = std::make_shared<ov::op::v8::Slice>(gathered, noStart, noStop, step, gatherAxis);

                // yes_logit - no_logit → sigmoid of this = softmax P(yes)
                auto diff = std::make_shared<ov::op::v1::Subtract>(yesSlice, noSlice);

                return diff;  // [batch, 1]
            });

        return processor.build();
    }
};

using RerankServableMap = std::unordered_map<std::string, std::shared_ptr<RerankServable>>;
}  // namespace ovms
