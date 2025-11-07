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
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <openvino/genai/cache_eviction.hpp>
#include <openvino/genai/sparse_attention.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../json_parser.hpp"
#include "../../../logging.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../../status.hpp"
#include "llm_executor.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"

namespace ovms {

ov::genai::SparseAttentionConfig prepareSparseAttentionConfig(const mediapipe::LLMCalculatorOptions& nodeOptions) {
    ov::genai::SparseAttentionMode mode;
    if (nodeOptions.sparse_attention_config().mode() == mediapipe::LLMCalculatorOptions::SparseAttentionConfig::TRISHAPE) {
        mode = ov::genai::SparseAttentionMode::TRISHAPE;
    } else {
        mode = ov::genai::SparseAttentionMode::XATTENTION;
    }
    // Use default constructor to rely on GenAI defined defaults if user did not set specific fields
    ov::genai::SparseAttentionConfig sparseAttentionConfig;
    sparseAttentionConfig.mode = mode;
    if (nodeOptions.sparse_attention_config().has_num_last_dense_tokens_in_prefill()) {
        sparseAttentionConfig.num_last_dense_tokens_in_prefill = nodeOptions.sparse_attention_config().num_last_dense_tokens_in_prefill();
    }
    if (nodeOptions.sparse_attention_config().has_num_retained_start_tokens_in_cache()) {
        sparseAttentionConfig.num_retained_start_tokens_in_cache = nodeOptions.sparse_attention_config().num_retained_start_tokens_in_cache();
    }
    if (nodeOptions.sparse_attention_config().has_num_retained_recent_tokens_in_cache()) {
        sparseAttentionConfig.num_retained_recent_tokens_in_cache = nodeOptions.sparse_attention_config().num_retained_recent_tokens_in_cache();
    }
    if (nodeOptions.sparse_attention_config().has_xattention_threshold()) {
        sparseAttentionConfig.xattention_threshold = nodeOptions.sparse_attention_config().xattention_threshold();
    }
    if (nodeOptions.sparse_attention_config().has_xattention_block_size()) {
        sparseAttentionConfig.xattention_block_size = nodeOptions.sparse_attention_config().xattention_block_size();
    }
    if (nodeOptions.sparse_attention_config().has_xattention_stride()) {
        sparseAttentionConfig.xattention_stride = nodeOptions.sparse_attention_config().xattention_stride();
    }

    return sparseAttentionConfig;
}

ov::genai::CacheEvictionConfig prepareCacheEvictionConfig(const mediapipe::LLMCalculatorOptions& nodeOptions) {
    ov::genai::AggregationMode aggregationMode;
    if (nodeOptions.cache_eviction_config().aggregation_mode() == mediapipe::LLMCalculatorOptions::CacheEvictionConfig::SUM) {
        aggregationMode = ov::genai::AggregationMode::SUM;
    } else {
        aggregationMode = ov::genai::AggregationMode::NORM_SUM;
    }
    size_t startSize = nodeOptions.cache_eviction_config().start_size();
    size_t recentSize = nodeOptions.cache_eviction_config().recent_size();
    size_t maxCacheSize = nodeOptions.cache_eviction_config().max_cache_size();
    bool applyRotation = nodeOptions.cache_eviction_config().apply_rotation();
    size_t snapkvWindowSize = nodeOptions.cache_eviction_config().snapkv_window_size();

    ov::genai::KVCrushConfig kvCrushConfig;
    if (nodeOptions.cache_eviction_config().has_kv_crush_config()) {
        ov::genai::KVCrushAnchorPointMode anchorPointMode;
        switch (nodeOptions.cache_eviction_config().kv_crush_config().anchor_point_mode()) {
        case mediapipe::LLMCalculatorOptions::KVCrushConfig::RANDOM:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::RANDOM;
            break;
        case mediapipe::LLMCalculatorOptions::KVCrushConfig::ZEROS:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::ZEROS;
            break;
        case mediapipe::LLMCalculatorOptions::KVCrushConfig::ONES:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::ONES;
            break;
        case mediapipe::LLMCalculatorOptions::KVCrushConfig::MEAN:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::MEAN;
            break;
        case mediapipe::LLMCalculatorOptions::KVCrushConfig::ALTERNATING:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::ALTERNATING;
            break;
        default:
            anchorPointMode = ov::genai::KVCrushAnchorPointMode::RANDOM;
            break;
        }
        size_t budget = nodeOptions.cache_eviction_config().kv_crush_config().budget();
        size_t rngSeed = nodeOptions.cache_eviction_config().kv_crush_config().rng_seed();
        kvCrushConfig = ov::genai::KVCrushConfig(budget, anchorPointMode, rngSeed);
    }
    return ov::genai::CacheEvictionConfig(startSize, recentSize, maxCacheSize, aggregationMode, applyRotation, snapkvWindowSize, kvCrushConfig);
}

ov::genai::SchedulerConfig ContinuousBatchingServableInitializer::prepareDraftPipelineSchedulerConfig(const mediapipe::LLMCalculatorOptions& nodeOptions) {
    ov::genai::SchedulerConfig config;
    config.max_num_batched_tokens = nodeOptions.has_draft_max_num_batched_tokens() ? nodeOptions.draft_max_num_batched_tokens() : nodeOptions.max_num_batched_tokens();
    config.cache_size = nodeOptions.has_draft_cache_size() ? nodeOptions.draft_cache_size() : nodeOptions.cache_size();
    config.dynamic_split_fuse = nodeOptions.has_draft_dynamic_split_fuse() ? nodeOptions.draft_dynamic_split_fuse() : nodeOptions.dynamic_split_fuse();
    config.max_num_seqs = nodeOptions.has_draft_max_num_seqs() ? nodeOptions.draft_max_num_seqs() : nodeOptions.max_num_seqs();
    config.enable_prefix_caching = nodeOptions.enable_prefix_caching();
    return config;
}

Status ContinuousBatchingServableInitializer::initialize(std::shared_ptr<GenAiServable>& servable, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) {
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (!status.ok()) {
        return status;
    }
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());
    properties->modelsPath = parsedModelsPath;
    std::filesystem::path modelGenerationConfigPath = std::filesystem::path(parsedModelsPath) / "generation_config.json";
    if (std::filesystem::exists(modelGenerationConfigPath)) {
        properties->baseGenerationConfig = ov::genai::GenerationConfig(modelGenerationConfigPath.string());
    }
    if (nodeOptions.has_tool_parser()) {
        properties->toolParserName = nodeOptions.tool_parser();
    }
    if (nodeOptions.has_reasoning_parser()) {
        properties->reasoningParserName = nodeOptions.reasoning_parser();
    }

    properties->schedulerConfig.max_num_batched_tokens = nodeOptions.max_num_batched_tokens();
    properties->schedulerConfig.cache_size = nodeOptions.cache_size();
    properties->schedulerConfig.dynamic_split_fuse = nodeOptions.dynamic_split_fuse();
    properties->schedulerConfig.max_num_seqs = nodeOptions.max_num_seqs();
    properties->schedulerConfig.enable_prefix_caching = nodeOptions.enable_prefix_caching();

    if (nodeOptions.has_cache_eviction_config()) {
        properties->schedulerConfig.cache_eviction_config = prepareCacheEvictionConfig(nodeOptions);
        properties->schedulerConfig.use_cache_eviction = true;
    } else {
        properties->schedulerConfig.use_cache_eviction = false;
    }

    if (nodeOptions.has_sparse_attention_config()) {
        properties->schedulerConfig.use_sparse_attention = true;
        properties->schedulerConfig.sparse_attention_config = prepareSparseAttentionConfig(nodeOptions);
    } else {
        properties->schedulerConfig.use_sparse_attention = false;
    }

    properties->device = nodeOptions.device();
    properties->bestOfLimit = nodeOptions.best_of_limit();
    properties->enableToolGuidedGeneration = nodeOptions.enable_tool_guided_generation();

    if (!nodeOptions.draft_models_path().empty()) {
        // draft models
        auto fsDraftModelsPath = std::filesystem::path(nodeOptions.draft_models_path());
        std::string draftPipelinePath;
        if (fsDraftModelsPath.is_relative()) {
            draftPipelinePath = (std::filesystem::path(graphPath) / fsDraftModelsPath).string();
        } else {
            draftPipelinePath = fsDraftModelsPath.string();
        }
        auto draftSchedulerConfig = prepareDraftPipelineSchedulerConfig(nodeOptions);

        try {
            auto draftPipeline = ov::genai::draft_model(draftPipelinePath, nodeOptions.draft_device(),
                ov::genai::scheduler_config(draftSchedulerConfig));
            properties->pluginConfig.insert(draftPipeline);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error during draft model initialization for draft models_path: {} exception: {}", draftPipelinePath, e.what());
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        } catch (...) {
            SPDLOG_ERROR("Error during draft model initialization for draft models_path: {}", draftPipelinePath);
            return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
        }

    } else if (nodeOptions.has_draft_max_num_batched_tokens() || nodeOptions.has_draft_cache_size() || nodeOptions.has_draft_dynamic_split_fuse() || nodeOptions.has_draft_max_num_seqs() || nodeOptions.has_draft_block_size() || nodeOptions.has_draft_device()) {
        SPDLOG_ERROR("Draft model path is not provided, but draft scheduler options are set.");
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), properties->pluginConfig);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        return status;
    }

    properties->tokenizerPluginConfig = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
    try {
        properties->pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(parsedModelsPath,
            properties->schedulerConfig, properties->device,
            properties->pluginConfig, properties->tokenizerPluginConfig);
        properties->tokenizer = properties->pipeline->get_tokenizer();
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {} exception: {}", parsedModelsPath, e.what());
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for models_path: {}", parsedModelsPath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }
    loadChatTemplate(properties, parsedModelsPath);
    if (nodeOptions.has_max_tokens_limit()) {
        properties->maxTokensLimit = nodeOptions.max_tokens_limit();
    }
    properties->maxModelLength = parseMaxModelLength(parsedModelsPath);

    properties->llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(properties->pipeline);

    return StatusCode::OK;
}

}  // namespace ovms
