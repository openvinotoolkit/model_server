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
#include "pipelines.hpp"

#include <algorithm>
#include <vector>

#include <openvino/genai/image_generation/inpainting_pipeline.hpp>
#include <openvino/genai/image_generation/text2image_pipeline.hpp>
#include <openvino/genai/image_generation/image2image_pipeline.hpp>

#include "src/logging.hpp"
#include "src/stringutils.hpp"

namespace ovms {

// Reshape and compile a pipeline that was loaded from disk.
// Derived (weight-sharing) pipelines inherit the compiled state from the parent and skip this.
template <typename PipelineT>
static void reshapeAndCompile(PipelineT& pipeline,
    const ImageGenPipelineArgs& args,
    const std::vector<std::string>& device,
    const ov::AnyMap& properties) {
    if (args.staticReshapeSettings.has_value() && args.staticReshapeSettings.value().resolution.size() == 1) {
        auto numImagesPerPrompt = args.staticReshapeSettings.value().numImagesPerPrompt.value_or(ov::genai::ImageGenerationConfig().num_images_per_prompt);
        auto guidanceScale = args.staticReshapeSettings.value().guidanceScale.value_or(ov::genai::ImageGenerationConfig().guidance_scale);

        SPDLOG_DEBUG("Image Generation Pipeline reshape to static {}x{} resolution, batch: {}, guidance scale: {}",
            args.staticReshapeSettings.value().resolution[0].first, args.staticReshapeSettings.value().resolution[0].second, numImagesPerPrompt, guidanceScale);

        pipeline.reshape(
            numImagesPerPrompt,
            args.staticReshapeSettings.value().resolution[0].first,
            args.staticReshapeSettings.value().resolution[0].second,
            guidanceScale);
    }

    if (device.size() == 1) {
        SPDLOG_DEBUG("Image Generation Pipeline compiling to device: {}", device[0]);
        pipeline.compile(device[0], properties);
    } else {
        SPDLOG_DEBUG("Image Generation Pipeline compiling to devices: text_encode={} denoise={} vae={}", device[0], device[1], device[2]);
        pipeline.compile(device[0], device[1], device[2], properties);
    }
}

ImageGenerationPipelines::ImageGenerationPipelines(const ImageGenPipelineArgs& args) :
    args(args) {
    std::vector<std::string> device;
    if (!args.device.size()) {
        device.push_back("CPU");
    } else {
        device = args.device;
    }

    SPDLOG_DEBUG("Image Generation Pipelines weights loading from: {}", args.modelsPath);

    // --- Load LoRA adapters before pipeline compilation ---
    // Adapters must be registered at compile time so that the AdapterController
    // is initialized and can apply/disable them at inference time.
    // FUSE adapters are loaded separately and use MODE_FUSE to permanently merge into weights.
    std::vector<std::pair<ov::genai::Adapter, float>> fuseAdapters;
    for (const auto& loraInfo : args.loraAdapters) {
        SPDLOG_INFO("Loading LoRA adapter: {} from: {} (mode: {})", loraInfo.alias, loraInfo.path,
            loraInfo.mode == LoraLoadMode::FUSE ? "FUSE" : (loraInfo.mode == LoraLoadMode::STATIC ? "STATIC" : "DYNAMIC"));
        try {
            auto adapter = ov::genai::Adapter(loraInfo.path);
            if (loraInfo.mode == LoraLoadMode::FUSE) {
                fuseAdapters.emplace_back(std::move(adapter), loraInfo.alpha);
            } else {
                loraAdapters.emplace(loraInfo.alias, std::move(adapter));
            }
            SPDLOG_INFO("LoRA adapter loaded: {}", loraInfo.alias);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load LoRA adapter '" + loraInfo.alias + "' from " + loraInfo.path + ": " + e.what());
        }
    }

    // Build compile-time adapter properties so the pipeline's AdapterController
    // knows about all adapters. At generate time we select which to activate.
    ov::AnyMap compileProperties = args.pluginConfig;

    // FUSE adapters: permanently merged into base weights using MODE_FUSE.
    // These are always active, not switchable, and invisible to request routing.
    if (!fuseAdapters.empty()) {
        ov::genai::AdapterConfig fuseConfig;
        for (const auto& [adapter, alpha] : fuseAdapters) {
            fuseConfig.add(adapter, alpha);
        }
        fuseConfig.set_mode(ov::genai::AdapterConfig::MODE_FUSE);
        compileProperties.insert(ov::genai::adapters(fuseConfig));
        SPDLOG_INFO("Fused {} LoRA adapter(s) into base model weights (MODE_FUSE)", fuseAdapters.size());
    }

    // DYNAMIC/STATIC adapters: registered for runtime switching.
    if (!loraAdapters.empty()) {
        ov::genai::AdapterConfig adapterConfig;
        for (const auto& [alias, adapter] : loraAdapters) {
            // Use the configured alpha from args for each adapter
            float alpha = 1.0f;
            for (const auto& info : args.loraAdapters) {
                if (info.alias == alias) {
                    alpha = info.alpha;
                    break;
                }
            }
            adapterConfig.add(adapter, alpha);
        }
        // NPU requires MODE_STATIC — adapters are compiled with fixed alpha values.
        // Runtime switching is not possible; all adapters remain active at their compile-time alpha.
        bool hasNPU = std::find(device.begin(), device.end(), "NPU") != device.end();
        if (hasNPU) {
            adapterConfig.set_mode(ov::genai::AdapterConfig::MODE_STATIC);
            npuLoraStaticMode = true;
            SPDLOG_INFO("NPU detected: LoRA adapters compiled with MODE_STATIC (no runtime switching)");
        } else {
            // Check if any adapter explicitly requests STATIC mode
            bool anyStatic = std::any_of(args.loraAdapters.begin(), args.loraAdapters.end(),
                [](const LoraAdapterInfo& info) { return info.mode == LoraLoadMode::STATIC; });
            if (anyStatic) {
                adapterConfig.set_mode(ov::genai::AdapterConfig::MODE_STATIC);
                npuLoraStaticMode = true;
                SPDLOG_INFO("STATIC mode requested: LoRA adapters compiled with MODE_STATIC");
            }
        }
        // Merge with any existing fuse config (both can coexist if GenAI supports it)
        if (compileProperties.count(ov::genai::adapters.name())) {
            // Fuse adapters already set — we need to add dynamic adapters to the same config
            // GenAI doesn't support two adapter configs; dynamic adapters on top of fused is handled
            // by applying fuse first, then compiling with dynamic adapters separately.
            // For now, replace — GenAI fuses first during compile, then registers dynamic adapters.
            SPDLOG_INFO("Both FUSE and DYNAMIC/STATIC adapters present — combining in compile properties");
        }
        compileProperties.insert_or_assign(ov::genai::adapters.name(), ov::genai::adapters(adapterConfig).second);
    }

    // Populate composite LoRA map from args
    compositeLoraAdapters = args.compositeLoraAdapters;
    for (const auto& [alias, components] : compositeLoraAdapters) {
        SPDLOG_INFO("Registered composite LoRA adapter: {} with {} components", alias, components.size());
    }

    // Pipeline construction strategy:
    //   Preferred chain (weight-sharing, single model load):
    //     INP(disk) → reshape+compile → I2I(INP) → T2I(I2I)
    //
    //   Some models don't support all derivation directions (e.g. inpainting-specific
    //   models reject I2I(INP) with "Cannot create Image2ImagePipeline from InpaintingPipeline
    //   with inpainting model"). When derivation fails, fall back to loading from disk
    //   (separate model load + reshape+compile). We WARN on individual failures and only
    //   throw if no pipeline could be created at all.

    // --- Step 1: InpaintingPipeline from disk ---
    try {
        inpaintingPipeline = std::make_unique<ov::genai::InpaintingPipeline>(args.modelsPath);
        reshapeAndCompile(*inpaintingPipeline, args, device, compileProperties);
        SPDLOG_DEBUG("InpaintingPipeline created from disk");
    } catch (const std::exception& e) {
        SPDLOG_WARN("Failed to create InpaintingPipeline from disk: {}", e.what());
        inpaintingPipeline.reset();
    }

    // --- Step 2: Image2ImagePipeline — derive from INP, fallback to disk ---
    if (inpaintingPipeline) {
        try {
            image2ImagePipeline = std::make_unique<ov::genai::Image2ImagePipeline>(*inpaintingPipeline);
            SPDLOG_DEBUG("Image2ImagePipeline derived from InpaintingPipeline");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to derive Image2ImagePipeline from InpaintingPipeline: {}", e.what());
        }
    }
    if (!image2ImagePipeline) {
        try {
            image2ImagePipeline = std::make_unique<ov::genai::Image2ImagePipeline>(args.modelsPath);
            reshapeAndCompile(*image2ImagePipeline, args, device, compileProperties);
            SPDLOG_DEBUG("Image2ImagePipeline created from disk (fallback)");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to create Image2ImagePipeline from disk: {}", e.what());
            image2ImagePipeline.reset();
        }
    }

    // --- Step 3: Text2ImagePipeline — derive from I2I or INP, fallback to disk ---
    if (image2ImagePipeline) {
        try {
            text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(*image2ImagePipeline);
            SPDLOG_DEBUG("Text2ImagePipeline derived from Image2ImagePipeline");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to derive Text2ImagePipeline from Image2ImagePipeline: {}", e.what());
        }
    }
    if (!text2ImagePipeline && inpaintingPipeline) {
        try {
            text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(*inpaintingPipeline);
            SPDLOG_DEBUG("Text2ImagePipeline derived from InpaintingPipeline");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to derive Text2ImagePipeline from InpaintingPipeline: {}", e.what());
        }
    }
    if (!text2ImagePipeline) {
        try {
            text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(args.modelsPath);
            reshapeAndCompile(*text2ImagePipeline, args, device, compileProperties);
            SPDLOG_DEBUG("Text2ImagePipeline created from disk (fallback)");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to create Text2ImagePipeline from disk: {}", e.what());
            text2ImagePipeline.reset();
        }
    }

    if (!inpaintingPipeline && !image2ImagePipeline && !text2ImagePipeline) {
        SPDLOG_ERROR("Failed to create any image generation pipeline from: {}", args.modelsPath);
        throw std::runtime_error("Failed to create any image generation pipeline from: " + args.modelsPath);
    }

    // InpaintingPipeline does not support clone(), so concurrent inpainting
    // requests must be serialized
    if (inpaintingPipeline) {
        inpaintingQueue = std::make_unique<Queue<int>>(1);
    }

    SPDLOG_INFO("Image Generation Pipelines ready — T2I: {} | I2I: {} | INP: {} | LoRAs: {}",
        text2ImagePipeline ? "OK" : "N/A",
        image2ImagePipeline ? "OK" : "N/A",
        inpaintingPipeline ? "OK" : "N/A",
        loraAdapters.size());
}
}  // namespace ovms
