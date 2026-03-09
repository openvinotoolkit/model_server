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
    const std::vector<std::string>& device) {
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
        pipeline.compile(device[0], args.pluginConfig);
    } else {
        SPDLOG_DEBUG("Image Generation Pipeline compiling to devices: text_encode={} denoise={} vae={}", device[0], device[1], device[2]);
        pipeline.compile(device[0], device[1], device[2], args.pluginConfig);
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
        reshapeAndCompile(*inpaintingPipeline, args, device);
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
            reshapeAndCompile(*image2ImagePipeline, args, device);
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
            reshapeAndCompile(*text2ImagePipeline, args, device);
            SPDLOG_DEBUG("Text2ImagePipeline created from disk (fallback)");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to create Text2ImagePipeline from disk: {}", e.what());
            text2ImagePipeline.reset();
        }
    }

    if (!inpaintingPipeline && !image2ImagePipeline && !text2ImagePipeline) {
        throw std::runtime_error("Failed to create any image generation pipeline from: " + args.modelsPath);
    }

    SPDLOG_INFO("Image Generation Pipelines ready — T2I: {} | I2I: {} | INP: {}",
        text2ImagePipeline ? "OK" : "N/A",
        image2ImagePipeline ? "OK" : "N/A",
        inpaintingPipeline ? "OK" : "N/A");
}
}  // namespace ovms
