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

#include "../logging.hpp"
#include "../stringutils.hpp"

namespace ovms {

ImageGenerationPipelines::ImageGenerationPipelines(const ImageGenPipelineArgs& args) :
    args(args) {
    std::vector<std::string> device;
    if (!args.device.size()) {
        device.push_back("CPU");
    } else {
        device = args.device;
    }

    SPDLOG_DEBUG("Image Generation Pipelines weights loading from: {}", args.modelsPath);

    image2ImagePipeline = std::make_unique<ov::genai::Image2ImagePipeline>(args.modelsPath);
    if (!image2ImagePipeline) {
        // TODO -> that should only turn off that routing
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create Image2ImagePipeline");
        throw std::runtime_error("Failed to create Image2ImagePipeline");
    }
    if (args.staticReshapeSettings.has_value() && args.staticReshapeSettings.value().resolution.size() == 1) {
        auto numImagesPerPrompt = args.staticReshapeSettings.value().numImagesPerPrompt.value_or(ov::genai::ImageGenerationConfig().num_images_per_prompt);
        auto guidanceScale = args.staticReshapeSettings.value().guidanceScale.value_or(ov::genai::ImageGenerationConfig().guidance_scale);

        SPDLOG_DEBUG("Image Generation Pipelines will be reshaped to static {}x{} resolution, batch: {}, guidance scale: {}",
            args.staticReshapeSettings.value().resolution[0].first, args.staticReshapeSettings.value().resolution[0].second, numImagesPerPrompt, guidanceScale);

        image2ImagePipeline->reshape(
            numImagesPerPrompt,
            args.staticReshapeSettings.value().resolution[0].first,   // at this point it should be validated for existence
            args.staticReshapeSettings.value().resolution[0].second,  // at this point it should be validated for existence
            guidanceScale);
    }

    if (device.size() == 1) {
        SPDLOG_DEBUG("Image Generation Pipelines compiling to devices: text_encode={} denoise={} vae={}", device[0], device[0], device[0]);
        image2ImagePipeline->compile(device[0], args.pluginConfig);
    } else {
        SPDLOG_DEBUG("Image Generation Pipelines compiling to devices: text_encode={} denoise={} vae={}", device[0], device[1], device[2]);
        image2ImagePipeline->compile(device[0], device[1], device[2], args.pluginConfig);
    }

    text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(*image2ImagePipeline);
    if (!text2ImagePipeline) {
        // TODO -> that should only turn off that routing
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create Text2ImagePipeline");
        throw std::runtime_error("Failed to create Text2ImagePipeline");
    }
    // TODO: Initialize optional GenAI pipelines based on model capabilities (e.g. inpainting support)
    // instead of constructing all of them unconditionally.
    inpaintingPipeline = std::make_unique<ov::genai::InpaintingPipeline>(*image2ImagePipeline);
    if (!inpaintingPipeline) {
        // TODO -> that should only turn off that routing
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create Text2ImagePipeline");
        throw std::runtime_error("Failed to create InpaintingPipeline");
    }
}
}  // namespace ovms
