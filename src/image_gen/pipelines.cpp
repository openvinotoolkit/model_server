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

    SPDLOG_DEBUG("Loading weights from: {}", args.modelsPath);
    text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(args.modelsPath);

    if (args.staticReshapeSettings.has_value() && args.staticReshapeSettings.value().resolution.size() == 1) {
        SPDLOG_DEBUG("Reshaping to static WxH: {}x{}", args.staticReshapeSettings.value().resolution[0].first, args.staticReshapeSettings.value().resolution[0].second);
        text2ImagePipeline->reshape(
            args.staticReshapeSettings.value().numImagesPerPrompt.value_or(ov::genai::ImageGenerationConfig().num_images_per_prompt),
            args.staticReshapeSettings.value().resolution[0].first,   // at this point it should be validated for existence
            args.staticReshapeSettings.value().resolution[0].second,  // at this point it should be validated for existence
            args.staticReshapeSettings.value().guidanceScale.value_or(ov::genai::ImageGenerationConfig().guidance_scale));
    }

    SPDLOG_DEBUG("Compiling Text2ImagePipeline on devices: {}", ovms::joins(device, ", "));

    if (device.size() == 1) {
        text2ImagePipeline->compile(device[0], args.pluginConfig);
    } else {
        text2ImagePipeline->compile(device[0], device[1], device[2], args.pluginConfig);
    }
}
// TODO: Make other pipelines out of the basic one, with shared models, GenAI API supports that
}  // namespace ovms
