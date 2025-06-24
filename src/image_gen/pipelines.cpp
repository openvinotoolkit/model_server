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

namespace ovms {

ImageGenerationPipelines::ImageGenerationPipelines(const ImageGenPipelineArgs& args) :
    args(args) {
    const std::string device = args.device.value_or("CPU");
    if (device == "NPU") {
        text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(args.modelsPath);
        text2ImagePipeline->reshape(
            args.defaultNumImagesPerPrompt.value_or(ov::genai::ImageGenerationConfig().num_images_per_prompt),
            args.defaultResolution.value().first,   // at this point it should be validated for existence
            args.defaultResolution.value().second,  // at this point it should be validated for existence
            args.defaultGuidanceScale.value_or(ov::genai::ImageGenerationConfig().guidance_scale));
        text2ImagePipeline->compile(device, args.pluginConfig);
    } else {
        text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(args.modelsPath, device, args.pluginConfig);
    }
}
// TODO: Make other pipelines out of the basic one, with shared models, GenAI API supports that
}  // namespace ovms
