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

ImageGenerationPipelines::ImageGenerationPipelines(const ImageGenPipelineArgs& args) {
//    image2ImagePipeline(ov::genai::Image2ImagePipeline(args.modelsPath,
//        args.device.value_or("CPU"),
//        args.pluginConfig)),
//    text2ImagePipeline(image2ImagePipeline),
//    args(args) {
  
    if (args.device.has_value() && args.device.value() == "NPU") {
        this->image2ImagePipeline = std::make_unique<ov::genai::Image2ImagePipeline>(args.modelsPath);
        this->image2ImagePipeline->reshape(1/*???*/, args.defaultResolution.value().first, args.defaultResolution.value().second, /*guidance scale??*/0.1f);
        this->image2ImagePipeline->compile(args.device.value_or("CPU"), args.pluginConfig);
        this->text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(*this->image2ImagePipeline);
        this->args = args;
    } else {
        this->image2ImagePipeline = std::make_unique<ov::genai::Image2ImagePipeline>(args.modelsPath,
            args.device.value_or("CPU"),
            args.pluginConfig);
        this->text2ImagePipeline = std::make_unique<ov::genai::Text2ImagePipeline>(*this->image2ImagePipeline);
        this->args = args;
    }
}
}  // namespace ovms
