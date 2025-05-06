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

#include <string>

#include <openvino/genai/image_generation/text2image_pipeline.hpp>

namespace ovms {

struct ImageGenerationPipelines {
    ov::genai::Text2ImagePipeline text2ImagePipeline;

    ImageGenerationPipelines() = delete;
    ImageGenerationPipelines(const std::string& models_path);
    ImageGenerationPipelines(const ImageGenerationPipelines&) = delete;
    ImageGenerationPipelines& operator=(const ImageGenerationPipelines&) = delete;
    ImageGenerationPipelines(ImageGenerationPipelines&&) = delete;
    ImageGenerationPipelines& operator=(ImageGenerationPipelines&&) = delete;
};

}  // namespace ovms
