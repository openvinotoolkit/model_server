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
#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../logging.hpp"

#include "pipelines.hpp"

using namespace ovms;

namespace mediapipe {

using ImageGenerationPipelinesMap = std::unordered_map<std::string, std::shared_ptr<ImageGenerationPipelines>>;

const std::string IMAGE_GEN_SESSION_SIDE_PACKET_TAG = "IMAGE_GEN_NODE_RESOURCES";

class ImageGenCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->InputSidePackets().Tag(IMAGE_GEN_SESSION_SIDE_PACKET_TAG).Set<ImageGenerationPipelinesMap>();  // TODO: template?
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Process start", cc->NodeName());

        ImageGenerationPipelinesMap pipelinesNap = cc->InputSidePackets().Tag(IMAGE_GEN_SESSION_SIDE_PACKET_TAG).Get<ImageGenerationPipelinesMap>();
        auto it = pipelinesNap.find(cc->NodeName());
        RET_CHECK(it != pipelinesNap.end()) << "Could not find initialized Image Gen node named: " << cc->NodeName();
        auto pipe = it->second;

        // curl -X POST localhost:11338/v3/endpoint -d '{}'
        ov::genai::Text2ImagePipeline::GenerationRequest request = pipe->text2ImagePipeline.create_generation_request();
        ov::Tensor image = request.generate("a cat",  // TODO: get from payload
            ov::AnyMap{
                ov::genai::width(512),
                ov::genai::height(512),
                ov::genai::num_inference_steps(20),
                ov::genai::num_images_per_prompt(1)});

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }
};

const std::string ImageGenCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string ImageGenCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(ImageGenCalculator);

}  // namespace mediapipe
