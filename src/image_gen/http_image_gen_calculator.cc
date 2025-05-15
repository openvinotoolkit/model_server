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
#include "../image_conversion.hpp"

#include "pipelines.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

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

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        if (payload.parsedJson->HasParseError())
            return absl::InvalidArgumentError("Failed to parse JSON");

        if (!payload.parsedJson->IsObject()) {
            return absl::InvalidArgumentError("JSON body must be an object");
        }

        // get prompt field as string
        auto promptIt = payload.parsedJson->FindMember("prompt");
        if (promptIt == payload.parsedJson->MemberEnd()) {
            return absl::InvalidArgumentError("prompt field is missing in JSON body");
        }
        if (!promptIt->value.IsString()) {
            return absl::InvalidArgumentError("prompt field is not a string");
        }
        std::string prompt = promptIt->value.GetString();

        // TODO: Support more pipeline types
        // Depending on URI, select text2ImagePipeline/image2ImagePipeline/inpaintingPipeline

        // curl -X POST localhost:11338/v3/images/generations -H "Content-Type: application/json" -d '{ "model": "endpoint", "prompt": "A cute baby sea otter", "n": 1, "size": "1024x1024" }'
        ov::genai::Text2ImagePipeline request = pipe->text2ImagePipeline.clone();
        ov::Tensor image = request.generate(prompt,
            ov::AnyMap{
                ov::genai::width(512),  // todo: get from req
                ov::genai::height(512),  // todo: get from req
                ov::genai::num_inference_steps(20),  // todo: get from req
                ov::genai::num_images_per_prompt(1)});  // todo: get from req

        std::string res = save_image_stbi(image);

        // Convert the image to a base64 string
        std::string base64_image;
        absl::Base64Escape(res, &base64_image);

        // Create the JSON response
        std::string json_response = absl::StrCat("{\"data\":[{\"b64_json\":\"", base64_image, "\"}]}");
        // Produce std::string packet
        auto output = absl::make_unique<std::string>(json_response);
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }
};

const std::string ImageGenCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string ImageGenCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(ImageGenCalculator);

}  // namespace mediapipe
