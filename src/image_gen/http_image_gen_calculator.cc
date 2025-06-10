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
#include "imagegenutils.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

using namespace ovms;

namespace mediapipe {

using ImageGenerationPipelinesMap = std::unordered_map<std::string, std::shared_ptr<ImageGenerationPipelines>>;

const std::string IMAGE_GEN_SESSION_SIDE_PACKET_TAG = "IMAGE_GEN_NODE_RESOURCES";

static bool progress_bar(size_t step, size_t num_steps, ov::Tensor&) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image Generation Step: {}/{}", step, num_steps);
    return false;
}
// written out separately to avoid msvc crashing when using try-catch in process method ...
static absl::Status generateTensor(ov::genai::Text2ImagePipeline& request,
    const std::string& prompt, ov::AnyMap& requestOptions,
    std::unique_ptr<ov::Tensor>& images) {
    try {
        requestOptions.insert(ov::genai::callback(progress_bar));
        images = std::make_unique<ov::Tensor>(request.generate(prompt, requestOptions));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "ImageGenCalculator Error: {}", e.what());
        return absl::InternalError("Error during images generation");
    } catch (...) {
        return absl::InternalError("Unknown error during image generation");
    }
    return absl::OkStatus();
}
// written out separately to avoid msvc crashing when using try-catch in process method ...
static absl::Status convert2String(const std::unique_ptr<ov::Tensor>& images, std::unique_ptr<std::string>& imageAsString) {
    try {
        *imageAsString = saveImageStbi(*images);
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "ImageGenCalculator Error: {}", e.what());
        return absl::InternalError("Error during image conversion");
    } catch (...) {
        return absl::InternalError("Unknown error during image conversion");
    }
    return absl::OkStatus();
}

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

        ImageGenerationPipelinesMap pipelinesMap = cc->InputSidePackets().Tag(IMAGE_GEN_SESSION_SIDE_PACKET_TAG).Get<ImageGenerationPipelinesMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized Image Gen node named: " << cc->NodeName();
        auto pipe = it->second;

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        if (payload.parsedJson->HasParseError())
            return absl::InvalidArgumentError("Failed to parse JSON");

        if (!payload.parsedJson->IsObject()) {
            return absl::InvalidArgumentError("JSON body must be an object");
        }
        SET_OR_RETURN(std::string, prompt, getPromptField(payload));

        // TODO: Support more pipeline types
        // Depending on URI, select text2ImagePipeline/image2ImagePipeline/inpaintingPipeline

        ov::genai::Text2ImagePipeline request = pipe->text2ImagePipeline.clone();
        SET_OR_RETURN(ov::AnyMap, requestOptions, getImageGenerationRequestOptions(payload, pipe->args));
        // preview limitation put here to not mess up tests underneath
        auto imagesPerPromptIt = requestOptions.find("num_images_per_prompt");
        if (imagesPerPromptIt != requestOptions.end()) {
            auto numImages = imagesPerPromptIt->second.as<int>();
            if (numImages != 1) {
                return absl::InvalidArgumentError(absl::StrCat("Only 1 image in response can be requested. n value:", numImages, " is not supported."));
            }
        }

        std::unique_ptr<ov::Tensor> images;
        auto status = generateTensor(request, prompt, requestOptions, images);
        if (!status.ok()) {
            return status;
        }
        auto imageAsString = std::make_unique<std::string>();
        status = convert2String(images, imageAsString);
        if (!status.ok()) {
            return status;
        }

        std::string base64image;
        absl::Base64Escape(*imageAsString, &base64image);
        // Create the JSON response
        auto output = generateJSONResponseFromB64Image(base64image);
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string ImageGenCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string ImageGenCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(ImageGenCalculator);

}  // namespace mediapipe
