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
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Image Generation Step: {}/{}", step + 1, num_steps);
    return false;
}
// written out separately to avoid msvc crashing when using try-catch in process method ...
static absl::Status generateTensor(ov::genai::Text2ImagePipeline& request,
    const std::string& prompt, ov::AnyMap& requestOptions,
    std::unique_ptr<ov::Tensor>& images) {
    try {
        requestOptions.insert(ov::genai::callback(progress_bar));
        images = std::make_unique<ov::Tensor>(request.generate(prompt, requestOptions));
        auto dims = images->get_shape();
        std::stringstream ss;
        for (const auto& dim : dims) {
            ss << dim << " ";
        }
        ss << " element type: " << images->get_element_type().get_type_name();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator generated tensor: {}", ss.str());
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "ImageGenCalculator Error: {}", e.what());
        return absl::InternalError("Error during images generation");
    } catch (...) {
        return absl::InternalError("Unknown error during image generation");
    }
    return absl::OkStatus();
}
// written out separately to avoid msvc crashing when using try-catch in process method ...
static absl::Status generateTensorImg2Img(ov::genai::Image2ImagePipeline& request,
    const std::string& prompt, ov::Tensor image, ov::AnyMap& requestOptions,
    std::unique_ptr<ov::Tensor>& images) {
    try {
        requestOptions.insert(ov::genai::callback(progress_bar));
        images = std::make_unique<ov::Tensor>(request.generate(prompt, image, requestOptions));
        auto dims = images->get_shape();
        std::stringstream ss;
        for (const auto& dim : dims) {
            ss << dim << " ";
        }
        ss << " element type: " << images->get_element_type().get_type_name();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator generated tensor: {}", ss.str());
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "ImageGenCalculator Error: {}", e.what());
        return absl::InternalError("Error during images generation");
    } catch (...) {
        return absl::InternalError("Unknown error during image generation");
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

        std::unique_ptr<ov::Tensor> images;  // output

        if (absl::StartsWith(payload.uri, "/v3/images/generations")) {
            if (payload.parsedJson->HasParseError())
                return absl::InvalidArgumentError("Failed to parse JSON");

            if (!payload.parsedJson->IsObject()) {
                return absl::InvalidArgumentError("JSON body must be an object");
            }

            SET_OR_RETURN(std::string, prompt, getPromptField(*payload.parsedJson));
            SET_OR_RETURN(ov::AnyMap, requestOptions, getImageGenerationRequestOptions(*payload.parsedJson, pipe->args));

            ov::genai::Text2ImagePipeline request = pipe->text2ImagePipeline->clone();

            auto status = generateTensor(request, prompt, requestOptions, images);
            if (!status.ok()) {
                return status;
            }
        } else if (absl::StartsWith(payload.uri, "/v3/images/edits")) {
            if (payload.multipartParser->hasParseError())
                return absl::InvalidArgumentError("Failed to parse multipart data");

            SET_OR_RETURN(std::string, prompt, getPromptField(*payload.multipartParser));
            SET_OR_RETURN(std::optional<std::string>, image, getStringFromPayload(*payload.multipartParser, "image"));
            RET_CHECK(image.has_value() && !image.value().empty()) << "Image field is missing in multipart body";

            ov::Tensor imageTensor;
            try {
                imageTensor = loadImageStbiFromMemory(image.value());
            } catch (std::runtime_error& e) {
                std::stringstream ss;
                ss << "Image parsing failed: " << e.what();
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
                return absl::InvalidArgumentError(ss.str());
            }

            SET_OR_RETURN(ov::AnyMap, requestOptions, getImageEditRequestOptions(*payload.multipartParser, pipe->args));

            ov::genai::Image2ImagePipeline request = pipe->image2ImagePipeline->clone();
            auto status = generateTensorImg2Img(request, prompt, imageTensor, requestOptions, images);
            if (!status.ok()) {
                return status;
            }
        } else {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

        auto outputOrStatus = generateJSONResponseFromOvTensor(*images);
        RETURN_IF_HOLDS_STATUS(outputOrStatus);
        auto output = std::move(std::get<std::unique_ptr<std::string>>(outputOrStatus));
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "ImageGenCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string ImageGenCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string ImageGenCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(ImageGenCalculator);

}  // namespace mediapipe
