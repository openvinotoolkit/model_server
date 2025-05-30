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
#include "imagegenutils.hpp"
#include <utility>
#include <set>
#include <string>
#include <memory>
#include <variant>

#include <openvino/genai/image_generation/text2image_pipeline.hpp>
#include <openvino/genai/image_generation/image2image_pipeline.hpp>

#include "absl/strings/str_cat.h"

#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include "src/stringutils.hpp"
namespace ovms {

std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const std::string& dimensions) {
    if (dimensions == "auto") {
        return std::nullopt;
    }
    size_t xPos = dimensions.find('x');
    if (xPos == std::string::npos) {
        return absl::InvalidArgumentError("size field is not in correct format");
    }
    std::string left = dimensions.substr(0, xPos);
    std::string right = dimensions.substr(xPos + 1);
    auto leftInt = ovms::stoi64(left);
    auto rightInt = ovms::stoi64(right);
    if (!leftInt.has_value() || !rightInt.has_value()) {
        return absl::InvalidArgumentError("size field is not in correct format");
    }
    if (leftInt.value() <= 0 || rightInt.value() <= 0) {
        return absl::InvalidArgumentError("size field values must be greater than 0");
    }
    return std::make_pair(leftInt.value(), rightInt.value());
}
std::variant<absl::Status, std::optional<resolution_t>> getDimensions(const ovms::HttpPayload& payload) {
    auto sizeIt = payload.parsedJson->FindMember("size");
    if (sizeIt == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!sizeIt->value.IsString()) {
        return absl::InvalidArgumentError("size field is not a string");
    }
    return getDimensions(sizeIt->value.GetString());
}

std::variant<absl::Status, std::optional<std::string>> getStringFromPayload(const ovms::HttpPayload& payload, const std::string& keyName) {
    auto it = payload.parsedJson->FindMember(keyName.c_str());
    if (it == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!it->value.IsString()) {
        return absl::InvalidArgumentError(absl::StrCat(keyName, " field is not a string"));
    }
    return std::string(it->value.GetString());
}
std::variant<absl::Status, std::optional<float>> getFloatFromPayload(const ovms::HttpPayload& payload, const std::string& keyName) {
    auto it = payload.parsedJson->FindMember(keyName.c_str());
    if (it == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!it->value.IsFloat()) {
        return absl::InvalidArgumentError(absl::StrCat(keyName, " field is not a float"));
    }
    return it->value.GetFloat();
}

std::variant<absl::Status, std::optional<int64_t>> getInt64FromPayload(const ovms::HttpPayload& payload, const std::string& keyName) {
    auto it = payload.parsedJson->FindMember(keyName.c_str());
    if (it == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!it->value.IsInt64()) {
        return absl::InvalidArgumentError(absl::StrCat(keyName, " field is not a int64"));
    }
    return it->value.GetInt64();
}
std::variant<absl::Status, std::optional<int>> getIntFromPayload(const ovms::HttpPayload& payload, const std::string& keyName) {
    auto it = payload.parsedJson->FindMember(keyName.c_str());
    if (it == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!it->value.IsInt()) {
        return absl::InvalidArgumentError(absl::StrCat(keyName, " field is not a int"));
    }
    return it->value.GetInt();
}
std::variant<absl::Status, std::optional<size_t>> getSizetFromPayload(const ovms::HttpPayload& payload, const std::string& keyName) {
    auto it = payload.parsedJson->FindMember(keyName.c_str());
    if (it == payload.parsedJson->MemberEnd()) {
        return std::nullopt;
    }
    if (!it->value.IsUint64()) {
        return absl::InvalidArgumentError(absl::StrCat(keyName, " field is not a size_t"));
    }
    return it->value.GetUint64();
}

#define INSERT_IF_HAS_VALUE_RETURN_IF_FAIL(KEY, VALUE)                                                            \
    if (VALUE.has_value()) {                                                                                      \
        if (!requestOptions.insert({KEY, VALUE.value()}).second) {                                                \
            return absl::InvalidArgumentError(absl::StrCat("Key: ", KEY, " already exists in request options.")); \
        }                                                                                                         \
    }

#define SET_OPTIONAL_KEY_OR_RETURN(TYPE, FUNCTION)                     \
    SET_OR_RETURN(std::optional<TYPE>, value, FUNCTION(payload, key)); \
    INSERT_IF_HAS_VALUE_RETURN_IF_FAIL(key, value);

absl::Status ensureAcceptableAndDefaultsSetRequestOptions(ov::AnyMap& requestOptions, const ovms::ImageGenPipelineArgs& args) {
    // check if we have any unhandled parameters
    auto it = requestOptions.find("num_images_per_prompt");
    if (it != requestOptions.end()) {
        auto numImages = it->second.as<int>();
        if (numImages > args.maxNumImagesPerPrompt) {
            return absl::InvalidArgumentError(absl::StrCat("num_images_per_prompt is greater than maxNumImagesPerPrompt: ", args.maxNumImagesPerPrompt));
        }
    }
    it = requestOptions.find("num_inference_steps");
    if (it != requestOptions.end()) {
        auto numInferenceSteps = it->second.as<int>();
        if (numInferenceSteps > args.maxNumInferenceSteps) {
            return absl::InvalidArgumentError(absl::StrCat("num_inference_steps is greater than maxNumInferenceSteps: ", args.maxNumInferenceSteps));
        }
    } else {
        requestOptions.insert({"num_inference_steps", args.defaultNumInferenceSteps});
    }
    it = requestOptions.find("strength");
    if (it != requestOptions.end()) {
        auto strength = it->second.as<float>();
        if (strength > 1.0f) {
            return absl::InvalidArgumentError(absl::StrCat("strength is greater than maxStrength: ", 1));
        } else if (strength < 0.0f) {
            return absl::InvalidArgumentError(absl::StrCat("strength is less than minStrength: ", 0));
        }
    }
    return absl::OkStatus();
}

std::variant<absl::Status, ov::AnyMap> getImageGenerationRequestOptions(const ovms::HttpPayload& payload, const ovms::ImageGenPipelineArgs& args) {
    // NO -not handled yet
    // OpenAI parameters
    // https://platform.openai.com/docs/api-reference/images/create 15/05/2025
    // prompt REQUIRED DONE
    // background  REJECTED string NO optional default=auto
    // model string NO optional default=dall-e-2
    // moderation REJECTED  string NO optional default=auto
    // n NO optional default=1   ----> num_images_per_prompt
    // output_compression REJECTED  int NO optional default=100
    // output_format REJECTED  string NO optional default=png
    // quality string REJECTED  NO optional default=auto
    // response_format string NO optional default=url
    // size DONE optional default=auto
    // style string REJECTED optional default=vivid

    // GenAI parameters
    // https://github.com/openvinotoolkit/openvino.genai/blob/3c28e8279ca168ba28a79b50c62ec3b2f61d9f29/src/cpp/include/openvino/genai/image_generation/generation_config.hpp 15/05/2025
    // prompt_2 string DONE
    // prompt_3 string DONE
    // negative_prompt string DONE
    // negative_prompt_2 string DONE
    // negative_prompt_3 string DONE
    // num_images_per_prompt int DONE
    // max_sequence_length int DONE
    // height int64_t DONE
    // width int64_t DONE
    // rng_seed size_t DONE
    // num_inference_steps size_t DONE
    // strength float DONE
    // guidance_scale float DONE
    // generator ov::genaiGenerator NO
    // callback std::function<bool(size_t, size_t, ov::Tensor&) NO
    ov::AnyMap requestOptions{};
    SET_OR_RETURN(std::optional<resolution_t>, dimensionsOpt, getDimensions(payload));
    if (dimensionsOpt.has_value()) {
        auto& dimensions = dimensionsOpt.value();
        requestOptions.insert({"width", dimensions.first});
        requestOptions.insert({"height", dimensions.second});
    }

    // now get optional string parameters
    for (auto key : {"prompt_2", "prompt_3", "negative_prompt", "negative_prompt_2", "negative_prompt_3"}) {
        SET_OPTIONAL_KEY_OR_RETURN(std::string, getStringFromPayload);
    }
    // now get optional int parameters
    SET_OR_RETURN(std::optional<int>, nOpt, getIntFromPayload(payload, "n"));
    INSERT_IF_HAS_VALUE_RETURN_IF_FAIL("num_images_per_prompt", nOpt);
    for (auto key : {"num_images_per_prompt", "max_sequence_length"}) {
        SET_OPTIONAL_KEY_OR_RETURN(int, getIntFromPayload);
    }
    // now get optional float parameters
    for (auto key : {"guidance_scale", "strength"}) {
        SET_OPTIONAL_KEY_OR_RETURN(float, getFloatFromPayload);
    }
    // now get optional int64_t parameters
    for (auto key : {"width", "height"}) {
        SET_OPTIONAL_KEY_OR_RETURN(int64_t, getInt64FromPayload);
    }
    // now get optional size_t parameters
    for (auto key : {"num_inference_steps", "rng_seed"}) {
        SET_OPTIONAL_KEY_OR_RETURN(size_t, getSizetFromPayload);
    }
    // return error on unhandled parameters
    // background/moderation/output_compression/output_format/quality/style
    // TODO possibly to be handled outside since output_compresiont/format are nonGenai
    for (auto key : {"background", "moderation", "output_compression", "output_format", "quality", "style"}) {
        auto it = payload.parsedJson->FindMember(key);
        if (it != payload.parsedJson->MemberEnd()) {
            return absl::InvalidArgumentError(absl::StrCat("Unhandled parameter: ", key));
        }
    }
    // now insert default values if not already populated ?
    if (args.defaultResolution.has_value()) {
        if (requestOptions.find("height") == requestOptions.end()) {
            requestOptions.insert({"height", args.defaultResolution->first});
        }
        if (requestOptions.find("width") == requestOptions.end()) {
            requestOptions.insert({"width", args.defaultResolution->second});
        }
    }
    // now check if in httpPaylod.parsedJson we have any fields other than the ones we accept
    // accepted are: prompt, prompt_2, prompt_3, negative_prompt, negative_prompt_2, negative_prompt_3,
    // num_images_per_prompt, max_sequence_length, height, width, rng_seed, num_inference_steps,
    // strength, guidance_scale, size
    // if we have any other fields return error
    static std::set<std::string> acceptedFields{
        "prompt", "prompt_2", "prompt_3",
        "image",
        "negative_prompt", "negative_prompt_2", "negative_prompt_3",
        "size", "height", "width",
        "n", "num_images_per_prompt",
        "num_inference_steps", "rng_seed", "strength", "guidance_scale", "max_sequence_length", "model"};
    for (auto it = payload.parsedJson->MemberBegin(); it != payload.parsedJson->MemberEnd(); ++it) {
        if (acceptedFields.find(it->name.GetString()) == acceptedFields.end()) {
            return absl::InvalidArgumentError(absl::StrCat("Unhandled parameter: ", it->name.GetString()));
        }
    }
    auto status = ensureAcceptableAndDefaultsSetRequestOptions(requestOptions, args);
    if (!status.ok()) {
        return status;
    }
    // prepare log in string stream of ov::anymap
    std::ostringstream oss;
    for (const auto& [key, value] : requestOptions) {
        oss << key << ": " << value.as<std::string>() << " (type: " << value.type_info().name() << ")" << std::endl;
    }
    SPDLOG_DEBUG("Image generation request options: \n{}", oss.str());

    return std::move(requestOptions);
}

std::variant<absl::Status, ov::AnyMap> getImageEditRequestOptions(const ovms::HttpPayload& payload, const ovms::ImageGenPipelineArgs& args) {
    // NO -not handled yet
    // OpenAI parameters
    // https://platform.openai.com/docs/api-reference/images/createEdit 20/05/2025
    // prompt REQUIRED DONE
    // image string or array REQUIRED NO
    // background  REJECTED string NO optional default=auto
    // mask file NO
    // model string NO optional default=dall-e-2
    // n NO optional default=1   ----> num_images_per_prompt
    // quality string REJECTED  NO optional default=auto
    // response_format string NO optional default=url
    // size DONE optional default=1024x1024
    // user string REJECTED optional

    // GenAI parameters
    // https://github.com/openvinotoolkit/openvino.genai/blob/3c28e8279ca168ba28a79b50c62ec3b2f61d9f29/src/cpp/include/openvino/genai/image_generation/generation_config.hpp 15/05/2025
    // prompt_2 string DONE
    // prompt_3 string DONE
    // negative_prompt string DONE
    // negative_prompt_2 string DONE
    // negative_prompt_3 string DONE
    // num_images_per_prompt int DONE
    // max_sequence_length int DONE
    // height int64_t DONE
    // width int64_t DONE
    // rng_seed size_t DONE
    // num_inference_steps size_t DONE
    // strength float DONE
    // guidance_scale float DONE
    // generator ov::genaiGenerator NO
    // callback std::function<bool(size_t, size_t, ov::Tensor&) NO
    return getImageGenerationRequestOptions(payload, args);  // so far no differences in logic
}

std::variant<absl::Status, ov::AnyMap> getImageVariationRequestOptions(const ovms::HttpPayload& payload, const ovms::ImageGenPipelineArgs& args) {
    // NO -not handled yet
    // OpenAI parameters
    // https://platform.openai.com/docs/api-reference/images/createVariation 20/05/2025
    // image string or array NO
    // model string NO optional default=dall-e-2
    // n NO optional default=1   ----> num_images_per_prompt
    // quality string REJECTED  NO optional default=auto
    // response_format string NO optional default=url
    // size DONE optional default=auto
    // user string REJECTED optional default=vivid

    // GenAI parameters
    // https://github.com/openvinotoolkit/openvino.genai/blob/3c28e8279ca168ba28a79b50c62ec3b2f61d9f29/src/cpp/include/openvino/genai/image_generation/generation_config.hpp 15/05/2025
    // prompt_2 string DONE
    // prompt_3 string DONE
    // negative_prompt string DONE
    // negative_prompt_2 string DONE
    // negative_prompt_3 string DONE
    // num_images_per_prompt int DONE
    // max_sequence_length int DONE
    // height int64_t DONE
    // width int64_t DONE
    // rng_seed size_t DONE
    // num_inference_steps size_t DONE
    // strength float DONE
    // guidance_scale float DONE
    // generator ov::genaiGenerator NO
    // callback std::function<bool(size_t, size_t, ov::Tensor&) NO
    return getImageGenerationRequestOptions(payload, args);  // so far no differences in logic
}

std::variant<absl::Status, std::string> getPromptField(const HttpPayload& payload) {
    auto promptIt = payload.parsedJson->FindMember("prompt");
    if (promptIt == payload.parsedJson->MemberEnd()) {
        return absl::InvalidArgumentError("prompt field is missing in JSON body");
    }
    if (!promptIt->value.IsString()) {
        return absl::InvalidArgumentError("prompt field is not a string");
    }
    return promptIt->value.GetString();
}

std::unique_ptr<std::string> generateJSONResponseFromB64Image(const std::string& base64_image) {
    std::string json_response = absl::StrCat("{\"data\":[{\"b64_json\":\"", base64_image, "\"}]}");
    return std::make_unique<std::string>(json_response);
}
}  // namespace ovms
