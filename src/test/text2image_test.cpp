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
#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rest_parser.hpp"
#include "../http_payload.hpp"
#pragma warning(push)
#pragma warning(disable : 6001)
#include "absl/strings/escaping.h"
#pragma warning(pop)
#include "test_utils.hpp"

#include "src/image_gen/imagegenutils.hpp"
#include "src/image_gen/imagegen_init.hpp"

using ovms::prepareImageGenPipelineArgs;
using ovms::resolution_t;

ovms::ImageGenPipelineArgs DEFAULTIMAGE_GEN_ARGS{
    std::string("/ovms/src/test/dummy"),
    std::nullopt,
    ov::AnyMap(),
    {4096, 4096},  // maxResolution
    std::nullopt,  // defaultResolution
    std::nullopt,  // seed
    10,
    10,
    100};

TEST(Text2ImageTest, testGetDimensions) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"size":"512x513"})");
    auto dimensions = ovms::getDimensions(payload);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    auto dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_TRUE(dimsOpt.has_value());
    auto dims = dimsOpt.value();
    EXPECT_EQ(dims.first, 512);
    EXPECT_EQ(dims.second, 513);

    payload.parsedJson->Parse(R"({"size":"auto"})");
    dimensions = ovms::getDimensions(payload);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions))) << std::get<absl::Status>(dimensions).message();
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());

    payload.parsedJson->Parse(R"({"other_field":"auto"})");
    dimensions = ovms::getDimensions(payload);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());
}
void testNegativeDimensions(const std::string& dims) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(dims.c_str());
    auto dimensions = ovms::getDimensions(payload);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(dimensions)) << dims;
    EXPECT_EQ(std::get<absl::Status>(dimensions).code(), absl::StatusCode::kInvalidArgument) << dims;
}
TEST(Text2ImageTest, testGetDimensionsNegativeImproperFormat) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    std::variant<absl::Status, std::pair<int64_t, int64_t>> dimensions;

    testNegativeDimensions(R"({"size":"51:512"})");
    testNegativeDimensions(R"({"size":"51:512"})");
    testNegativeDimensions(R"({"size":"512_51x"})");
    testNegativeDimensions(R"({"size":"51x512x"})");
    testNegativeDimensions(R"({"size":"-51x52"})");
    testNegativeDimensions(R"({"size":"51x-52"})");
    testNegativeDimensions(R"({"size":"0x52"})");
    testNegativeDimensions(R"({"size":"51x0"})");
    testNegativeDimensions(R"({"size":"abcx512"})");
    testNegativeDimensions(R"({"size":"5151xabc"})");
    // max int64_t 9223372036854775807
    // min int64_t -9223372036854775808
    testNegativeDimensions(R"({"size":"9223372036854775808x1"})");
    testNegativeDimensions(R"({"size":"1x9223372036854775808"})");
    SPDLOG_DEBUG("Maximum int64_t value: {}", std::numeric_limits<int64_t>::max());
    SPDLOG_DEBUG("Minimum int64_t value: {}", std::numeric_limits<int64_t>::min());
}
TEST(Text2ImageTest, testGetStringFromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":"test val"})");
    auto fieldVal = ovms::getStringFromPayload(payload, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<std::string>>(fieldVal));
    auto optionalString = std::get<std::optional<std::string>>(fieldVal);
    ASSERT_TRUE(optionalString.has_value());
    EXPECT_EQ(optionalString.value(), "test val");
    EXPECT_EQ(std::nullopt, std::get<std::optional<std::string>>(ovms::getStringFromPayload(payload, "nonexistent_field")));
}
void testNegativeString(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getStringFromPayload(payload, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal));
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument);
}
TEST(Text2ImageTest, testGetStringFromPayloadNegative) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    std::variant<absl::Status, std::string> field;
    testNegativeString("prompt", R"({"prompt":123})");
    testNegativeString("prompt", R"({"prompt":true})");
    testNegativeString("prompt", R"({"prompt":null})");
    testNegativeString("prompt", R"({"prompt":123.45})");
    testNegativeString("prompt", R"({"prompt":[1,2,3]})");
    testNegativeString("prompt", R"({"prompt":{}})");
    testNegativeString("prompt", R"({"prompt":{"a":1}})");
}
TEST(Text2ImageTest, testGetInt64FromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":1234567890123})");
    auto fieldVal = ovms::getInt64FromPayload(payload, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int64_t>>(fieldVal));
    auto optionalInt64 = std::get<std::optional<int64_t>>(fieldVal);
    ASSERT_TRUE(optionalInt64.has_value());
    EXPECT_EQ(optionalInt64.value(), 1234567890123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int64_t>>(ovms::getInt64FromPayload(payload, "nonexistent_field")));
}
// TODO need to write for nonexistent fields for all functions
void testNegativeInt64(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getInt64FromPayload(payload, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal)) << content;
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument) << content;
}
TEST(Text2ImageTest, testGetInt64FromPayloadNegative) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    std::variant<absl::Status, int64_t> field;
    testNegativeInt64("some_field", R"({"some_field":"123"})");
    testNegativeInt64("some_field", R"({"some_field":true})");
    testNegativeInt64("some_field", R"({"some_field":null})");
    testNegativeInt64("some_field", R"({"some_field":123.45})");
    testNegativeInt64("some_field", R"({"some_field":[1,2,3]})");
    testNegativeInt64("some_field", R"({"some_field":{}})");
    testNegativeInt64("some_field", R"({"some_field":{"a":1}})");
    testNegativeInt64("some_field", R"({"some_field":123456789012345678901234567890})");
    testNegativeInt64("some_field", R"({"some_field":-123456789012345678901234567890})");
}
TEST(Text2ImageTest, testGetIntFromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":123})");
    auto fieldVal = ovms::getIntFromPayload(payload, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int>>(fieldVal));
    EXPECT_EQ(std::get<std::optional<int>>(fieldVal).value(), 123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int>>(ovms::getIntFromPayload(payload, "nonexistent_field")));
}
void testNegativeInt(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getIntFromPayload(payload, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal)) << content;
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument) << content;
}
TEST(Text2ImageTest, testGetIntFromPayloadNegative) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    std::variant<absl::Status, int> field;
    testNegativeInt("some_field", R"({"some_field":"123"})");
    testNegativeInt("some_field", R"({"some_field":true})");
    testNegativeInt("some_field", R"({"some_field":null})");
    testNegativeInt("some_field", R"({"some_field":123.45})");
    testNegativeInt("some_field", R"({"some_field":[1,2,3]})");
    testNegativeInt("some_field", R"({"some_field":{}})");
    testNegativeInt("some_field", R"({"some_field":{"a":1}})");
    testNegativeInt("some_field", R"({"some_field":123456789012345678901234567890})");
    testNegativeInt("some_field", R"({"some_field":-123456789012345678901234567890})");
}
TEST(Text2ImageTest, testGetFloatFromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":123.45})");
    auto fieldVal = ovms::getFloatFromPayload(payload, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<float>>(fieldVal));
    auto optionalFloat = std::get<std::optional<float>>(fieldVal);
    ASSERT_TRUE(optionalFloat.has_value());
    EXPECT_NEAR(std::get<std::optional<float>>(fieldVal).value(), 123.45, 0.0001);
    EXPECT_EQ(std::nullopt, std::get<std::optional<float>>(ovms::getFloatFromPayload(payload, "nonexistent_field")));
}
void testNegativeFloat(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getFloatFromPayload(payload, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal)) << content;
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument) << content;
}
TEST(Text2ImageTest, testGetFloatFromPayloadNegative) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    std::variant<absl::Status, float> field;
    testNegativeFloat("some_field", R"({"some_field":"123"})");
    testNegativeFloat("some_field", R"({"some_field":true})");
    testNegativeFloat("some_field", R"({"some_field":null})");
    testNegativeFloat("some_field", R"({"some_field":123})");
    testNegativeFloat("some_field", R"({"some_field":[1,2,3]})");
    testNegativeFloat("some_field", R"({"some_field":{}})");
    testNegativeFloat("some_field", R"({"some_field":{"a":1}})");
    testNegativeFloat("some_field", R"({"some_field":3.40282347e+39})");
    testNegativeFloat("some_field", R"({"some_field":-1.70141173e+39})");
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsAllHandledOpenAIFields) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    // write request with prompt, size 512x1024, n=4 model=test_model
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "n": 4,
        "model":"test model"
    })");
    /*
        "background": "test background",
        "moderation": "test moderation",
        "output_compression": "test output compression",
        "output_format": "test output format",
        "quality": "test quality",
        "style": "test style"        
    */
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    auto& options = std::get<ov::AnyMap>(requestOptions);
    EXPECT_EQ(options.size(), 4);
    for (auto& [key, value] : options) {
        SPDLOG_DEBUG("key: {}, value: {}", key, value.as<std::string>());
    }
    EXPECT_EQ(options.at("width").as<int64_t>(), 512);
    EXPECT_EQ(options.at("height").as<int64_t>(), 1024);
    EXPECT_EQ(options.at("num_images_per_prompt").as<int>(), 4);
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsAllHandledGenAIFields) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "prompt_2": "test prompt 2",
        "prompt_3": "test prompt 3",
        "negative_prompt": "test negative prompt",
        "negative_prompt_2": "test negative prompt 2",
        "negative_prompt_3": "test negative prompt 3",
        "rng_seed": 123456789,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 1024,
        "num_images_per_prompt": 4,
        "num_inference_steps": 7,
        "max_sequence_length": 256,
        "strength": 0.75
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    auto& options = std::get<ov::AnyMap>(requestOptions);
    EXPECT_EQ(options.size(), 13);
    for (auto& [key, value] : options) {
        SPDLOG_DEBUG("key: {}, value: {}", key, value.as<std::string>());
    }
    EXPECT_EQ(options.at("prompt_2").as<std::string>(), "test prompt 2");
    EXPECT_EQ(options.at("prompt_3").as<std::string>(), "test prompt 3");
    EXPECT_EQ(options.at("negative_prompt").as<std::string>(), "test negative prompt");

    EXPECT_EQ(options.at("negative_prompt_2").as<std::string>(), "test negative prompt 2");
    EXPECT_EQ(options.at("negative_prompt_3").as<std::string>(), "test negative prompt 3");
    EXPECT_EQ(options.at("rng_seed").as<size_t>(), 123456789);
    EXPECT_EQ(options.at("guidance_scale").as<float>(), 7.5);
    EXPECT_EQ(options.at("strength").as<float>(), 0.75);
    EXPECT_EQ(options.at("max_sequence_length").as<int>(), 256);
    EXPECT_EQ(options.at("width").as<int64_t>(), 512);
    EXPECT_EQ(options.at("height").as<int64_t>(), 1024);
    EXPECT_EQ(options.at("num_images_per_prompt"), 4);
    EXPECT_EQ(options.at("num_inference_steps").as<size_t>(), 7);
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsNegativeSizeAndWidthHeightTogether) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "width": 512,
        "height": 1024
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "height": 1024
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "width": 512
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsNegativeNAndNumImagesPerPromptTogether) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "n": 4,
        "num_images_per_prompt": 4
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsDefaultSizeBehavior) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "auto"
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    auto& options = std::get<ov::AnyMap>(requestOptions);
    EXPECT_EQ(options.size(), 1);
    // size nor width/height not specified
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    EXPECT_EQ(options.size(), 1);
    auto imageGenArgsWithAdminSetDefaultResolution = DEFAULTIMAGE_GEN_ARGS;
    imageGenArgsWithAdminSetDefaultResolution.defaultResolution = std::make_pair(512, 256);
    requestOptions = ovms::getImageGenerationRequestOptions(payload, imageGenArgsWithAdminSetDefaultResolution);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
    })");
    EXPECT_EQ(options.size(), 3);
    EXPECT_EQ(options.at("height").as<int64_t>(), 256);
    EXPECT_EQ(options.at("width").as<int64_t>(), 512);
}
TEST(Text2ImageTest, getImageGenerationRequestOptionsRejectedFields) {
    // OpenAI fields background, mask, quality, response_format, user
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "background": "test background"
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "mask": "test mask"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "quality": "test quality"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "response_format": "test response format"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "user": "test user"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    // undeclared field "nonexistend_field" : 5
    payload.parsedJson->Parse(R"({
            "prompt": "test prompt",
            "image": "base64_image",
            "n": 4,
            "size": "512x1024",
            "nonexistend_field": 5
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));
}

TEST(Image2ImageTest, getImageEditGenerationRequestOptionsAllHandledOpenAIFields) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "model": "test model"
    })");
    /*
        "background": "transparent",
        "mask": "base64_mask",
        "quality": "high",
        "response_format": "url",
        "user"
    */
    auto requestOptions = ovms::getImageEditRequestOptions(payload, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions)) << std::get<absl::Status>(requestOptions).message();
    auto& options = std::get<ov::AnyMap>(requestOptions);
    EXPECT_EQ(options.size(), 4);
    for (auto& [key, value] : options) {
        SPDLOG_DEBUG("key: {}, value: {}", key, value.as<std::string>());
    }
    EXPECT_EQ(options.at("width").as<int64_t>(), 512);
    EXPECT_EQ(options.at("height").as<int64_t>(), 1024);
    EXPECT_EQ(options.at("num_images_per_prompt").as<int>(), 4);
}

using mediapipe::CalculatorContract;
using mediapipe::CalculatorGraphConfig;
using ovms::ImageGenPipelineArgs;
TEST(ImageGenCalculatorOptionsTest, PositiveAllfields) {
    auto node =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                name: "ImageGenExecutor"
                calculator: "ImageGenCalculator"
                input_stream: "HTTP_REQUEST_PAYLOAD:input"
                input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
                output_stream: "HTTP_RESPONSE_PAYLOAD:output"
                node_options: {
                      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                        models_path: "/ovms/src/test/dummy",
                        device: "GPU",
                        plugin_config: "{\"NUM_STREAMS\": 2}",
                        max_resolution: "512x256",
                        default_resolution: "256x256",
                        max_num_images_per_prompt: 4,
                        default_num_inference_steps: 10,
                        max_num_inference_steps: 50,
                      }
                }
            )pb");
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, "/ovms/src/test/dummy");
    ASSERT_TRUE(imageGenArgs.device.has_value());
    ASSERT_EQ(imageGenArgs.device.value(), "GPU");
    ASSERT_EQ(imageGenArgs.pluginConfig.size(), 1);
    ASSERT_EQ(imageGenArgs.pluginConfig["NUM_STREAMS"].as<int>(), 2);
    ASSERT_EQ(imageGenArgs.maxResolution, resolution_t(512, 256));
    ASSERT_TRUE(imageGenArgs.defaultResolution.has_value());
    ASSERT_EQ(imageGenArgs.defaultResolution.value(), resolution_t(256, 256));
    ASSERT_EQ(imageGenArgs.maxNumImagesPerPrompt, 4);
    // ASSERT_TRUE(imageGenArgs.seed.has_value());
    // ASSERT_EQ(imageGenArgs.seed.value(), 123456789);
    ASSERT_EQ(imageGenArgs.defaultNumInferenceSteps, 10);
    ASSERT_EQ(imageGenArgs.maxNumInferenceSteps, 50);
}
TEST(ImageGenCalculatorOptionsTest, PositiveAllRequiredFields) {
    auto node =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                name: "ImageGenExecutor"
                calculator: "ImageGenCalculator"
                input_stream: "HTTP_REQUEST_PAYLOAD:input"
                input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
                output_stream: "HTTP_RESPONSE_PAYLOAD:output"
                node_options: {
                      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                        models_path: "/ovms/src/test/dummy",
                      }
                }
            )pb");
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, "/ovms/src/test/dummy");
    ASSERT_FALSE(imageGenArgs.device.has_value());
    ASSERT_TRUE(imageGenArgs.pluginConfig.empty());
    ASSERT_EQ(imageGenArgs.maxResolution, resolution_t(4096, 4096));
    ASSERT_FALSE(imageGenArgs.defaultResolution.has_value());
    ASSERT_EQ(imageGenArgs.maxNumImagesPerPrompt, 10);
    ASSERT_FALSE(imageGenArgs.seed.has_value());
    ASSERT_EQ(imageGenArgs.defaultNumInferenceSteps, 10);
    ASSERT_EQ(imageGenArgs.maxNumInferenceSteps, 100);
}
TEST(ImageGenCalculatorOptionsTest, PositiveEmptyPluginConfig) {
    auto node =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                name: "ImageGenExecutor"
                calculator: "ImageGenCalculator"
                input_stream: "HTTP_REQUEST_PAYLOAD:input"
                input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
                output_stream: "HTTP_RESPONSE_PAYLOAD:output"
                node_options: {
                      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                        models_path: "/ovms/src/test/dummy",
                        plugin_config: "",
                      }
                }
            )pb");
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, "/ovms/src/test/dummy");
    ASSERT_FALSE(imageGenArgs.device.has_value());
    ASSERT_TRUE(imageGenArgs.pluginConfig.empty());
}
TEST(ImageGenCalculatorOptionsTest, PositiveRelativePathToGraphPbtxt) {
    auto node =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                name: "ImageGenExecutor"
                calculator: "ImageGenCalculator"
                input_stream: "HTTP_REQUEST_PAYLOAD:input"
                input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
                output_stream: "HTTP_RESPONSE_PAYLOAD:output"
                node_options: {
                      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                        models_path: "./"
                      }
                }
            )pb");
    const std::string graphPath = "/ovms/src";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, "/ovms/src/./");
}
// write parametrized test for ImageGenCalculatorOptions with negative cases
// parameter will be a tuple of <node_options, ovms::StatusCode, expected_error_message>

class ImageGenCalculatorOptionsNegative : public ::testing::TestWithParam<std::tuple<std::string, ovms::StatusCode>> {
};

TEST_P(ImageGenCalculatorOptionsNegative, NegativeCases) {
    // param is tuple of string, status code & string

    auto nodeOptionsText = std::get<0>(GetParam());
    auto expectedCode = std::get<1>(GetParam());
    auto nodeString = R"pb(
                          name: "ImageGenExecutor"
                          calculator: "ImageGenCalculator"
                          input_stream: "HTTP_REQUEST_PAYLOAD:input"
                          input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
                          output_stream: "HTTP_RESPONSE_PAYLOAD:output"
                          node_options: {
                                [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                      )pb" + nodeOptionsText + R"pb(
        }
        }
        )pb";
    SPDLOG_DEBUG("Node string: {}", nodeString);
    const std::string graphPath = "";
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodeString);
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ovms::Status>(imageGenArgsOrStatus));
    auto status = std::get<ovms::Status>(imageGenArgsOrStatus);
    EXPECT_EQ(status.getCode(), expectedCode) << status.string();
}

const std::string& getExistingModelsPath() {
    static const std::string EXISTING_MODELS_PATH = R"pb(models_path: ")pb" + dummy_model_location + "\"";
    return EXISTING_MODELS_PATH;
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    ImageGenCalculatorOptionsNegative,
    ::testing::Values(
        // rewrite those commented make_pair to tuples:
        std::make_tuple(R"pb(models_path: "/nonexistentpath")pb", ovms::StatusCode::PATH_INVALID),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                default_resolution: "4097x256")pb",
            ovms::StatusCode::DEFAULT_EXCEEDS_MAXIMUM_ALLOWED_RESOLUTION),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                max_resolution: "(4096x4096)")pb",
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                max_resolution: "auto")pb",
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                plugin_config: "NUM_STREAMS=2")pb",
            ovms::StatusCode::PLUGIN_CONFIG_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                max_resolution: "high_resolution")pb",
            ovms::StatusCode::SHAPE_WRONG_FORMAT)));

TEST(Text2ImageTest, getImageGenerationRequestOptionsValidatedFields) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device = "GPU";
    args.maxNumImagesPerPrompt = 4;
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 100;
    // now validate against args one by one
    std::unordered_map<std::string, std::string> payloadMap = {
        {"exceeded_num_images_per_prompt", R"({"prompt": "test prompt", "image": "base64_image", "n": 101, "model": "test model"})"},
        {"exceeded_num_inference_steps", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "num_inference_steps": 101})"},
        {"exceeded_strength", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "strength": 1.5})"},
        {"strength_below_0", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "strength": -0.5})"}};
    for (auto& [key, value] : payloadMap) {
        ovms::HttpPayload payload;
        payload.parsedJson = std::make_shared<rapidjson::Document>();
        payload.parsedJson->Parse(value.c_str());
        auto requestOptions = ovms::getImageGenerationRequestOptions(payload, args);
        bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
        ASSERT_TRUE(holdsStatus) << "scenario: " << key << " body: " << value;
        if (holdsStatus) {
            auto status = std::get<absl::Status>(requestOptions);
            EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "scenario: " << key << " body: " << value;
        }
    }
}
// TODO:
// -> test for all unhandled OpenAI fields define what to do - ignore/error text2image
// -> test for all unhandled OpenAI fields define what to do - ignore/error imageEdit
// -> test for all unhandled OpenAI fields define what to do - ignore/error imageVariation
