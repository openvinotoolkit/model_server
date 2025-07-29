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

#include "src/image_conversion.hpp"

using ovms::prepareImageGenPipelineArgs;
using ovms::resolution_t;

using ::testing::_;
using ::testing::Return;

class MockedMultiPartParser final : public ovms::MultiPartParser {
public:
    MOCK_METHOD(bool, parse, (), (override));
    MOCK_METHOD(bool, hasParseError, (), (const, override));
    MOCK_METHOD(std::string, getFieldByName, (const std::string& name), (const, override));
    MOCK_METHOD(std::string_view, getFileContentByFieldName, (const std::string& name), (const, override));
};

// clang-format off
ovms::ImageGenPipelineArgs DEFAULTIMAGE_GEN_ARGS{
    std::string("/ovms/src/test/dummy"),
    {},
    ov::AnyMap(),
    {4096, 4096},  // maxResolution
    std::nullopt,  // defaultResolution
    std::nullopt,  // seed
    10,  // maxNumImagesPerPrompt
    10,  // defaultNumInferenceSteps
    10,  // maxNumInferenceSteps
    std::nullopt};  // staticReshapeSettings

TEST(Text2ImageTest, testGetDimensions) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"size":"512x513"})");
    MockedMultiPartParser multipartParser;
    
    // /create JSON
    auto dimensions = ovms::getDimensions(*payload.parsedJson);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    auto dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_TRUE(dimsOpt.has_value());
    auto dims = dimsOpt.value();
    EXPECT_EQ(dims.first, 512);
    EXPECT_EQ(dims.second, 513);

    // /edit Multipart
    ON_CALL(multipartParser, getFieldByName("size")).WillByDefault(Return("512x513"));
    dimensions = ovms::getDimensions(multipartParser);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_TRUE(dimsOpt.has_value());
    dims = dimsOpt.value();
    EXPECT_EQ(dims.first, 512);
    EXPECT_EQ(dims.second, 513);

    // /create JSON
    payload.parsedJson->Parse(R"({"size":"auto"})");
    dimensions = ovms::getDimensions(*payload.parsedJson);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions))) << std::get<absl::Status>(dimensions).message();
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());
    
    // /edit Multipart
    ON_CALL(multipartParser, getFieldByName("size")).WillByDefault(Return("auto"));
    dimensions = ovms::getDimensions(multipartParser);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());

    // /create JSON
    payload.parsedJson->Parse(R"({"other_field":"auto"})");
    dimensions = ovms::getDimensions(*payload.parsedJson);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());
    
    
    // /edit Multipart
    ON_CALL(multipartParser, getFieldByName("size")).WillByDefault(Return(""));
    ON_CALL(multipartParser, getFieldByName("other_field")).WillByDefault(Return("auto"));
    dimensions = ovms::getDimensions(multipartParser);
    ASSERT_TRUE((std::holds_alternative<std::optional<resolution_t>>(dimensions)));
    dimsOpt = std::get<std::optional<resolution_t>>(dimensions);
    ASSERT_FALSE(dimsOpt.has_value());
}
void testNegativeDimensions(const std::string& dims) {
    // /create JSON
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(dims.c_str());
    auto dimensions = ovms::getDimensions(*payload.parsedJson);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(dimensions)) << dims;
    EXPECT_EQ(std::get<absl::Status>(dimensions).code(), absl::StatusCode::kInvalidArgument) << dims;

    // /edit Multipart
    MockedMultiPartParser multipartParser;
    ON_CALL(multipartParser, getFieldByName("size")).WillByDefault(Return(payload.parsedJson->GetObject()["size"].GetString()));
    dimensions = ovms::getDimensions(multipartParser);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(dimensions)) << dims;
    EXPECT_EQ(std::get<absl::Status>(dimensions).code(), absl::StatusCode::kInvalidArgument) << dims;
}
TEST(Text2ImageTest, testGetDimensionsNegativeImproperFormat) {
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
    // /create JSON
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":"test val"})");
    auto fieldVal = ovms::getStringFromPayload(*payload.parsedJson, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<std::string>>(fieldVal));
    auto optionalString = std::get<std::optional<std::string>>(fieldVal);
    ASSERT_TRUE(optionalString.has_value());
    EXPECT_EQ(optionalString.value(), "test val");
    EXPECT_EQ(std::nullopt, std::get<std::optional<std::string>>(ovms::getStringFromPayload(*payload.parsedJson, "nonexistent_field")));

    // /edit Multipart
    MockedMultiPartParser multipartParser;
    ON_CALL(multipartParser, getFieldByName("some_field")).WillByDefault(Return("test val"));
    fieldVal = ovms::getStringFromPayload(multipartParser, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<std::string>>(fieldVal));
    optionalString = std::get<std::optional<std::string>>(fieldVal);
    ASSERT_TRUE(optionalString.has_value());
    EXPECT_EQ(optionalString.value(), "test val");
    EXPECT_EQ(std::nullopt, std::get<std::optional<std::string>>(ovms::getStringFromPayload(multipartParser, "nonexistent_field")));
}
void testNegativeString(const std::string& key, const std::string& content) {
    // /create JSON
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getStringFromPayload(*payload.parsedJson, key);
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

    // /edit Multipart
    // There is no way to fail from this operation
}
TEST(Text2ImageTest, testGetInt64FromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":1234567890123})");
    // /create JSON
    auto fieldVal = ovms::getInt64FromPayload(*payload.parsedJson, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int64_t>>(fieldVal));
    auto optionalInt64 = std::get<std::optional<int64_t>>(fieldVal);
    ASSERT_TRUE(optionalInt64.has_value());
    EXPECT_EQ(optionalInt64.value(), 1234567890123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int64_t>>(ovms::getInt64FromPayload(*payload.parsedJson, "nonexistent_field")));

    payload.parsedJson->Parse(R"({"some_field":-1234567890123})");
    fieldVal = ovms::getInt64FromPayload(*payload.parsedJson, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int64_t>>(fieldVal));
    optionalInt64 = std::get<std::optional<int64_t>>(fieldVal);
    ASSERT_TRUE(optionalInt64.has_value());
    EXPECT_EQ(optionalInt64.value(), -1234567890123);

    // /edit Multipart
    MockedMultiPartParser multipartParser;
    ON_CALL(multipartParser, getFieldByName("some_field")).WillByDefault(Return("1234567890123"));
    ON_CALL(multipartParser, getFieldByName("nonexistent_field")).WillByDefault(Return(""));
    fieldVal = ovms::getInt64FromPayload(multipartParser, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int64_t>>(fieldVal));
    optionalInt64 = std::get<std::optional<int64_t>>(fieldVal);
    ASSERT_TRUE(optionalInt64.has_value());
    EXPECT_EQ(optionalInt64.value(), 1234567890123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int64_t>>(ovms::getInt64FromPayload(multipartParser, "nonexistent_field")));

    ON_CALL(multipartParser, getFieldByName("some_field")).WillByDefault(Return("-1234567890123"));
    fieldVal = ovms::getInt64FromPayload(multipartParser, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int64_t>>(fieldVal));
    optionalInt64 = std::get<std::optional<int64_t>>(fieldVal);
    ASSERT_TRUE(optionalInt64.has_value());
    EXPECT_EQ(optionalInt64.value(), -1234567890123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int64_t>>(ovms::getInt64FromPayload(multipartParser, "nonexistent_field")));
}
// TODO need to write for nonexistent fields for all functions
void testNegativeInt64(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getInt64FromPayload(*payload.parsedJson, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal)) << content;
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument) << content;
}
void testNegativeInt64MultiPart(const std::string& key, const std::string& content) {
    MockedMultiPartParser multipartParser;
    ON_CALL(multipartParser, getFieldByName(key)).WillByDefault(Return(content));
    auto fieldVal = ovms::getInt64FromPayload(multipartParser, key);
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

    testNegativeInt64MultiPart("some_field", "    123 ");
    testNegativeInt64MultiPart("some_field", "123.5");
    testNegativeInt64MultiPart("some_field", "true");
    testNegativeInt64MultiPart("some_field", "null");
    testNegativeInt64MultiPart("some_field", "[1,2,3]");
    testNegativeInt64MultiPart("some_field", "{}");
    testNegativeInt64MultiPart("some_field", "{\"a\":1}");
    testNegativeInt64MultiPart("some_field", "123456789012345678901234567890");
    testNegativeInt64MultiPart("some_field", "-123456789012345678901234567890");
}
TEST(Text2ImageTest, testGetIntFromPayload) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(R"({"some_field":123})");
    auto fieldVal = ovms::getIntFromPayload(*payload.parsedJson, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<int>>(fieldVal));
    EXPECT_EQ(std::get<std::optional<int>>(fieldVal).value(), 123);
    EXPECT_EQ(std::nullopt, std::get<std::optional<int>>(ovms::getIntFromPayload(*payload.parsedJson, "nonexistent_field")));
}
void testNegativeInt(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getIntFromPayload(*payload.parsedJson, key);
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
    auto fieldVal = ovms::getFloatFromPayload(*payload.parsedJson, "some_field");
    ASSERT_TRUE(std::holds_alternative<std::optional<float>>(fieldVal));
    auto optionalFloat = std::get<std::optional<float>>(fieldVal);
    ASSERT_TRUE(optionalFloat.has_value());
    EXPECT_NEAR(std::get<std::optional<float>>(fieldVal).value(), 123.45, 0.0001);
    EXPECT_EQ(std::nullopt, std::get<std::optional<float>>(ovms::getFloatFromPayload(*payload.parsedJson, "nonexistent_field")));
}
void testNegativeFloat(const std::string& key, const std::string& content) {
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(content.c_str());
    auto fieldVal = ovms::getFloatFromPayload(*payload.parsedJson, key);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(fieldVal)) << content;
    EXPECT_EQ(std::get<absl::Status>(fieldVal).code(), absl::StatusCode::kInvalidArgument) << content;
}
void testNegativeFloatMultiPart(const std::string& key, const std::string& content) {
    MockedMultiPartParser multipartParser;
    ON_CALL(multipartParser, getFieldByName(key)).WillByDefault(Return(content));
    auto fieldVal = ovms::getFloatFromPayload(multipartParser, key);
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

    testNegativeFloatMultiPart("some_field", "    123.45 ");
    testNegativeFloatMultiPart("some_field", "123.45.67");
    testNegativeFloatMultiPart("some_field", "true");
    testNegativeFloatMultiPart("some_field", "null");
    testNegativeFloatMultiPart("some_field", "[1,2,3]");
    testNegativeFloatMultiPart("some_field", "{}");
    testNegativeFloatMultiPart("some_field", "{\"a\":1}");
    testNegativeFloatMultiPart("some_field", "3.40282347e+39");
    testNegativeFloatMultiPart("some_field", "-1.70141173e+39");
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
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
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
        "strength": 0.75,
        "response_format": "b64_json"
    })");
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    auto& options = std::get<ov::AnyMap>(requestOptions);
    ASSERT_EQ(options.size(), 13);
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
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "height": 1024
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(requestOptions));
    EXPECT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument);
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "size": "512x1024",
        "width": 512
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
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
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
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
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    auto& options = std::get<ov::AnyMap>(requestOptions);
    EXPECT_EQ(options.size(), 1);
    // size nor width/height not specified
    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_TRUE(std::holds_alternative<ov::AnyMap>(requestOptions));
    EXPECT_EQ(options.size(), 1);
    auto imageGenArgsWithAdminSetDefaultResolution = DEFAULTIMAGE_GEN_ARGS;
    imageGenArgsWithAdminSetDefaultResolution.defaultResolution = std::make_pair(512, 256);
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, imageGenArgsWithAdminSetDefaultResolution);
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
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "mask": "test mask"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "quality": "test quality"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "response_format": "test response format",
        "size": "512x1024"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    payload.parsedJson->Parse(R"({
        "prompt": "test prompt",
        "image": "base64_image",
        "n": 4,
        "size": "512x1024",
        "user": "test user"
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
    ASSERT_FALSE(std::holds_alternative<ov::AnyMap>(requestOptions));

    // undeclared field "nonexistend_field" : 5
    payload.parsedJson->Parse(R"({
            "prompt": "test prompt",
            "image": "base64_image",
            "n": 4,
            "size": "512x1024",
            "nonexistend_field": 5
    })");
    requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, DEFAULTIMAGE_GEN_ARGS);
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
        "model": "test model",
        "response_format": "b64_json"
    })");
    /*
        "background": "transparent",
        "mask": "base64_mask",
        "quality": "high",
        "user"
    */
   // TODO: Segfault
    auto requestOptions = ovms::getImageEditRequestOptions(*payload.multipartParser, DEFAULTIMAGE_GEN_ARGS);
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
#ifdef _WIN32
    const std::string dummyLocation = dummy_model_location;
#else
    const std::string dummyLocation = "/ovms/src/test/dummy";
#endif
    std::ostringstream oss;
    oss << R"pb(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")pb"
    <<  dummyLocation;
    oss << R"(")";
    oss << R"pb(
            device: "GPU",
            plugin_config: "{\"NUM_STREAMS\": 2}",
            max_resolution: "512x256",
            default_resolution: "256x256",
            max_num_images_per_prompt: 4,
            default_num_inference_steps: 10,
            max_num_inference_steps: 50,
          }
                    }
)pb";
    auto nodePbtxt = oss.str();
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, dummyLocation);
    ASSERT_EQ(imageGenArgs.device.size(), 1);
    ASSERT_EQ(imageGenArgs.device[0], "GPU");
    ASSERT_EQ(imageGenArgs.pluginConfig.size(), 1);
    ASSERT_EQ(imageGenArgs.pluginConfig["NUM_STREAMS"].as<int>(), 2);
    ASSERT_EQ(imageGenArgs.maxResolution, resolution_t(512, 256));
    ASSERT_TRUE(imageGenArgs.defaultResolution.has_value());
    ASSERT_EQ(imageGenArgs.defaultResolution.value(), resolution_t(256, 256));
    ASSERT_EQ(imageGenArgs.maxNumImagesPerPrompt, 4);
    ASSERT_EQ(imageGenArgs.defaultNumInferenceSteps, 10);
    ASSERT_EQ(imageGenArgs.maxNumInferenceSteps, 50);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings, std::nullopt);
}
TEST(ImageGenCalculatorOptionsTest, MultiDevices) {
#ifdef _WIN32
    const std::string dummyLocation = dummy_model_location;
#else
    const std::string dummyLocation = "/ovms/src/test/dummy";
#endif
    std::ostringstream oss;
    oss << R"pb(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")pb"
    <<  dummyLocation;
    oss << R"(")";
    oss << R"pb(
            device: "  GPU.0   MULTI:GPU.0,GPU.1   AUTO  ",
          }
                    }
)pb";
    auto nodePbtxt = oss.str();
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, dummyLocation);
    ASSERT_EQ(imageGenArgs.device.size(), 3);
    ASSERT_EQ(imageGenArgs.device[0], "GPU.0");
    ASSERT_EQ(imageGenArgs.device[1], "MULTI:GPU.0,GPU.1");
    ASSERT_EQ(imageGenArgs.device[2], "AUTO");
}
TEST(ImageGenCalculatorOptionsTest, MultiStaticResolutions) {
#ifdef _WIN32
    const std::string dummyLocation = dummy_model_location;
#else
    const std::string dummyLocation = "/ovms/src/test/dummy";
#endif
    std::ostringstream oss;
    oss << R"pb(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")pb"
    <<  dummyLocation;
    oss << R"(")";
    oss << R"pb(
            resolution: "  128x256  128x300 512x1024        1000x1000  ",
          }
                    }
)pb";
    auto nodePbtxt = oss.str();
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, dummyLocation);
    ASSERT_TRUE(imageGenArgs.staticReshapeSettings.has_value());
    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution.size(), 4);

    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[0].first, 128);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[0].second, 256);

    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[1].first, 128);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[1].second, 300);

    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[2].first, 512);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[2].second, 1024);

    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[3].first, 1000);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings.value().resolution[3].second, 1000);
}
TEST(ImageGenCalculatorOptionsTest, PositiveAllRequiredFields) {
#ifdef _WIN32
    const std::string dummyLocation = dummy_model_location;
#else
    const std::string dummyLocation = "/ovms/src/test/dummy";
#endif
    const std::string nodePbtxt =
        R"(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")" + dummyLocation + R"(",
                  }
                          }
            )";
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, dummyLocation);
    ASSERT_EQ(imageGenArgs.device.size(), 0);
    ASSERT_TRUE(imageGenArgs.pluginConfig.empty());
    ASSERT_EQ(imageGenArgs.maxResolution, resolution_t(4096, 4096));
    ASSERT_FALSE(imageGenArgs.defaultResolution.has_value());
    ASSERT_EQ(imageGenArgs.maxNumImagesPerPrompt, 10);
    ASSERT_FALSE(imageGenArgs.seed.has_value());
    ASSERT_EQ(imageGenArgs.defaultNumInferenceSteps, 50);
    ASSERT_EQ(imageGenArgs.maxNumInferenceSteps, 100);
    ASSERT_EQ(imageGenArgs.staticReshapeSettings, std::nullopt);
}
TEST(ImageGenCalculatorOptionsTest, PositiveEmptyPluginConfig) {
#ifdef _WIN32
    const std::string dummyLocation = dummy_model_location;
#else
    const std::string dummyLocation = "/ovms/src/test/dummy";
#endif
    std::ostringstream oss;
    oss << R"pb(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")pb"
    << dummyLocation;
    oss << R"(")";
    oss << R"pb(
                plugin_config: "",
    }
                        }
)pb";
    auto nodePbtxt = oss.str();
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = "";
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
    ASSERT_EQ(imageGenArgs.modelsPath, dummyLocation);
    ASSERT_EQ(imageGenArgs.device.size(), 0);
    ASSERT_TRUE(imageGenArgs.pluginConfig.empty());
}
TEST(ImageGenCalculatorOptionsTest, PositiveRelativePathToGraphPbtxt) {
#ifdef _WIN32
    const std::string cwd = R"(.\\)";
#else
    const std::string cwd = "./";
#endif
    std::ostringstream oss;
    oss << R"pb(
            name: "ImageGenExecutor"
            calculator: "ImageGenCalculator"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            node_options: {
                  [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
                    models_path: ")pb"
    << cwd;
    oss << R"(")";
    oss << R"pb(
        }
        }
    )pb";
    auto nodePbtxt = oss.str();
    SPDLOG_DEBUG("Node pbtxt: {}", nodePbtxt);
    auto node = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(nodePbtxt);
    const std::string graphPath = getGenericFullPathForSrcTest("/ovms/src/test/dummy/", true);
    auto nodeOptions = node.node_options(0);
    auto imageGenArgsOrStatus = prepareImageGenPipelineArgs(nodeOptions, graphPath);
    ASSERT_TRUE(std::holds_alternative<ImageGenPipelineArgs>(imageGenArgsOrStatus));
    auto imageGenArgs = std::get<ImageGenPipelineArgs>(imageGenArgsOrStatus);
#ifdef _WIN32
    ASSERT_EQ(getGenericFullPathForSrcTest(imageGenArgs.modelsPath),
        getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/dummy\\.\\", false))
        << imageGenArgs.modelsPath;
#else
    ASSERT_EQ(imageGenArgs.modelsPath, "/ovms/src/test/dummy/./")
        << imageGenArgs.modelsPath;
#endif
}

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
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "auto")pb",
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                guidance_scale: -1.0)pb",
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),  // reshape to guidance_scale requested, however no resolution specified
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                num_images_per_prompt: -1)pb",
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),  // reshape to batch size requested, however no resolution specified
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                max_resolution: "1024x1024")pb",  // there is no point in using max_resolution when static resolutions are defined
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                default_resolution: "256x256")pb",  // resolution is not among the ones allowed
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                device: "NPU")pb",  // resolution is not static, but device is set to NPU
            ovms::StatusCode::SHAPE_DYNAMIC_BUT_NPU_USED),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                device: "NPU")pb",  // resolution is not static, but device is set to NPU
            ovms::StatusCode::SHAPE_DYNAMIC_BUT_NPU_USED),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                device: " GPU MULTI:GPU.0,GPU.1 NPU ")pb",  // resolution is not static, but one of devices include NPU
            ovms::StatusCode::SHAPE_DYNAMIC_BUT_NPU_USED),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x10x24")pb",  // one of the resolutions on the list invalid
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 100x100 512x512")pb",  // duplicate resolutions
            ovms::StatusCode::SHAPE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                device: "GPU CPU")pb",  // only 1 or 3 devices supported
            ovms::StatusCode::DEVICE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                device: "GPU CPU GPU CPU")pb",  // only 1 or 3 devices supported
            ovms::StatusCode::DEVICE_WRONG_FORMAT),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                guidance_scale: 7.2)pb",  // resolution is not static, but guidance_scale is used
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                guidance_scale: 7.2)pb",  // resolution is not static, but guidance_scale is used
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512 1024x1024"
                num_images_per_prompt: 7)pb",  // resolution is not static, but num_images_per_prompt is used
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                num_images_per_prompt: 7)pb",  // resolution is not static, but num_images_per_prompt is used
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512"
                max_num_images_per_prompt: 7)pb",  // there is no point, there needs to be static set
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE),
        std::make_tuple(getExistingModelsPath() +
                            R"pb(
                resolution: "512x512"
                max_resolution: "512x512")pb",  // there is no point, there needs to be static set
            ovms::StatusCode::STATIC_RESOLUTION_MISUSE)));

TEST(Text2ImageTest, getImageGenerationRequestOptionsValidatedFields) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device.push_back("GPU");
    args.maxNumImagesPerPrompt = 4;
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 100;
    // now validate against args one by one
    std::unordered_map<std::string, std::string> payloadMap = {
        {"exceeded_num_images_per_prompt", R"({"prompt": "test prompt", "image": "base64_image", "n": 101, "model": "test model"})"},
        {"exceeded_num_inference_steps", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "num_inference_steps": 101})"},
        {"exceeded_strength", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "strength": 1.5})"},
        {"strength_below_0", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "strength": -0.5})"},
        {"response_format_unsupported", R"({"prompt": "test prompt", "image": "base64_image", "model": "test model", "response_format": "unsupported"})"}};
    for (auto& [key, value] : payloadMap) {
        ovms::HttpPayload payload;
        payload.parsedJson = std::make_shared<rapidjson::Document>();
        payload.parsedJson->Parse(value.c_str());
        ASSERT_FALSE(payload.parsedJson->HasParseError());
        auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, args);
        bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
        ASSERT_TRUE(holdsStatus) << "scenario: " << key << " body: " << value;
        if (holdsStatus) {
            auto status = std::get<absl::Status>(requestOptions);
            EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "scenario: " << key << " body: " << value;
        }
    }
}
TEST(Text2ImageTest, validateForStaticReshapeSettings_MatchesOneResolution) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device.push_back("NPU");
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 50;
    args.maxNumImagesPerPrompt = 10;
    args.staticReshapeSettings = ovms::StaticReshapeSettingsArgs({{512, 256}, {1024, 512}, {2048, 1024}});

    std::string value = R"({"prompt": "test prompt", "size": "1024x512", "n": 1, "model": "test model"})";
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(value.c_str());
    ASSERT_FALSE(payload.parsedJson->HasParseError());
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, args);
    bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
    ASSERT_FALSE(holdsStatus) << std::get<absl::Status>(requestOptions).ToString();
}

TEST(Text2ImageTest, validateForStaticReshapeSettings_DoesntMatchResolution) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device.push_back("NPU");
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 50;
    args.maxNumImagesPerPrompt = 10;
    args.staticReshapeSettings = ovms::StaticReshapeSettingsArgs({{512, 256}, {1024, 512}, {2048, 1024}});

    // size is 5x5, but static reshape settings requires 512x256, 1024x512 or 2048x1024
    std::string value = R"({"prompt": "test prompt", "size": "5x5", "n": 1, "model": "test model"})";
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(value.c_str());
    ASSERT_FALSE(payload.parsedJson->HasParseError());
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, args);
    bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
    ASSERT_TRUE(holdsStatus);
    ASSERT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument)
        << std::get<absl::Status>(requestOptions).message();
}

TEST(Text2ImageTest, validateForStaticReshapeSettings_NegativeStatic4ButRequested5NumImagesPerPrompt) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device.push_back("NPU");
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 50;
    args.maxNumImagesPerPrompt = 10;
    args.staticReshapeSettings = ovms::StaticReshapeSettingsArgs({{512, 256}}, 4);

    // num_images_per_prompt is 5, but static reshape settings requires 4
    std::string value = R"({"prompt": "test prompt", "size": "512x256", "n": 5, "model": "test model"})";
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(value.c_str());
    ASSERT_FALSE(payload.parsedJson->HasParseError());
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, args);
    bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
    ASSERT_TRUE(holdsStatus);
    ASSERT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument)
        << std::get<absl::Status>(requestOptions).message();
}


TEST(Text2ImageTest, validateForStaticReshapeSettings_DoesntMatchGuidanceScale) {
    ImageGenPipelineArgs args;
    args.modelsPath = "/ovms/src/test/dummy";
    args.device.push_back("NPU");
    args.defaultNumInferenceSteps = 10;
    args.maxNumInferenceSteps = 50;
    args.staticReshapeSettings = ovms::StaticReshapeSettingsArgs({{512, 256}}, std::nullopt, 7.1f);

    // Guidance scale is 7.3, but static reshape settings requires 7.1
    std::string value = R"({"prompt": "test prompt", "size": "512x256", "n": 1, "guidance_scale": 7.3, "model": "test model"})";
    ovms::HttpPayload payload;
    payload.parsedJson = std::make_shared<rapidjson::Document>();
    payload.parsedJson->Parse(value.c_str());
    ASSERT_FALSE(payload.parsedJson->HasParseError());
    auto requestOptions = ovms::getImageGenerationRequestOptions(*payload.parsedJson, args);
    bool holdsStatus = std::holds_alternative<absl::Status>(requestOptions);
    ASSERT_TRUE(holdsStatus);
    ASSERT_EQ(std::get<absl::Status>(requestOptions).code(), absl::StatusCode::kInvalidArgument)
        << std::get<absl::Status>(requestOptions).message();
}
void printNHWCOVTensor(const ov::Tensor& tensor) {
    const auto& tensorShape = tensor.get_shape();
    ASSERT_EQ(tensorShape.size(), 4);
    ASSERT_EQ(tensorShape[1], 4);
    ASSERT_EQ(tensorShape[2], 4);
    ASSERT_EQ(tensorShape[3], 3);
    auto* dataR = tensor.data();
    uint8_t* data = static_cast<uint8_t*>(dataR);
    std::ostringstream oss;
    // print shape
    oss << "Tensor shape: (";
    for (size_t i = 0; i < tensorShape.size(); ++i) {
        oss << tensorShape[i];
        if (i < tensorShape.size() - 1) {
            oss << ", ";
        }
    }
    oss << ")\n";
    for (size_t b = 0; b < tensorShape[0]; ++b) {
        oss << "image " << b << ": \n";
    for (size_t c = 0; c < tensorShape[3]; ++c) {
        oss << "\nChannel " << c << ": \n";
        for (size_t h = 0; h < tensorShape[1]; ++h) {
            for (size_t w = 0; w < tensorShape[2]; ++w) {
                size_t index = (h * tensorShape[2] + w) * tensorShape[3] + c + b * tensorShape[1] * tensorShape[2] * tensorShape[3];
                oss << std::setw(3) << static_cast<int>(data[index]) << " ";
            }
            oss << "\n";
        }
    }
    }
    SPDLOG_DEBUG("\n{}", oss.str());
}
void testResponseFromOvTensor(size_t n) {
    ov::Tensor tensor(ov::element::u8, ov::Shape{n, 4, 4, 3});
    auto* dataR = tensor.data();
    uint8_t* originalData = static_cast<uint8_t*>(dataR);
    // fill first channel with multiplies of 2
    // fill second channel with multiplies of 3
    // fill third channel with multiplies of 5
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        if (i % 3 == 0) {
            originalData[i] = static_cast<uint8_t>(((i / 3) + 1) * 2);  // first channel
        } else if (i % 3 == 1) {
            originalData[i] = static_cast<uint8_t>(((i / 3) + 1) * 3);  // second channel
        } else {
            originalData[i] = static_cast<uint8_t>(((i / 3) + 1) * 5);  // third channel
        }
    }
    printNHWCOVTensor(tensor);
    auto responseOrStatus = ovms::generateJSONResponseFromOvTensor(tensor);
    ASSERT_FALSE(std::holds_alternative<absl::Status>(responseOrStatus));
    auto response = std::move(std::get<std::unique_ptr<std::string>>(responseOrStatus));
    SPDLOG_TRACE("Response: {}", *response);
    rapidjson::Document document;
    document.Parse(response->c_str());
    ASSERT_TRUE(document.IsObject());
    ASSERT_TRUE(document.HasMember("data"));
    ASSERT_TRUE(document["data"].IsArray());
    const auto& dataArray = document["data"].GetArray();
    ASSERT_EQ(dataArray.Size(), n) << "Expected " << n << " images in response, got " << dataArray.Size();

    for (size_t i = 0; i < n; ++i) {
        const auto& imageB64PngString = dataArray[i].GetObject()["b64_json"].GetString();
        SPDLOG_TRACE("Image base64 string: {}", imageB64PngString);
        std::string decodedImage;
        ASSERT_TRUE(absl::Base64Unescape(imageB64PngString, &decodedImage)) << "Failed to decode base64 image";
        auto tensorFromImage = ovms::loadImageStbiFromMemory(decodedImage);
        printNHWCOVTensor(tensorFromImage);
        ASSERT_EQ(tensorFromImage.get_element_type(), tensor.get_element_type());
        ASSERT_EQ(tensorFromImage.get_shape().size(), 4);
        ASSERT_EQ(tensorFromImage.get_byte_size(), tensor.get_byte_size() / n);
        ASSERT_EQ(tensorFromImage.get_shape()[0], 1);
        ASSERT_EQ(tensorFromImage.get_shape()[1], tensor.get_shape()[1]);
        ASSERT_EQ(tensorFromImage.get_shape()[2], tensor.get_shape()[2]);
        ASSERT_EQ(tensorFromImage.get_shape()[3], tensor.get_shape()[3]);
        EXPECT_EQ(0, std::memcmp(tensorFromImage.data(), (char*)tensor.data() + i * (tensor.get_byte_size() / n), tensorFromImage.get_byte_size()))
            << "Data mismatch for image " << i;
    }
}
TEST(Text2ImageTest, ResponseFromOvTensorBatch1) {
    uint16_t n = 1;
    testResponseFromOvTensor(n);
}
TEST(Text2ImageTest, ResponseFromOvTensorBatch3) {
    uint16_t n = 3;
    testResponseFromOvTensor(n);
}
// TODO:
// -> test for all unhandled OpenAI fields define what to do - ignore/error imageEdit
// -> test for all unhandled OpenAI fields define what to do - ignore/error imageVariation
