//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <gtest/gtest.h>

#include "../schema.hpp"

using namespace ovms;

class TestCustomLoader : public ::testing::Test {};

TEST_F(TestCustomLoader, CustomLoaderConfigMatchingSchema) {
    const char* customloaderConfigMatchingSchema = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "library_path": "/tmp/loader/dummyloader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"loader_name":  "dummy-loader"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMatchingSchemaParsed;
    customloaderConfigMatchingSchemaParsed.Parse(customloaderConfigMatchingSchema);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderName) {
    const char* customloaderConfigMissingLoaderName = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "library_path": "dummyloader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderNameParsed;
    customloaderConfigMissingLoaderNameParsed.Parse(customloaderConfigMissingLoaderName);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLibraryPath) {
    const char* customloaderConfigMissingLibraryPath = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLibraryPathParsed;
    customloaderConfigMissingLibraryPathParsed.Parse(customloaderConfigMissingLibraryPath);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLibraryPathParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderConfig) {
    const char* customloaderConfigMissingLoaderConfig = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "library_path": "dummyloader"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderConfigParsed;
    customloaderConfigMissingLoaderConfigParsed.Parse(customloaderConfigMissingLoaderConfig);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderConfigInvalidCustomLoaderConfig) {
    const char* customloaderConfigInvalidCustomLoaderConfig = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options_invalid": {"loader_name":  "dummy-loader"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigInvalidCustomLoaderConfigParsed;
    customloaderConfigInvalidCustomLoaderConfigParsed.Parse(customloaderConfigInvalidCustomLoaderConfig);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigInvalidCustomLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderNameInCustomLoaderOptions) {
    const char* customloaderConfigMissingLoaderNameInCustomLoaderOptions = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"a": "SS"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed;
    customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed.Parse(customloaderConfigMissingLoaderNameInCustomLoaderOptions);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMultiplePropertiesInCustomLoaderOptions) {
    const char* customloaderConfigMultiplePropertiesInCustomLoaderOptions = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"loader_name": "dummy-loader", "1": "a", "2": "b", "3": "c", "4":"d", "5":"e", "6":"f"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed;
    customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed.Parse(customloaderConfigMultiplePropertiesInCustomLoaderOptions);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}
