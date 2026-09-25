//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../schema.hpp"
#include "src/status.hpp"

TEST(SchemaTest, PipelineConfigRejected) {
    const char* pipelineConfigRejected = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "model_name": "dummy",
                        "type": "DL model",
                        "inputs": [
                            {"b": {"node_name": "request",
                                "data_item": "custom_dummy_input"}}
                        ],
                        "outputs": [
                            {"data_item": "a",
                            "alias": "new_dummy_output"}
                        ]
                    }
                ],
                "outputs": [
                    {"custom_dummy_output": {"node_name": "dummyNode",
                                            "data_item": "new_dummy_output"}
                    }
                ]
            }
        ]
    })";

    rapidjson::Document pipelineConfigRejectedParsed;
    pipelineConfigRejectedParsed.Parse(pipelineConfigRejected);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigRejectedParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenJsonMatchSchema) {
    const char* mappingConfigMatchSchema = R"({
       "inputs":{
            "key":"value1",
            "key":"value2"
        },
       "outputs":{
            "key":"value3",
            "key":"value4"
        }
    })";

    rapidjson::Document mappingConfigMatchSchemaParsed;
    mappingConfigMatchSchemaParsed.Parse(mappingConfigMatchSchema);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMatchSchemaParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, parseModelMappingWhenOutputsMissingInConfig) {
    const char* mappingConfigMissingOutputs = R"({
       "inputs":{
            "key":"value1"
        }
    })";

    rapidjson::Document mappingConfigMissingOutputsParsed;
    mappingConfigMissingOutputsParsed.Parse(mappingConfigMissingOutputs);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMissingOutputsParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, parseModelMappingWhenInputsMissingInConfig) {
    const char* mappingConfigMissingInputs = R"({
       "outputs":{
            "key":"value2"
        }
    })";

    rapidjson::Document mappingConfigMissingInputsParsed;
    mappingConfigMissingInputsParsed.Parse(mappingConfigMissingInputs);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMissingInputsParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, parseModelMappingWhenAdditionalObjectInConfig) {
    const char* mappingConfigWithAdditionalObject = R"({
       "inputs":{
            "key":"value1"
        },
       "outputs":{
            "key":"value2"
        },
       "object":{
            "key":"value3"
        }
    })";

    rapidjson::Document mappingConfigWithAdditionalObjectParsed;
    mappingConfigWithAdditionalObjectParsed.Parse(mappingConfigWithAdditionalObject);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWithAdditionalObjectParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenNonStringInConfig) {
    const char* mappingConfigWithNonString1 = R"({
       "inputs":{
            "key":"value1"
        },
       "outputs":{
            "key":"value2",
            "object":{
               "key":"value3"
            }
        },
    })";
    rapidjson::Document doc1;
    doc1.Parse(mappingConfigWithNonString1);
    auto result = ovms::validateJsonAgainstSchema(doc1, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    const char* mappingConfigWithNonString2 = R"({
       "inputs":{
            "key":"value1",
            "object":{
               "key":"value3"
            }
        },
       "outputs":{
            "key":"value2"
        },
    })";
    rapidjson::Document doc2;
    doc2.Parse(mappingConfigWithNonString2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    const char* mappingConfigWithNonString3 = R"({
       "inputs":{
            "key":"value1",
            "object": 1231231
        },
       "outputs":{
            "key":"value2"
        },
    })";
    rapidjson::Document doc3;
    doc3.Parse(mappingConfigWithNonString3);
    result = ovms::validateJsonAgainstSchema(doc3, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenInputsIsNotAnObject) {
    const char* mappingConfigWhenInputsIsNotAnObject = R"({
       "inputs":["Array", "is", "not", "an", "object"],
       "outputs":{
            "key":"value2"
        }
    })";

    rapidjson::Document mappingConfigWhenInputsIsNotAnObjectParsed;
    mappingConfigWhenInputsIsNotAnObjectParsed.Parse(mappingConfigWhenInputsIsNotAnObject);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWhenInputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenOutputsIsNotAnObject) {
    const char* mappingConfigWhenOutputsIsNotAnObject = R"({
       "inputs":{
            "key":"value"
        },
       "outputs":["Array", "is", "not", "an", "object"]
    })";

    rapidjson::Document mappingConfigWhenOutputsIsNotAnObjectParsed;
    mappingConfigWhenOutputsIsNotAnObjectParsed.Parse(mappingConfigWhenOutputsIsNotAnObject);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWhenOutputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenConfigIsNotJson) {
    const char* mappingConfigIsNotAJson = "asdasdasd";

    rapidjson::Document mappingConfigIsNotAJsonParsed;
    mappingConfigIsNotAJsonParsed.Parse(mappingConfigIsNotAJson);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigIsNotAJsonParsed, ovms::MODELS_MAPPING_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

using SchemaTestCase_t = std::tuple<std::string, std::string, std::string>;
using testing::HasSubstr;
enum SchemaTestCasePart {
    NAME = 0,
    CONFIG = 1,
    ERROR_MSG = 2
};

class ConfigSchema : public ::testing::TestWithParam<SchemaTestCase_t> {};
TEST_P(ConfigSchema, DoubledFields) {
    if (std::get<SchemaTestCasePart::ERROR_MSG>(GetParam()).find("SKIPPED") != std::string::npos)
        GTEST_SKIP();
    const char* invalidConfig = std::get<SchemaTestCasePart::CONFIG>(GetParam()).c_str();
    rapidjson::Document invalidConfigDocument;
    invalidConfigDocument.Parse(invalidConfig);
    auto result = ovms::validateJsonAgainstSchema(invalidConfigDocument, ovms::MODELS_CONFIG_SCHEMA.c_str(), true);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << std::get<SchemaTestCasePart::CONFIG>(GetParam()) << "\n"
                                                      << result.string();
    EXPECT_THAT(result.string(), testing::HasSubstr(std::get<SchemaTestCasePart::ERROR_MSG>(GetParam())));
}

const uint32_t CONFIGS = 9;
std::array<SchemaTestCase_t, CONFIGS> DOUBLED_MODEL_CONFIG_KEYS_CONFIGS = {
    std::tuple(std::string("Doubled_ModelConfig"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path"
                  },
                  "config": {
                      "name": "dummy2",
                      "base_path": "dummy_path"
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_config. Keyword:maxProperties Key: #/model_config_list/0")),
    std::tuple(std::string("Doubled_ModelConfigList"), std::string(R"(
      {
          "model_config_list": [],
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path"
                  }
              }
          ]
  })"),
        std::string("SKIPPED")),
    std::tuple(std::string("Doubled_CustomLoaderConfig"), std::string(R"(
      {
          "model_config_list": [],
          "custom_loader_config_list": [
             {
                 "config":{ "loader_name": "A", "library_path":"B"},
                 "config":{ "loader_name": "A", "library_path":"B"}
             }
          ]
  })"),
        std::string("#/definitions/custom_loader_config. Keyword:maxProperties Key: #/custom_loader_config_list/0")),
    std::tuple(std::string("Doubled_ModelConfigVersionPolicyAll"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path",
                      "model_version_policy": {
                          "all": {},
                          "all": {}
                      }
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_version_policy")),
    std::tuple(std::string("Doubled_ModelConfigVersionPolicySpecific"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path",
                      "model_version_policy": {
                          "specific": {
                              "versions": [1, 2]
                          },
                          "specific": {
                              "versions": [1, 3]
                          }
                      }
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_version_policy")),
    std::tuple(std::string("Doubled_ModelConfigVersionPolicySpecificVersions"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path",
                      "model_version_policy": {
                          "specific": {
                              "versions": [1, 2],
                              "versions": [1, 2]
                          }
                      }
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_version_policy")),
    std::tuple(std::string("Doubled_ModelConfigVersionPolicyLatest"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path",
                      "model_version_policy": {
                          "latest": {
                              "num_versions":1
                          },
                          "latest": {
                              "num_versions":1
                          }
                      }
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_version_policy")),
    std::tuple(std::string("Doubled_ModelConfigVersionPolicyLatestNumVersions"), std::string(R"(
      {
          "model_config_list": [
              {
                  "config": {
                      "name": "dummy",
                      "base_path": "dummy_path",
                      "model_version_policy": {
                          "latest": {
                              "num_versions":1,
                              "num_versions":2
                          }
                      }
                  }
              }
          ]
  })"),
        std::string("#/definitions/model_version_policy")),
    std::tuple(std::string("Doubled_MonitoringMetrics"), std::string(R"(
      {
          "model_config_list": [],
          "monitoring": {
              "metrics": {
                  "enable" : true
              },
              "metrics": {
                  "enable" : true
              }
          }
  })"),
        std::string("#/properties/monitoring. Keyword:maxProperties Key: #/monitoring"))};

INSTANTIATE_TEST_SUITE_P(Doubled,
    ConfigSchema,
    ::testing::ValuesIn(DOUBLED_MODEL_CONFIG_KEYS_CONFIGS),
    [](const ::testing::TestParamInfo<ConfigSchema::ParamType>& info) {
        return std::get<SchemaTestCasePart::NAME>(info.param);
    });

TEST(SchemaTest, ModelConfigNireqNegative) {
    const char* modelConfigNireqNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "nireq": -1
            }
        }
    ]
    })";

    rapidjson::Document modelConfigNireqNegativeParsed;
    modelConfigNireqNegativeParsed.Parse(modelConfigNireqNegative);
    auto result = ovms::validateJsonAgainstSchema(modelConfigNireqNegativeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelConfigSequenceMaxNumberNegative) {
    // stateful models were removed, so max_sequence_number is no longer a valid property, even with a valid value
    const char* modelConfigSeqNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "max_sequence_number": 1
            }
        }
    ]
    })";

    rapidjson::Document modelConfigSeqNegativeDoc;
    modelConfigSeqNegativeDoc.Parse(modelConfigSeqNegative);
    auto result = ovms::validateJsonAgainstSchema(modelConfigSeqNegativeDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelConfigTimeoutNegative) {
    // stateful models were removed, so sequence_timeout_seconds is no longer a valid property, even with a valid value
    const char* modelConfigTimeoutNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "sequence_timeout_seconds": 1
            }
        }
    ]
    })";

    rapidjson::Document modelConfigSeqNegativeDoc;
    modelConfigSeqNegativeDoc.Parse(modelConfigTimeoutNegative);
    auto result = ovms::validateJsonAgainstSchema(modelConfigSeqNegativeDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelConfigVersionPolicyAll) {
    const char* modelConfigVersionPolicyAll1 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"all": {}}
                }
            }
        ]
    })";
    rapidjson::Document doc;
    doc.Parse(modelConfigVersionPolicyAll1);
    auto result = ovms::validateJsonAgainstSchema(doc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK) << modelConfigVersionPolicyAll1;
    const char* modelConfigVersionPolicyAll2 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"all": 3}
                }
            }
        ]
    })";
    rapidjson::Document doc2;
    doc2.Parse(modelConfigVersionPolicyAll2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << modelConfigVersionPolicyAll2;
    const char* modelConfigVersionPolicyAll3 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"all": {"a":3}}
                }
            }
        ]
    })";
    rapidjson::Document doc3;
    doc2.Parse(modelConfigVersionPolicyAll3);
    result = ovms::validateJsonAgainstSchema(doc3, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << modelConfigVersionPolicyAll3;
}
TEST(SchemaTest, ModelConfigVersionPolicyLatest) {
    const char* modelConfigVersionPolicyLatest1 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"latest": {"num_versions": 2}}
                }
            }
        ]
    })";
    rapidjson::Document doc;
    doc.Parse(modelConfigVersionPolicyLatest1);
    auto result = ovms::validateJsonAgainstSchema(doc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
    const char* modelConfigVersionPolicyLatest2 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"latest": {"num_versions": [2,3]}}
                }
            }
        ]
    })";
    rapidjson::Document doc2;
    doc2.Parse(modelConfigVersionPolicyLatest2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    const char* modelConfigVersionPolicyLatest3 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"latest": {"num_versions": {2}}}
                }
            }
        ]
    })";
    rapidjson::Document doc3;
    doc3.Parse(modelConfigVersionPolicyLatest3);
    result = ovms::validateJsonAgainstSchema(doc3, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}
TEST(SchemaTest, ModelConfigVersionPolicySpecific) {
    const char* modelConfigVersionPolicySpecific1 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"specific": {"versions": [1, 2]}}
                }
            }
        ]
    })";
    rapidjson::Document doc1;
    doc1.Parse(modelConfigVersionPolicySpecific1);
    auto result = ovms::validateJsonAgainstSchema(doc1, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
    const char* modelConfigVersionPolicySpecific2 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"specific": {"versions": 3}}
                }
            }
        ]
    })";
    rapidjson::Document doc2;
    doc2.Parse(modelConfigVersionPolicySpecific2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    const char* modelConfigVersionPolicySpecific3 = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "model_version_policy": {"specific": {"versions": [1, "2"]}}
                }
            }
        ]
    })";
    rapidjson::Document doc3;
    doc3.Parse(modelConfigVersionPolicySpecific3);
    result = ovms::validateJsonAgainstSchema(doc3, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelConfigPluginConfigPositive) {
    const char* modelConfigTimeoutNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "plugin_config": {"A":"B", "C":2, "D":2.5, "E":true, "F":false}
            }
        }
    ]
    })";
    rapidjson::Document modelConfigSeqNegativeDoc;
    modelConfigSeqNegativeDoc.Parse(modelConfigTimeoutNegative);
    auto result = ovms::validateJsonAgainstSchema(modelConfigSeqNegativeDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, ModelConfigPluginConfigLayoutShapeNegative) {
    const char* config1 = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "shape": {"A":"B", "C":"NCHW", "D":{}},
                "layout": {"A":"B", "C":"NCHW", "D":"NHWC"}
            }
        }
    ]
    })";

    rapidjson::Document doc1;
    doc1.Parse(config1);
    auto result = ovms::validateJsonAgainstSchema(doc1, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << config1;
    const char* config2 = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "shape": ["NHWC", "NCHW"],
                "layout": {"A":"B", "C":"NCHW", "D":"NHWC"}
            }
        }
    ]
    })";

    rapidjson::Document doc2;
    doc2.Parse(config2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << config2;
    const char* config3 = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "shape": {"A":"B", "C":"NCHW", "D":"NHWC:NHWC"},
                "layout": {"A":"B", "C":"NCHW", "D":[1,2,3]}
            }
        }
    ]
    })";

    rapidjson::Document doc3;
    doc3.Parse(config3);
    result = ovms::validateJsonAgainstSchema(doc3, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID) << config3;
}
TEST(SchemaTest, ModelConfigPluginConfigNegative) {
    const char* modelConfigNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "plugin_config": {"A":[12,2]}
            }
        }
    ]
    })";

    rapidjson::Document doc;
    doc.Parse(modelConfigNegative);
    auto result = ovms::validateJsonAgainstSchema(doc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    const char* modelConfigNegative2 = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "plugin_config": {"A":{"s":"f"}}
            }
        }
    ]
    })";

    rapidjson::Document doc2;
    doc2.Parse(modelConfigNegative2);
    result = ovms::validateJsonAgainstSchema(doc2, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeLibraryConfigRejected) {
    const char* customNodeLibraryConfig = R"(
    {
        "model_config_list": [],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ]
    })";

    rapidjson::Document customNodeLibraryConfigParsed;
    customNodeLibraryConfigParsed.Parse(customNodeLibraryConfig);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

#if (MEDIAPIPE_DISABLE == 0)
TEST(SchemaTest, MediapipeConfigPositive) {
    const char* mediapipeConfigPositive = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_model",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path_base"
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigPositive);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, MediapipeConfigInModelConfigPositive) {
    const char* mediapipeConfigPositive = R"(
    {
        "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "dummy_path",
                "graph_path": "dummy_path.pbtxt"
            }
        }
    ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigPositive);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, MediapipeConfigIdleUnloadTimeoutPositive) {
    const char* mediapipeConfigPositive = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_model",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path_base",
            "idle_unload_timeout_seconds": 300
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigPositive);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, MediapipeConfigIdleUnloadTimeoutNegativeValueRejected) {
    const char* mediapipeConfigNegative = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_model",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path_base",
            "idle_unload_timeout_seconds": -5
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigNegative);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, MediapipeConfigIdleUnloadTimeoutWrongTypeRejected) {
    const char* mediapipeConfigNegative = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_model",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path_base",
            "idle_unload_timeout_seconds": "notAnInteger"
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigNegative);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}
#endif

TEST(SchemaTest, ModelConfigGroupNameValidString) {
    const char* config = R"(
    {
        "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "group_name": "rag"
            }
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(config);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, ModelConfigGroupNameInvalidType) {
    const char* config = R"(
    {
        "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "group_name": 123
            }
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(config);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

#if (MEDIAPIPE_DISABLE == 0)
TEST(SchemaTest, MediapipeConfigGroupNameValidString) {
    const char* config = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_graph",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path",
            "group_name": "llm_group"
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(config);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, MediapipeConfigGroupNameInvalidType) {
    const char* config = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_graph",
            "graph_path": "graph.pbtxt",
            "base_path": "dummy_path",
            "group_name": 42
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(config);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}
#endif

TEST(SchemaTest, MediapipeConfigNegativeAdditionalMediapipeConfigField) {
    const char* mediapipeConfigNegative = R"(
    {
        "model_config_list": [],
        "mediapipe_config_list": [
        {
            "name": "dummy_model",
            "graph_path": "dummy_path",
            "someField": "ovms_rules"
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigNegative);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}
