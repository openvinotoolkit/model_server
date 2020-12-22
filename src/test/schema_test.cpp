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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../schema.hpp"

TEST(SchemaTest, PipelineConfigMatchingSchema) {
    const char* pipelineConfigMatchingSchema = R"(
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

    rapidjson::Document pipelineConfigMatchingSchemaParsed;
    pipelineConfigMatchingSchemaParsed.Parse(pipelineConfigMatchingSchema);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, PipelineConfigWithNegativeNodeVersion) {
    const char* PipelineConfigWithNegativeNodeVersion = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "version": -1,
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

    rapidjson::Document PipelineConfigWithNegativeNodeVersionParsed;
    PipelineConfigWithNegativeNodeVersionParsed.Parse(PipelineConfigWithNegativeNodeVersion);
    auto result = ovms::validateJsonAgainstSchema(PipelineConfigWithNegativeNodeVersionParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNameInvalidType) {
    const char* pipelineConfigNameInvalidType = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": 0,
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

    rapidjson::Document pipelineConfigNameInvalidTypeParsed;
    pipelineConfigNameInvalidTypeParsed.Parse(pipelineConfigNameInvalidType);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNameInvalidTypeParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeOutputsInvalidType) {
    const char* pipelineConfigNodeOutputsInvalidType = R"(
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
                            {"alias": "new_dummy_output"}
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

    rapidjson::Document pipelineConfigNodeOutputsInvalidTypeParsed;
    pipelineConfigNodeOutputsInvalidTypeParsed.Parse(pipelineConfigNodeOutputsInvalidType);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeOutputsInvalidTypeParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigMissingName) {
    const char* pipelineConfigMissingName = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
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

    rapidjson::Document pipelineConfigMissingNameParsed;
    pipelineConfigMissingNameParsed.Parse(pipelineConfigMissingName);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingNameParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigMissingNodes) {
    const char* PipelineConfigMissingNodes = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "outputs": [
                    {"custom_dummy_output": {"node_name": "dummyNode",
                                            "data_item": "new_dummy_output"}
                    }
                ]
            }
        ]
    })";

    rapidjson::Document pipelineConfigMatchingSchemaParsed;
    pipelineConfigMatchingSchemaParsed.Parse(PipelineConfigMissingNodes);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigMissingInputs) {
    const char* pipelineConfigMissingInputs = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
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

    rapidjson::Document pipelineConfigMissingInputsParsed;
    pipelineConfigMissingInputsParsed.Parse(pipelineConfigMissingInputs);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingInputsParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigMissingOutputs) {
    const char* pipelineConfigMissingOutputs = R"(
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
                ]
            }
        ]
    })";

    rapidjson::Document pipelineConfigMissingOutputsParsed;
    pipelineConfigMissingOutputsParsed.Parse(pipelineConfigMissingOutputs);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingOutputsParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigContainsNotAllowedKeys) {
    const char* pipelineConfigContainsNotAllowedKeys = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "illegal" : "key",
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

    rapidjson::Document pipelineConfigContainsNotAllowedKeysParsed;
    pipelineConfigContainsNotAllowedKeysParsed.Parse(pipelineConfigContainsNotAllowedKeys);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigContainsNotAllowedKeysParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeContainsNotAllowedKeys) {
    const char* pipelineConfigNodeContainsNotAllowedKeys = R"(
    {
        "model_config_list": [],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "illegal" : "key",
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

    rapidjson::Document pipelineConfigNodeContainsNotAllowedKeysParsed;
    pipelineConfigNodeContainsNotAllowedKeysParsed.Parse(pipelineConfigNodeContainsNotAllowedKeys);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeContainsNotAllowedKeysParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeTypeNotAllowed) {
    const char* pipelineConfigNodeTypeNotAllowed = R"(
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
                        "type": "illegalTypa",
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

    rapidjson::Document pipelineConfigNodeTypeNotAllowedParsed;
    pipelineConfigNodeTypeNotAllowedParsed.Parse(pipelineConfigNodeTypeNotAllowed);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeTypeNotAllowedParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeOutputsInvalid) {
    const char* pipelineConfigNodeOutputsInvalid = R"(
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
                            {a: {"node_name": "request",
                                "data_item": "custom_dummy_input"}}
                        ], 
                        "outputs": [
                            {"data_item": "a",
                            "alias": "new_dummy_output"}
                        ] 
                    }
                ],
                "outputs": [
                    {{"node_name": "dummyNode",
                     "data_item": "new_dummy_output"}
                    }
                ]
            }
        ]
    })";

    rapidjson::Document pipelineConfigNodeOutputsInvalidParsed;
    pipelineConfigNodeOutputsInvalidParsed.Parse(pipelineConfigNodeOutputsInvalid);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeOutputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeInputsInvalid) {
    const char* pipelineConfigNodeInputsInvalid = R"(
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
                            {{"node_name": "request",
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

    rapidjson::Document pipelineConfigNodeInputsInvalidParsed;
    pipelineConfigNodeInputsInvalidParsed.Parse(pipelineConfigNodeInputsInvalid);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeInputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodeInputsSourceNodeNameMissing) {
    const char* pipelineConfigNodeInputsSourceNodeNameMissing = R"(
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
                            {a:{"data_item": "custom_dummy_input"}}
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

    rapidjson::Document pipelineConfigNodeInputsSourceNodeNameMissingParsed;
    pipelineConfigNodeInputsSourceNodeNameMissingParsed.Parse(pipelineConfigNodeInputsSourceNodeNameMissing);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeInputsSourceNodeNameMissingParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigOutputsSourceNodeNameMissing) {
    const char* pipelineConfigOutputsSourceNodeNameMissing = R"(
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
                            {a: {"node_name": "request",
                                "data_item": "custom_dummy_input"}}
                        ], 
                        "outputs": [
                            {"data_item": "a",
                            "alias": "new_dummy_output"}
                        ] 
                    }
                ],
                "outputs": [
                    {"custom_dummy_output": {"data_item": "new_dummy_output"}
                    }
                ]
            }
        ]
    })";

    rapidjson::Document pipelineConfigOutputsSourceNodeNameMissingParsed;
    pipelineConfigOutputsSourceNodeNameMissingParsed.Parse(pipelineConfigOutputsSourceNodeNameMissing);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigOutputsSourceNodeNameMissingParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, PipelineConfigNodesInputsInvalid) {
    const char* pipelineConfigNodesInputsInvalid = R"(
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
                                "data_item": "custom_dummy_input"},
                             "c": {"node_name": "request",
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

    rapidjson::Document pipelineConfigNodesInputsInvalidParsed;
    pipelineConfigNodesInputsInvalidParsed.Parse(pipelineConfigNodesInputsInvalid);
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodesInputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMatchSchemaParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
    result = ovms::validateJsonAgainstSchema(mappingConfigMatchSchemaParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMissingOutputsParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigMissingInputsParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWithAdditionalObjectParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    result = ovms::validateJsonAgainstSchema(mappingConfigWithAdditionalObjectParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWhenInputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    result = ovms::validateJsonAgainstSchema(mappingConfigWhenInputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
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
    auto result = ovms::validateJsonAgainstSchema(mappingConfigWhenOutputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    result = ovms::validateJsonAgainstSchema(mappingConfigWhenOutputsIsNotAnObjectParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, parseModelMappingWhenConfigIsNotJson) {
    const char* mappingConfigIsNotAJson = "asdasdasd";

    rapidjson::Document mappingConfigIsNotAJsonParsed;
    mappingConfigIsNotAJsonParsed.Parse(mappingConfigIsNotAJson);
    auto result = ovms::validateJsonAgainstSchema(mappingConfigIsNotAJsonParsed, ovms::MODELS_MAPPING_INPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
    result = ovms::validateJsonAgainstSchema(mappingConfigIsNotAJsonParsed, ovms::MODELS_MAPPING_OUTPUTS_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

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
    auto result = ovms::validateJsonAgainstSchema(modelConfigNireqNegativeParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}
