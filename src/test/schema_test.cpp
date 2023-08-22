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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(PipelineConfigWithNegativeNodeVersionParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNameInvalidTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeOutputsInvalidTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingInputsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigMissingOutputsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigContainsNotAllowedKeysParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeContainsNotAllowedKeysParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeTypeNotAllowedParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeOutputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeInputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodeInputsSourceNodeNameMissingParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigOutputsSourceNodeNameMissingParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(pipelineConfigNodesInputsInvalidParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    const char* modelConfigSeqNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "max_sequence_number": -1
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
    const char* modelConfigTimeoutNegative = R"(
    {
    "model_config_list": [
        {
            "config": {
                "name": "dummy_model",
                "base_path": "dummy_path",
                "sequence_timeout_seconds": -1
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
                "plugin_config": {"A":"B", "C":2, "D":2.5}
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

TEST(SchemaTest, CustomNodeLibraryConfigMatchingSchema) {
    const char* customNodeLibraryConfig = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": "dummy_library",
                        "type": "custom",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
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
            },
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode2",
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

    rapidjson::Document customNodeLibraryConfigParsed;
    customNodeLibraryConfigParsed.Parse(customNodeLibraryConfig);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, CustomNodeLibraryConfigMissingLibraryName) {
    const char* customNodeLibraryConfigMissingLibraryName = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "base_path": "dummy_path"
            }
        ]
    })";

    rapidjson::Document customNodeLibraryConfigMissingLibraryNameParsed;
    customNodeLibraryConfigMissingLibraryNameParsed.Parse(customNodeLibraryConfigMissingLibraryName);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigMissingLibraryNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeLibraryConfigMissingBasePath) {
    const char* customNodeLibraryConfigMissingBasePath = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library"
            }
        ]
    })";

    rapidjson::Document customNodeLibraryConfigMissingBasePathParsed;
    customNodeLibraryConfigMissingBasePathParsed.Parse(customNodeLibraryConfigMissingBasePath);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigMissingBasePathParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeLibraryConfigInvalidNameType) {
    const char* customNodeLibraryConfigInvalidNameType = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": 2,
                "base_path": "dummy_path"
            }
        ]
    })";

    rapidjson::Document customNodeLibraryConfigInvalidNameTypeParsed;
    customNodeLibraryConfigInvalidNameTypeParsed.Parse(customNodeLibraryConfigInvalidNameType);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigInvalidNameTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeLibraryConfigInvalidBasePathType) {
    const char* customNodeLibraryConfigInvalidBasePathType = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": 2
            }
        ]
    })";

    rapidjson::Document customNodeLibraryConfigInvalidBasePathTypeParsed;
    customNodeLibraryConfigInvalidBasePathTypeParsed.Parse(customNodeLibraryConfigInvalidBasePathType);
    auto result = ovms::validateJsonAgainstSchema(customNodeLibraryConfigInvalidBasePathTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeConfigInvalidLibraryNameType) {
    const char* customNodeConfigInvalidLibraryNameType = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": 2,
                        "type": "custom",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
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

    rapidjson::Document customNodeConfigInvalidLibraryNameTypeParsed;
    customNodeConfigInvalidLibraryNameTypeParsed.Parse(customNodeConfigInvalidLibraryNameType);
    auto result = ovms::validateJsonAgainstSchema(customNodeConfigInvalidLibraryNameTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeConfigNoLibraryName) {
    const char* customNodeConfigNoLibraryName = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "type": "custom",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
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

    rapidjson::Document customNodeConfigNoLibraryNameParsed;
    customNodeConfigNoLibraryNameParsed.Parse(customNodeConfigNoLibraryName);
    auto result = ovms::validateJsonAgainstSchema(customNodeConfigNoLibraryNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeConfigModelNameShouldNotBeAcceptedInCustomNode) {
    const char* customNodeConfigModelName = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": "dummy_library",
                        "model_name": "dummy",
                        "type": "custom",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
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

    rapidjson::Document customNodeConfigModelNameParsed;
    customNodeConfigModelNameParsed.Parse(customNodeConfigModelName);
    auto result = ovms::validateJsonAgainstSchema(customNodeConfigModelNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeConfigNotAppropiateParameterShouldNotBeAcceptedInCustomNode) {
    const char* customNodeConfigNotAppropiateParameter = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": "dummy_library",
                        "type": "custom",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
                        "inputs": [
                            {"b": {"node_name": "request",
                                "data_item": "custom_dummy_input"}}
                        ],
                        "outputs": [
                            {"data_item": "a",
                            "alias": "new_dummy_output"}
                        ],
                        "not_appropiate": "not_appropiate"
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

    rapidjson::Document customNodeConfigNotAppropiateParameterParsed;
    customNodeConfigNotAppropiateParameterParsed.Parse(customNodeConfigNotAppropiateParameter);
    auto result = ovms::validateJsonAgainstSchema(customNodeConfigNotAppropiateParameterParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelNodeConfigLibraryNameShouldNotBeAcceptedInDLNode) {
    const char* modelNodeConfigLibraryName = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": "dummy_library",
                        "model_name": "dummy",
                        "type": "DL model",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
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

    rapidjson::Document modelNodeConfigLibraryNameParsed;
    modelNodeConfigLibraryNameParsed.Parse(modelNodeConfigLibraryName);
    auto result = ovms::validateJsonAgainstSchema(modelNodeConfigLibraryNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, ModelNodeConfigNotAppropiateParameterShouldNotBeAcceptedInDLNode) {
    const char* modelNodeConfigNotAppropiateParameter = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "model_name": "dummy",
                        "type": "DL model",
                        "params": {
                            "a": "1024",
                            "b": "512"
                        },
                        "inputs": [
                            {"b": {"node_name": "request",
                                "data_item": "custom_dummy_input"}}
                        ],
                        "outputs": [
                            {"data_item": "a",
                            "alias": "new_dummy_output"}
                        ],
                        "not_appropiate": "not_appropiate"
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

    rapidjson::Document modelNodeConfigNotAppropiateParameterParsed;
    modelNodeConfigNotAppropiateParameterParsed.Parse(modelNodeConfigNotAppropiateParameter);
    auto result = ovms::validateJsonAgainstSchema(modelNodeConfigNotAppropiateParameterParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, CustomNodeConfigParamsInvalidType) {
    const char* customNodeConfigParamsInvalidType = R"(
    {
        "model_config_list": [
            {
                "config": {
                    "name": "dummy",
                    "base_path": "dummy_path",
                    "target_device": "CPU",
                    "model_version_policy": {"all": {}},
                    "nireq": 1
                }
            }
        ],
        "custom_node_library_config_list": [
            {
                "name": "dummy_library",
                "base_path": "dummy_path"
            }
        ],
        "pipeline_config_list": [
            {
                "name": "pipeline1Dummy",
                "inputs": ["custom_dummy_input"],
                "nodes": [
                    {
                        "name": "dummyNode",
                        "library_name": "dummy_library",
                        "type": "custom",
                        "params": {
                            "a": 1024,
                            "b": "512"
                        },
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

    rapidjson::Document customNodeConfigParamsInvalidTypeParsed;
    customNodeConfigParamsInvalidTypeParsed.Parse(customNodeConfigParamsInvalidType);
    auto result = ovms::validateJsonAgainstSchema(customNodeConfigParamsInvalidTypeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

static const char* demultiplexerConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "dummy_path",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "custom_node_library_config_list": [
        {
            "name": "dummy_library",
            "base_path": "dummy_path"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "library_name": "dummy_library",
                    "type": "custom",
                    "params": {
                        "a": "1024",
                        "b": "512"
                    },
                    "inputs": [
                        {"b": {"node_name": "request",
                            "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                        "alias": "new_dummy_output"}
                    ],
                    "demultiply_count": 10,
                    "gather_from_node": "dummy"
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

TEST(SchemaTest, DemultiplexerConfigMatchingSchema) {
    rapidjson::Document demultiplexerConfigMatchingSchemaParsed;
    demultiplexerConfigMatchingSchemaParsed.Parse(demultiplexerConfig);
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, DemultiplexerConfigDemultiplyNegativeOneAllowed) {
    const std::string demultiplyCountToReplace{"\"demultiply_count\": 10"};
    const std::string demultiplyCount{"\"demultiply_count\": -1"};
    std::string config(demultiplexerConfig);
    config.replace(config.find(demultiplyCountToReplace), demultiplyCountToReplace.size(), demultiplyCount);
    rapidjson::Document demultiplexerConfigDemultiplyCountNegativeParsed;
    demultiplexerConfigDemultiplyCountNegativeParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigDemultiplyCountNegativeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, DemultiplexerConfigDemultiplyCountNegativeLowerThanNegativeOneNotAllowed) {
    const std::string demultiplyCountToReplace{"\"demultiply_count\": 10"};
    const std::string demultiplyCount{"\"demultiply_count\": -2"};
    std::string config(demultiplexerConfig);
    config.replace(config.find(demultiplyCountToReplace), demultiplyCountToReplace.size(), demultiplyCount);
    rapidjson::Document demultiplexerConfigDemultiplyCountNegativeParsed;
    demultiplexerConfigDemultiplyCountNegativeParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigDemultiplyCountNegativeParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, DemultiplexerConfigDemultiplyCountEqualsZeroAllowed) {
    // this is to allow dynamic demultiplexing
    const std::string demultiplyCountToReplace{"\"demultiply_count\": 10"};
    const std::string demultiplyCount{"\"demultiply_count\": 0"};
    std::string config(demultiplexerConfig);
    config.replace(config.find(demultiplyCountToReplace), demultiplyCountToReplace.size(), demultiplyCount);
    rapidjson::Document demultiplexerConfigDemultiplyCountEqualsZeroParsed;
    demultiplexerConfigDemultiplyCountEqualsZeroParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigDemultiplyCountEqualsZeroParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, DemultiplexerConfigDemultiplyCountEqualsOneAllowed) {
    const std::string demultiplyCountToReplace{"\"demultiply_count\": 10"};
    const std::string demultiplyCount{"\"demultiply_count\": 1"};
    std::string config(demultiplexerConfig);
    config.replace(config.find(demultiplyCountToReplace), demultiplyCountToReplace.size(), demultiplyCount);
    rapidjson::Document demultiplexerConfigDemultiplyCountEqualsOneParsed;
    demultiplexerConfigDemultiplyCountEqualsOneParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigDemultiplyCountEqualsOneParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST(SchemaTest, DemultiplexerConfigDemultiplyCountTypeInvalid) {
    const std::string demultiplyCountToReplace{"\"demultiply_count\": 10"};
    const std::string demultiplyCount{"\"demultiply_count\": \"10\""};
    std::string config(demultiplexerConfig);
    config.replace(config.find(demultiplyCountToReplace), demultiplyCountToReplace.size(), demultiplyCount);
    rapidjson::Document demultiplexerConfigDemultiplyCountTypeInvalidParsed;
    demultiplexerConfigDemultiplyCountTypeInvalidParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigDemultiplyCountTypeInvalidParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST(SchemaTest, DemultiplexerConfigGatherFromNodeTypeInvalid) {
    const std::string gatherFromNodeToReplace{"\"gather_from_node\": \"dummy\""};
    const std::string gatherFromNode{"\"gather_from_node\": 10"};
    std::string config(demultiplexerConfig);
    config.replace(config.find(gatherFromNodeToReplace), gatherFromNodeToReplace.size(), gatherFromNode);
    rapidjson::Document demultiplexerConfigGatherFromNodeTypeInvalidParsed;
    demultiplexerConfigGatherFromNodeTypeInvalidParsed.Parse(config.c_str());
    auto result = ovms::validateJsonAgainstSchema(demultiplexerConfigGatherFromNodeTypeInvalidParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
            "graph_path": "dummy_path"
        }
        ]
    })";

    rapidjson::Document configDoc;
    configDoc.Parse(mediapipeConfigPositive);
    auto result = ovms::validateJsonAgainstSchema(configDoc, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
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
