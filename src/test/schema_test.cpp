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
