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

#include <rapidjson/schema.h>
#include <rapidjson/error/error.h>
#include <rapidjson/error/en.h>

#include <spdlog/spdlog.h>

#include "schema.hpp"

namespace ovms {
    const char *MODELS_CONFIG_SCHEMA = R"({
    "definitions": {
        "model_config": {
            "type": "object",
            "required": ["config"],
            "properties": {
                "config": {
                    "type": "object",
                    "required": ["name", "base_path"],
                    "properties": {
                        "name": {"type": "string"},
                        "base_path": {"type": "string"},
                        "batch_size": {"type": ["integer", "string"]},
                        "model_version_policy": {"type": "object"},
                        "shape": {"type": ["object", "string"]},
                        "nireq": {"type": "integer"},
                        "target_device": {"type": "string"},
                        "plugin_config": {"type": "object"}
                    }
                }
            }
        }
    },
    "type": "object",
    "required": ["model_config_list"],
    "properties": {"model_config_list": {
        "type": "array",
        "items": {"$ref": "#/definitions/model_config"}
    }}
    })";

    const char *MODELS_MAPPING_INPUTS_SCHEMA = R"({
    "type": "object",
    "required": [
        "inputs"
    ],
    "properties": {
        "inputs":{
            "type": "object"
        }
    }
    })";

    const char *MODELS_MAPPING_OUTPUTS_SCHEMA = R"({
    "type": "object",
    "required": [
        "outputs"
    ],
    "properties": {
        "outputs":{
            "type": "object"
        }
    }
    })";

    StatusCode validateJsonAgainstSchema(rapidjson::Document &json, const char *schema) {
        rapidjson::Document schemaJson;
        rapidjson::ParseResult parsingSucceeded = schemaJson.Parse(schema);
        if (!parsingSucceeded) {
            spdlog::error("JSON schema parse error: %s", rapidjson::GetParseError_En(parsingSucceeded.Code()), parsingSucceeded.Offset());
            return StatusCode::JSON_INVALID;
        }
        rapidjson::SchemaDocument parsedSchema(schemaJson);
        rapidjson::SchemaValidator validator(parsedSchema);

        if (!json.Accept(validator)) {
            spdlog::error("JSON file does not match schema.");
            return StatusCode::JSON_INVALID;
        }

        return StatusCode::OK;
    }

}  // namespace ovms
