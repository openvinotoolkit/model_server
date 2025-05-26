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

#include "schema.hpp"

#include <fstream>
#include <string>
#include <sstream>
#include <utility>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>
#pragma warning(pop)
#ifdef _WIN32
#include <windows.h>
#endif

#include "filesystem.hpp"
#include "logging.hpp"
#include "status.hpp"

namespace ovms {
const std::string DAG_DEFINITIONS = R"(
		"source_node_names": {
			"type": "object",
			"required": ["node_name", "data_item"],
			"properties": {
				"node_name": {
					"type": "string"
				},
				"data_item": {
					"type": "string"
				}
			},
			"additionalProperties": false
		},
		"source_node": {
			"type": "object",
			"additionalProperties" : {
				"$ref": "#/definitions/source_node_names"
			},
			"minProperties": 1,
			"maxProperties": 1
		},
		"output_alias": {
			"type": "object",
			"required": ["data_item", "alias"],
			"properties": {
				"data_item": {
					"type": "string"
				},
				"alias": {
					"type": "string"
				}
			},
			"additionalProperties": false
		},
		"node_config": {
			"type": "object",
			"required": ["name", "type", "inputs", "outputs"],
			"oneOf": [
    			{
        			"properties": { "type": { "enum": ["custom"] } },
        			"required": ["library_name"],
					"not": { "required": ["model_name"] }
    			},
    			{
        			"properties": { "type": { "enum": ["DL model"] } },
        			"not": { "required": ["library_name"] },
					"required": ["model_name"]
    			}
  			],
			"properties": {
				"name": {
					"type": "string"
				},
				"model_name": {
					"type": "string"
				},
				"library_name": {
					"type": "string"
				},
				"type": {
					"type": "string",
					"enum": ["DL model", "custom"]
				},
				"version": {
					"type": "integer",
					"minimum": 1
				},
				"inputs": {
					"type": "array",
					"items": {
						"$ref": "#/definitions/source_node"
					}
				},
				"outputs": {
					"type": "array",
					"items": {
						"$ref": "#/definitions/output_alias"
					}
				},
				"params": {
					"type": "object",
					"additionalProperties": { "type": "string" } 
				},
				"demultiply_count": {
			"type": "integer",
			"minimum": -1,
			"maximum": 10000
				},
				"gather_from_node": {
					"type": "string"
				}
			},
			"additionalProperties": false
		},
		"pipeline_config": {
			"type": "object",
			"required": ["name", "nodes", "inputs", "outputs"],
			"properties": {
				"name": {
					"type": "string"
				},
				"nodes": {
					"type": "array",
					"items": {
						"$ref": "#/definitions/node_config"
					}
				},
				"inputs": {
					"type": "array",
					"items": {
						"type": "string"
					}
				},
				"outputs": {
					"type": "array",
					"items": {
						"$ref": "#/definitions/source_node"
					}
				},
        "demultiply_count" : {
			"type": "integer",
			"minimum": -1,
			"maximum": 10000
        }
			},
			"additionalProperties": false
		},
		"custom_node_library_config": {
			"type": "object",
			"required": ["name", "base_path"],
			"properties": {
				"name": {
					"type": "string"
				},
				"base_path": {
					"type": "string"
				}
			},
			"additionalProperties": false
		}
)";

const std::string MODEL_CONFIG_DEFINITION = R"(
"model_config": {
	"type": "object",
	"required": ["config"],
    "maxProperties": 1,
	"properties": {
		"config": {
			"type": "object",
			"required": ["name"],
			"properties": {
				"name": {
					"type": "string"
				},
				"base_path": {
					"type": "string"
				},
				"batch_size": {
					"type": ["integer", "string"],
					"minimum": 0
				},
				"graph_path": {
					"type": "string"
				},
				"subconfig": {
					"type": "string"
				},
				"model_version_policy": {
	"$ref": "#/definitions/model_version_policy"
				},
				"shape": {
		"$ref": "#/definitions/layout_shape_def"
				},
				"layout": {
		"$ref": "#/definitions/layout_shape_def"
				},
				"nireq": {
					"type": "integer",
					"minimum": 0
				},
				"target_device": {
					"type": "string"
				},
				"allow_cache": {
					"type": "boolean"
				},
				"plugin_config": {
					"type": "object",
		"additionalProperties": {"anyOf": [
						{"type": "string"},
						{"type": "boolean"},
						{"type": "number"}
					]}
				},
				"stateful": {
					"type": "boolean"
				},
				"idle_sequence_cleanup": {
					"type": "boolean"
				},
				"low_latency_transformation": {
					"type": "boolean"
				},
				"max_sequence_number": {
					"type": "integer",
					"minimum": 0
				},
				"custom_loader_options": {
					"type": "object",
					"required": ["loader_name"],
					"properties": {
						"loader_name": {
							"type": "string"
						}
					},
					"minProperties": 1
				}
			},
			"additionalProperties": false
		},
		"additionalProperties": false
})";

const std::string MODELS_CONFIG_SCHEMA = R"({
    "definitions": {)" + MODEL_CONFIG_DEFINITION +
                                         R"(},)" + DAG_DEFINITIONS + R"(,
		"custom_loader_config": {
			"type": "object",
			"required": ["config"],
            "maxProperties": 1,
			"properties": {
				"config": {
					"type": "object",
					"required": ["loader_name", "library_path"],
					"properties": {
						"loader_name": {
							"type": "string"
						},
						"library_path": {
							"type": "string"
						},
						"loader_config_file": {
							"type": "string"
						}
					},
					"additionalProperties": false
				},
				"additionalProperties": false
			}
		},
    "layout_shape_def": {
        "oneOf": [
        {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        {
            "type": "string"
        }
        ]
    },
        "all_version_policy":{
            "type": "object",
            "additionalProperties": false,
            "properties": {},
            "minProperties": 0,
            "maxProperties": 0
        },
        "specific_version_policy":{
            "type": "object",
            "additionalProperties": false,
            "maxProperties": 1,
            "properties": {
                "versions" : {
                  "type": "array",
                  "items": {
                        "type": "integer",
                        "minimum": 1
                  }
                }
            },
            "required": ["versions"]
        },
        "latest_version_policy":{
            "type": "object",
            "additionalProperties": false,
            "maxProperties": 1,
            "properties": {
                "num_versions" : {
                    "type": "integer",
                    "minimum": 1
                }
            },
            "required": ["num_versions"]
        },
        "model_version_policy": {
            "oneOf": [
                {
                  "maxProperties": 1,
                  "properties" : {"all" : {"$ref": "#/definitions/all_version_policy"}},
                  "required": ["all"],
                  "additionalProperties": false
                },
                {
                  "maxProperties": 1,
                  "properties" : {"specific" : {"$ref": "#/definitions/specific_version_policy"}},
                  "required": ["specific"],
                  "additionalProperties": false
                },
                {
                  "maxProperties": 1,
                  "properties" : {"latest" : {"$ref": "#/definitions/latest_version_policy"}},
                  "required": ["latest"],
                  "additionalProperties": false
                }
            ]
        },
    "mediapipe_config": {
        "type": "object",
        "required": ["name"],
        "properties": {
             "name": {
                 "type": "string"
             },
             "base_path": {
                 "type": "string"
             },
             "graph_path": {
                 "type": "string"
             },
             "subconfig": {
                 "type": "string"
             }
        },
        "additionalProperties": false
    }
	},
	"type": "object",
	"required": ["model_config_list"],
	"properties": {
		"custom_loader_config_list": {
			"type": "array",
			"items": {
				"$ref": "#/definitions/custom_loader_config"
			}
		},
		"model_config_list": {
			"type": "array",
			"items": {
				"$ref": "#/definitions/model_config"
			}
		},
        "pipeline_config_list": {
			"type": "array",
			"items": {
				"$ref": "#/definitions/pipeline_config"
			}
},)" +
#if (MEDIAPIPE_DISABLE == 0)
                                         R"("mediapipe_config_list": {
		"type": "array",
		"items": {
			"$ref": "#/definitions/mediapipe_config"
		}
},)" +
#endif
                                         R"("custom_node_library_config_list": {
			"type": "array",
			"items": {
				"$ref": "#/definitions/custom_node_library_config"
			}
		},
		"monitoring": {
            "maxProperties": 1,
			"type": "object",
			"required": ["metrics"],
			"properties":{
				"metrics": {
					"type": "object",
					"required": ["enable"],
					"properties": {
						"enable": {
							"type": "boolean"
						},
						"metrics_list": {
							"type": "array",
							"items": {
								"type": "string"
							}
						}
					},
					"additionalProperties": false
				},
				"additionalProperties": false
			},
			"additionalProperties": false
		}
	},
	"additionalProperties": false
})";

const char* MODELS_MAPPING_SCHEMA = R"(
{
    "type": "object",
    "properties": {
        "outputs":{
                "type": "object",
                "additionalProperties": {"type": "string"}
        },
        "inputs":{
            "type": "object",
            "additionalProperties": {"type": "string"}
        }
    },
    "additionalProperties": false
})";

const std::string MEDIAPIPE_SUBCONFIG_SCHEMA = R"({
    "definitions": {)" + MODEL_CONFIG_DEFINITION +
                                               R"(},
	"type": "object",
	"required": ["model_config_list"],
	"properties": {
		"model_config_list": {
			"type": "array",
			"items": {
				"$ref": "#/definitions/model_config"
			}
		}
    },
	"additionalProperties": false
}
})";

#ifdef _WIN32
class FileHandle {
public:
    FileHandle(const std::string& filename) :
        filename_(filename) {
        hFile_ = CreateFileA(
            filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
    }
    ~FileHandle() {
        if (hFile_ != INVALID_HANDLE_VALUE) {
            if (!CloseHandle(hFile_)) {
                SPDLOG_ERROR("Failed to close file: ", filename_);
            }
        }
    }

    HANDLE getHandle() const { return hFile_; }

private:
    HANDLE hFile_;
    std::string filename_;
};
#endif

class LoudFileInfoReporter {
    std::stringstream ss;

public:
    LoudFileInfoReporter(const std::string& filename, std::ifstream& file) {
#ifdef __linux__
        struct stat statTime;

        if (stat(filename.c_str(), &statTime) != 0) {
            SPDLOG_ERROR("Failed to debug-read fileconfig");
            return;
        }
        ss << "FileInfoReporter: " << filename
           << " time modification [s]: " << statTime.st_ctim.tv_sec
           << " [ns]: " << statTime.st_ctim.tv_nsec << std::endl;
#elif _WIN32
        // Windows implementation
        FileHandle fileHandle(filename);
        if (fileHandle.getHandle() == INVALID_HANDLE_VALUE) {
            SPDLOG_ERROR("Failed to open file for debug-read on Windows");
            return;
        }

        FILETIME creationTime, lastAccessTime, lastWriteTime;
        if (!GetFileTime(fileHandle.getHandle(), &creationTime, &lastAccessTime, &lastWriteTime)) {
            SPDLOG_ERROR("Failed to get file time on Windows");
            return;
        }

        SYSTEMTIME stUTC;
        FileTimeToSystemTime(&lastWriteTime, &stUTC);

        ULARGE_INTEGER fileTimeAsUint;  // FILETIME units are 100 nanoseconds
        fileTimeAsUint.LowPart = lastAccessTime.dwLowDateTime;
        fileTimeAsUint.HighPart = lastAccessTime.dwHighDateTime;
        uint64_t timeInNanoseconds = fileTimeAsUint.QuadPart * 100;

        ss << "FileInfoReporter: " << filename
           << " time modification [s]: " << std::setfill('0') << std::setw(2) << stUTC.wSecond
           << " [ns]: " << timeInNanoseconds << std::endl;
#endif
        std::string some;
        file.clear();
        file.seekg(0);
        while (file) {
            file >> some;
            ss << some << std::endl;
        }
        file.clear();
        file.seekg(0);
    }
    void log() {
        SPDLOG_DEBUG(ss.str());
    }
};

Status parseConfig(const std::string& jsonFilename, rapidjson::Document& configJson, std::string& jsonMd5, int wrongConfigFileRetryDelayMs, int maxConfigJsonReadRetry) {
    std::string md5;
    uint16_t counter = 0;
    Status intermediateStatus;
    do {
        SPDLOG_DEBUG("Loading configuration from {} for: {} time", jsonFilename, counter + 1);
        std::ifstream ifs(jsonFilename.c_str());
        LoudFileInfoReporter loud(jsonFilename, ifs);
        if (!ifs.good()) {
            SPDLOG_ERROR("Configuration file is invalid {}", jsonFilename);
            intermediateStatus = StatusCode::CONFIG_FILE_INVALID;
            loud.log();
            std::this_thread::sleep_for(std::chrono::milliseconds(wrongConfigFileRetryDelayMs));
            continue;
        }
        std::stringstream config;
        config << ifs.rdbuf();
        auto configContent = config.str();
        md5 = FileSystem::getStringMD5(configContent);
        rapidjson::ParseResult parseResult = configJson.Parse(configContent.c_str());
        if (parseResult.Code()) {
            SPDLOG_ERROR("Configuration file is not a valid JSON file. Error: {}",
                rapidjson::GetParseError_En(parseResult.Code()));
            intermediateStatus = StatusCode::JSON_INVALID;
            loud.log();
            std::this_thread::sleep_for(std::chrono::milliseconds(wrongConfigFileRetryDelayMs));
            continue;
        }
        intermediateStatus = StatusCode::OK;
        break;
    } while (++counter < maxConfigJsonReadRetry && !intermediateStatus.ok());
    jsonMd5 = md5;
    return intermediateStatus;
}

Status validateJsonAgainstSchema(rapidjson::Document& json, const char* schema, bool detailedError) {
    rapidjson::Document schemaJson;
    rapidjson::ParseResult parsingSucceeded = schemaJson.Parse(schema);
    if (parsingSucceeded.Code()) {
        std::string errorMsg = "JSON schema parse error:";
        errorMsg += rapidjson::GetParseError_En(parsingSucceeded.Code());
        errorMsg += ", at: ";
        errorMsg += parsingSucceeded.Offset();
        SPDLOG_ERROR("JSON schema parse error: {}, at: {}", rapidjson::GetParseError_En(parsingSucceeded.Code()), parsingSucceeded.Offset());
        return detailedError ? Status(StatusCode::JSON_INVALID, std::move(errorMsg)) : StatusCode::JSON_INVALID;
    }
    rapidjson::SchemaDocument parsedSchema(schemaJson);
    rapidjson::SchemaValidator validator(parsedSchema);
    if (!json.Accept(validator)) {
        rapidjson::StringBuffer sb;
        validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
        std::string invalidatingSchema = sb.GetString();
        std::string keyword = validator.GetInvalidSchemaKeyword();
        sb.Clear();
        validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
        std::string key = sb.GetString();

        std::string errorMsg = "JSON schema parse error:";
        errorMsg += invalidatingSchema;
        errorMsg += ". Keyword:";
        errorMsg += keyword;
        errorMsg += " Key: ";
        errorMsg += key;
        SPDLOG_ERROR(errorMsg);
        return detailedError ? Status(StatusCode::JSON_INVALID, std::move(errorMsg)) : StatusCode::JSON_INVALID;
    }

    return StatusCode::OK;
}

}  // namespace ovms
