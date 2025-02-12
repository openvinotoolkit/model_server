//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../capi_frontend/servablemetadata.hpp"
#include "../config.hpp"
#include "../dags/pipeline.hpp"
#include "../dags/pipeline_factory.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../model_service.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#endif

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

static const std::string PIPELINE_1_DUMMY_NAME = "pipeline1Dummy";

enum class SERVABLE_TYPE {
    DAG,
    MEDIAPIPE
};

static std::string createStressTestPipelineOneDummyConfig() {
    return R"(
{
    "monitoring": {
        "metrics": {
            "enable": true,
            "metrics_list": [
                ")" +
           METRIC_NAME_CURRENT_REQUESTS +
           R"(",")" + METRIC_NAME_INFER_REQ_ACTIVE +
           R"(",")" + METRIC_NAME_REQUESTS_SUCCESS +
           R"(",")" + METRIC_NAME_INFER_REQ_QUEUE_SIZE + R"("]
        }
    },
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
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
}

static const std::string initialClearConfig = R"(
{
    "model_config_list": [
    ]
})";

static const char* stressTestPipelineOneDummyRemovedConfig = R"(
{
    "model_config_list": [
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
static const char* stressTestPipelineOneDummyConfigChangedToAutoOneModel = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "auto"}
            }
        }
    ]
})";
static const char* stressTestPipelineOneDummyConfigChangedToAuto = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "auto"}
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
static const char* stressTestPipelineOneDummyConfigPipelineRemoved = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
    ]
})";
static const char* stressTestPipelineOneDummyConfigChangeConnectionName = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
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
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output_changed_name"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output_changed_name"}
                }
            ]
        }
    ]
})";
static const char* stressTestPipelineOneDummyConfigAddNewPipeline = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy2ndPipeline",
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
    ]
})";
static const char* stressTestPipelineOneDummyConfigSpecificVersionUsed = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
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
                    "model_name": "dummy",
                    "version": 1,
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

static const char* stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumRemovedLibraryConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 100
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "custom_node",
                    "params": {
                        "selection_criteria": "MAXIMUM_MINIMUM"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "choose_max",
                                         "data_item": "maximum_tensor_alias"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumChangedParamConfig = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_perform_different_operations",
            "base_path": "/ovms/bazel-bin/src/lib_node_perform_different_operations.so"
        },
        {
            "name": "lib_choose_maximum",
            "base_path": "/ovms/bazel-bin/src/lib_node_choose_maximum.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 100
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input", "pipeline_factors"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_perform_different_operations",
                    "type": "custom",
                    "demultiply_count": 4,
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}},
                        {"op_factors": {"node_name": "request",
                                           "data_item": "pipeline_factors"}}
                    ],
                    "outputs": [
                        {"data_item": "different_ops_results",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                },
                {
                    "name": "choose_max",
                    "library_name": "lib_choose_maximum",
                    "type": "custom",
                    "gather_from_node": "custom_node",
                    "params": {
                        "selection_criteria": "MAXIMUM_AVERAGE"
                    },
                    "inputs": [
                        {"input_tensors": {"node_name": "dummyNode",
                                           "data_item": "dummy_output"}}
                    ],
                    "outputs": [
                        {"data_item": "maximum_tensor",
                         "alias": "maximum_tensor_alias"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "choose_max",
                                     "data_item": "maximum_tensor_alias"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummy = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_add_one",
            "base_path": "/ovms/bazel-bin/src/libcustom_node_add_one.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "20",
                        "info_queue_size": "20",
                        "add_number": "1",
                        "sub_number": "0"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummyRemovedLibraryConfig = R"(
{
    "custom_node_library_config_list": [],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "20",
                        "info_queue_size": "20",
                        "add_number": "1",
                        "sub_number": "0"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummyChangedLibraryName = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_add_one_changed_name",
            "base_path": "/ovms/bazel-bin/src/libcustom_node_add_one.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one_changed_name",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "20",
                        "info_queue_size": "20",
                        "add_number": "1",
                        "sub_number": "0"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummyChangedParam = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_add_one",
            "base_path": "/ovms/bazel-bin/src/libcustom_node_add_one.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "20",
                        "info_queue_size": "20",
                        "add_number": "2",
                        "sub_number": "1"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummyReducedQueueSize = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_add_one",
            "base_path": "/ovms/bazel-bin/src/libcustom_node_add_one.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "10",
                        "info_queue_size": "10",
                        "add_number": "1",
                        "sub_number": "0"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressPipelineCustomNodeAddOneThenDummyIncreasedQueueSize = R"(
{
    "custom_node_library_config_list": [
        {
            "name": "lib_add_one",
            "base_path": "/ovms/bazel-bin/src/libcustom_node_add_one.so"
        }
    ],
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 20,
                "shape": {"b": "(1,10) "}
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "pipeline1Dummy",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "custom_node",
                    "library_name": "lib_add_one",
                    "type": "custom",
                    "params": {
                        "output_queue_size": "30",
                        "info_queue_size": "30",
                        "add_number": "1",
                        "sub_number": "0"
                    },
                    "inputs": [
                        {"input_numbers": {"node_name": "request",
                                           "data_item": "custom_dummy_input"}}
                    ],
                    "outputs": [
                        {"data_item": "output_numbers",
                         "alias": "custom_node_output"}
                    ]
                },
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "custom_node",
                               "data_item": "custom_node_output"}}
                    ],
                    "outputs": [
                        {"data_item": "a",
                         "alias": "dummy_output"}
                    ]
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                     "data_item": "dummy_output"}
                }
            ]
        }
    ]
})";

static const char* stressTestOneDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,10) "}
            }
        }
    ]
})";

static const char* stressTestOneDummyConfigWrongShape = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "shape": {"b": "(1,20) "}
            }
        }
    ]
})";

const std::string basicMediapipeConfig = R"({
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    }
    ]
})";

static const std::string basicMediapipeConfigWithAddedGraph = R"({
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    },
    {
        "name":"mediaDummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummy.pbtxt"
    }
    ]
})";

static const std::string basicMediapipeConfigWithRemovedGraph = R"({
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    ]
})";

static const std::string basicMediapipeConfigWithRemovedModel = R"({
    "model_config_list": [
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    }
    ]
})";

static const std::string basicMediapipeConfigWithReloadedModel = R"({
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "nireq": 47
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    }
    ]
})";

static const std::string basicMediapipeConfigWithNewGraphPath = R"({
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    }
    ]
})";

#if (MEDIAPIPE_DISABLE == 0)
template <typename Request, typename Response>
static void mediaexec(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, Request&, Response&, ovms::Status& status) {
    throw std::string("Unsupported");
}

template <typename Request, typename Response>
static void mediacreate(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, Request&, Response&, ovms::Status& status) {
    throw std::string("Unsupported");
}
#endif

#define GET_CAPI_STATUS_CODE(status)                          \
    uint32_t code = 0;                                        \
    OVMS_Status* codeStatus = OVMS_StatusCode(status, &code); \
    StatusCode sc;                                            \
    if (codeStatus != nullptr) {                              \
        sc = static_cast<StatusCode>(StatusCode::OK);         \
    } else {                                                  \
        sc = static_cast<StatusCode>(code);                   \
    }

class ConfigChangeStressTest : public TestWithTempDir {
protected:
    const uint32_t loadThreadCount = 20;
    const uint32_t beforeConfigChangeLoadTimeMs = 30;
    const uint32_t afterConfigChangeLoadTimeMs = 50;
    const int stressIterationsLimit = 5000;

    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;

    const std::string& pipelineName = PIPELINE_1_DUMMY_NAME;
    const std::string pipelineInputName = "custom_dummy_input";
    const std::string pipelineOutputName = "custom_dummy_output";
    // 1.1 for different ops test to be sure that always demultiplication
    // producess highest results
    const std::vector<float> requestData{1.1, 2., 3., 7., 5., 6., 4., 9., 10., 8.};

    ModelManager* manager;
    OVMS_Server* cserver;

public:
    virtual std::string getServableName() {
        return pipelineName;
    }
    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        std::string pathToReplace{"/ovms/src/test/dummy"};
        auto it = ovmsConfig.find(pathToReplace);
        if (it != std::string::npos) {
            ovmsConfig.replace(it, pathToReplace.size(), modelPath);
        }
        pathToReplace = std::string{"/ovms/bazel-bin"};
        it = ovmsConfig.find(pathToReplace);
        while (it != std::string::npos) {
            std::string newDir = getGenericFullPathForTmp("/tmp");
            std::filesystem::path resolved_path = std::filesystem::absolute(newDir + "/../bazel-bin").lexically_normal();
            newDir = resolved_path.generic_string();
            ovmsConfig.replace(it, pathToReplace.size(), newDir);
            it = ovmsConfig.find(pathToReplace);
        }
        pathToReplace = std::string{"/ovms/src/test"};
        it = ovmsConfig.find(pathToReplace);
        while (it != std::string::npos) {
            std::string newDir = getGenericFullPathForSrcTest("/ovms/src/test");
            std::filesystem::path resolved_path = std::filesystem::absolute(newDir).lexically_normal();
            newDir = resolved_path.generic_string();
            ovmsConfig.replace(it, pathToReplace.size(), newDir);
            it = ovmsConfig.find(pathToReplace);
        }
        configFilePath = directoryPath + "/ovms_config.json";
    }

    virtual inputs_info_t getExpectedInputsInfo() {
        return {{pipelineInputName,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Precision::FP32}}};
    }
    // we setup the OVMS so that it does not have any models loaded but also prepare the fixture to have ovmsConfig & configFilePath set up
    void SetUpCAPIServerInstance(const std::string& initialConfigContent) {
        TestWithTempDir::SetUp();
        std::string port = "9178";
        std::string restPort = "9178";
        modelPath = directoryPath + "/dummy/";
        SetUpConfig(initialConfigContent);
        std::string inputPath = getGenericFullPathForSrcTest("/ovms/src/test/dummy");
        std::filesystem::copy(inputPath.c_str(), modelPath, std::filesystem::copy_options::recursive);
        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
        randomizePorts(port, restPort);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
#if (USE_DROGON == 0)                                                                                  // when jusing drogon we cannot start rest server multiple times within the same process
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, std::stoi(restPort)));  // required for metrics  - but disabled because drogon http server cannot be restarted
#endif
        // ideally we would want to have emptyConfigWithMetrics
#if (USE_DROGON == 0)
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/emptyConfigWithMetrics.json").c_str()));  // the content of config json is irrelevant - we just need server to be ready for C-API use in mediapipe
#else
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/emptyConfig.json").c_str()));  // the content of config json is irrelevant - we just need server to be ready for C-API use in mediapipe
#endif
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(serverSettings, 0));  // set to 0 to reload only through test and avoid races
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        ovms::Server& server = ovms::Server::instance();
        manager = &(dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    }
    void SetUp() override {
        SetUpCAPIServerInstance(createStressTestPipelineOneDummyConfig());
    }
    void TearDown() override {
        OVMS_Server* cserver;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        ovms::Server& server = ovms::Server::instance();
        manager->join();
        server.setShutdownRequest(1);
        OVMS_ServerDelete(cserver);
        server.setShutdownRequest(0);
    }
    void defaultVersionRemove() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyRemovedConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void defaultVersionAdd() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/dummy/1"), modelPath + "/2", std::filesystem::copy_options::recursive);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void addFirstModel() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestOneDummyConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeToAutoShape() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyConfigChangedToAuto);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeToAutoShapeOneModel() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyConfigChangedToAutoOneModel);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeToWrongShapeOneModel() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestOneDummyConfigWrongShape);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeToEmptyConfig() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(initialClearConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removePipelineDefinition() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyConfigPipelineRemoved);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeConnectionName() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyConfigChangeConnectionName);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void addNewPipeline() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressTestPipelineOneDummyConfigAddNewPipeline);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void retireSpecificVersionUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/dummy/1"), modelPath + "/2", std::filesystem::copy_options::recursive);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removeCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumRemovedLibraryConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeCustomLibraryParam() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumChangedParamConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removePreallocatedCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeAddOneThenDummyRemovedLibraryConfig);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void renamePreallocatedCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeAddOneThenDummyChangedLibraryName);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeParamPreallocatedCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeAddOneThenDummyChangedParam);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void reduceQueueSizePreallocatedCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeAddOneThenDummyReducedQueueSize);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void increaseQueueSizePreallocatedCustomLibraryUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(stressPipelineCustomNodeAddOneThenDummyIncreasedQueueSize);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void addNewMediapipeGraph() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(basicMediapipeConfigWithAddedGraph);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removeMediapipeGraph() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(basicMediapipeConfigWithRemovedGraph);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removeMediapipeGraphUsedModel() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(basicMediapipeConfigWithRemovedModel);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void reloadMediapipeGraphUsedModel() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(basicMediapipeConfigWithReloadedModel);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void reloadMediapipeGraph() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        SetUpConfig(basicMediapipeConfigWithNewGraphPath);
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void checkMetricGreaterThan(const std::string& metricName, double value, std::string& metricOutput, bool& result) {
        ASSERT_THAT(metricOutput, ::testing::HasSubstr(metricName + std::string{"{name=\"dummy\",version=\"1\"} "})) << "cannot find dummys " << metricName << " metric\n"
                                                                                                                     << metricOutput;
        std::regex findActualMetricRgx(std::string{".*"} + metricName + std::string{"\\{name=\"dummy\",version=\"1\"\\} (.*)\n.*"});
        std::regex findRequestsSuccessMetricRgx(std::string{".*"} + METRIC_NAME_REQUESTS_SUCCESS + std::string{"\\{api=\"TensorFlowServing\",interface=\"gRPC\",method=\"Predict\",name=\"dummy\",version=\"1\"\\} (.*)\n.*"});
        std::smatch match;
        ASSERT_TRUE(std::regex_search(metricOutput, match, findActualMetricRgx)) << "cannot find dummys " << metricName << " metric\n"
                                                                                 << metricOutput;
        auto actualVal = ovms::stoi64(match[1]);
        ASSERT_TRUE(std::regex_search(metricOutput, match, findRequestsSuccessMetricRgx)) << "cannot find dummys " << METRIC_NAME_REQUESTS_SUCCESS << " metric\n"
                                                                                          << metricOutput;
        auto requestsSuccessCounter = ovms::stoi64(match[1]);
        ASSERT_TRUE(requestsSuccessCounter.has_value()) << "cannot parse " << METRIC_NAME_REQUESTS_SUCCESS << "\n"
                                                        << metricOutput;
        SPDLOG_DEBUG("{} value: {}", METRIC_NAME_REQUESTS_SUCCESS, requestsSuccessCounter.value());
        ASSERT_TRUE(actualVal.has_value()) << "cannot parse " << metricName << " metric to number\n"
                                           << metricOutput;
        result = actualVal.value() > value;
        if (!result)
            std::cerr << metricName << " metric needs to be greater than " << value << std::endl
                      << metricOutput;
    }
    void performStressTest(
        void (ConfigChangeStressTest::*triggerLoadInALoop)(
            std::future<void>&,
            std::future<void>&,
            const std::set<StatusCode>&,
            const std::set<StatusCode>&,
            std::unordered_map<StatusCode, std::atomic<uint64_t>>&),
        void (ConfigChangeStressTest::*configChangeOperation)(),
        bool reloadWholeConfig,
        std::set<StatusCode> requiredLoadResults,
        std::set<StatusCode> allowedLoadResults) {
        createConfigFileWithContent(ovmsConfig, configFilePath);
        auto status = manager->loadConfig(configFilePath);
        ASSERT_TRUE(status.ok());

        // setup helper variables for managing threads
        std::vector<std::promise<void>> startSignals(loadThreadCount);
        std::vector<std::promise<void>> stopSignals(loadThreadCount);
        std::vector<std::future<void>> futureStartSignals;
        std::vector<std::future<void>> futureStopSignals;
        std::transform(startSignals.begin(),
            startSignals.end(),
            std::back_inserter(futureStartSignals),
            [](auto& p) { return p.get_future(); });
        std::transform(stopSignals.begin(),
            stopSignals.end(),
            std::back_inserter(futureStopSignals),
            [](auto& p) { return p.get_future(); });
        std::unordered_map<StatusCode, std::atomic<uint64_t>> createPipelineRetCodesCounters;
        for (uint32_t i = 0; i != static_cast<uint32_t>(StatusCode::STATUS_CODE_END); ++i) {
            createPipelineRetCodesCounters[static_cast<StatusCode>(i)] = 0;
        }
        // create worker threads
        std::vector<std::unique_ptr<std::thread>> workerThreads;
        for (uint32_t i = 0; i < loadThreadCount; ++i) {
            workerThreads.emplace_back(std::make_unique<std::thread>(
                [this,
                    &triggerLoadInALoop,
                    &futureStartSignals,
                    &futureStopSignals,
                    &requiredLoadResults,
                    &allowedLoadResults,
                    &createPipelineRetCodesCounters,
                    i]() {
                    ((*this).*triggerLoadInALoop)(futureStartSignals[i],
                        futureStopSignals[i],
                        requiredLoadResults,
                        allowedLoadResults,
                        createPipelineRetCodesCounters);
                }));
        }
        // start initial load
        std::for_each(startSignals.begin(), startSignals.end(), [](auto& startSignal) { startSignal.set_value(); });
        // sleep to allow all load threads to stress ovms during config changes
        std::this_thread::sleep_for(std::chrono::milliseconds(beforeConfigChangeLoadTimeMs));
        ((*this).*configChangeOperation)();
        if (reloadWholeConfig) {
            manager->loadConfig(configFilePath);
        } else {
            manager->updateConfigurationWithoutConfigFile();
        }
        // wait to work strictly on config operations after change
        std::this_thread::sleep_for(std::chrono::milliseconds(afterConfigChangeLoadTimeMs));
        std::for_each(stopSignals.begin(), stopSignals.end(), [](auto& stopSignal) { stopSignal.set_value(); });
        std::for_each(workerThreads.begin(), workerThreads.end(), [](auto& t) { t->join(); });

        for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
            SPDLOG_TRACE("Create:[{}]={} -- {}", static_cast<uint32_t>(retCode), counter, ovms::Status(retCode).string());
            if (requiredLoadResults.find(retCode) != requiredLoadResults.end()) {
                EXPECT_GT(counter, 0) << static_cast<uint32_t>(retCode) << ":" << ovms::Status(retCode).string() << " did not occur. This may indicate fail or fail in test setup";
                continue;
            }
            if (counter == 0) {
                continue;
            }
            EXPECT_TRUE(allowedLoadResults.find(retCode) != allowedLoadResults.end()) << "Ret code:"
                                                                                      << static_cast<uint32_t>(retCode) << " message: " << ovms::Status(retCode).string()
                                                                                      << " was not allowed in test but occurred during load";
        }
    }
    void checkActiveNireqSmallerThanTotal() {
        std::string metricOutput = manager->getMetricRegistry()->collect();
        std::regex findNireqTotalRgx(std::string{".*"} + METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"\\{name=\"dummy\",version=\"1\"\\} (.*)\n.*"});
        std::regex findNireqActiveRgx(std::string{".*"} + METRIC_NAME_INFER_REQ_ACTIVE + std::string{"\\{name=\"dummy\",version=\"1\"\\} (.*)\n.*"});
        std::smatch match;
        ASSERT_TRUE(std::regex_search(metricOutput, match, findNireqTotalRgx)) << "cannot find dummys total nireq in metric\n"
                                                                               << metricOutput;
        auto totalNireq = ovms::stoi64(match[1]);
        ASSERT_TRUE(std::regex_search(metricOutput, match, findNireqActiveRgx)) << "cannot find dummys active nireq in metric\n"
                                                                                << metricOutput;
        auto activeNireq = ovms::stoi64(match[1]);
        ASSERT_TRUE(totalNireq.has_value()) << metricOutput;
        ASSERT_TRUE(activeNireq.has_value()) << metricOutput;
        ASSERT_LE(activeNireq.value(), totalNireq.value()) << metricOutput;
    }
    void testCurrentRequestsMetric() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        bool current_requests_pass = false, infer_req_active_pass = false;
        int retries = 3;
        for (int i = 0; i < retries; i++) {
            std::string metricOutput = manager->getMetricRegistry()->collect();
            checkMetricGreaterThan(METRIC_NAME_CURRENT_REQUESTS, 0, metricOutput, current_requests_pass);
            checkMetricGreaterThan(METRIC_NAME_INFER_REQ_ACTIVE, 0, metricOutput, infer_req_active_pass);
            if (current_requests_pass && infer_req_active_pass)
                break;
        }
        if (!current_requests_pass || !infer_req_active_pass)
            FAIL() << "Terminated after " << retries << " retries";
        checkActiveNireqSmallerThanTotal();
        SPDLOG_INFO("{} end", __FUNCTION__);
    }

    bool isMetadataResponseCorrect(tensorflow::serving::GetModelMetadataResponse& response, SERVABLE_TYPE servableType) {
        tensorflow::serving::SignatureDefMap def;
        EXPECT_EQ(response.model_spec().name(), pipelineName);
        EXPECT_TRUE(response.model_spec().has_version());
        EXPECT_EQ(response.model_spec().version().value(), 1);
        EXPECT_EQ(response.metadata_size(), 1);
        EXPECT_NE(
            response.metadata().find("signature_def"),
            response.metadata().end());
        response.metadata().at("signature_def").UnpackTo(&def);
        response.metadata().at("signature_def").UnpackTo(&def);
        const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
        const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();
        auto expectedInputs = getExpectedInputsInfo();
        std::cout << "Expected inputs: " << expectedInputs.size() << std::endl;
        bool inputsSizeCorrect{inputs.size() == expectedInputs.size()};
        EXPECT_TRUE(inputsSizeCorrect) << "Expected: " << expectedInputs.size() << " actual: " << inputs.size();
        bool outputsSizeCorrect{outputs.size() == 1};
        EXPECT_TRUE(outputsSizeCorrect) << "Expected: " << 1 << " actual: " << outputs.size();
        if (!inputsSizeCorrect || !outputsSizeCorrect) {
            return false;
        }
        for (auto& [expectedInputName, shapeTypeTuple] : expectedInputs) {
            bool inputNameExist = inputs.find(expectedInputName.c_str()) != inputs.end();
            EXPECT_TRUE(inputNameExist);
            if (!inputNameExist) {
                return false;
            }
            bool inputNameCorrect{inputs.at(pipelineInputName.c_str()).name() == pipelineInputName};
            EXPECT_TRUE(inputNameCorrect);
            if (!inputNameCorrect) {
                return false;
            }
            bool inputTypeCorrect{inputs.at(pipelineInputName.c_str()).dtype() == tensorflow::DT_FLOAT};
            EXPECT_TRUE(inputTypeCorrect);
            if (!inputTypeCorrect) {
                return false;
            }
            bool inputShapeCorrect = false;
            std::vector<int64_t> expectedShape;
            if (servableType == SERVABLE_TYPE::DAG) {
                expectedShape = std::vector<int64_t>{1, 10};
            } else if (servableType == SERVABLE_TYPE::MEDIAPIPE) {
                expectedShape = std::vector<int64_t>{};
            } else {
                EXPECT_TRUE(false) << "Unsupported checks";
            }
            inputShapeCorrect = isShapeTheSame(
                inputs.at(pipelineInputName.c_str()).tensor_shape(),
                std::move(expectedShape));
            EXPECT_TRUE(inputShapeCorrect);
            if (!inputShapeCorrect) {
                return false;
            }
        }
        bool outputNameExist{outputs.find(pipelineOutputName.c_str()) != outputs.end()};
        EXPECT_TRUE(outputNameExist);
        if (!outputNameExist) {
            return false;
        }
        bool outputNameCorrect{outputs.at(pipelineOutputName.c_str()).name() == pipelineOutputName};
        EXPECT_TRUE(outputNameCorrect);
        if (!outputNameCorrect) {
            return false;
        }
        bool outputTypeCorrect{outputs.at(pipelineOutputName.c_str()).dtype() == tensorflow::DT_FLOAT};
        EXPECT_TRUE(outputTypeCorrect);
        if (!outputTypeCorrect) {
            return false;
        }
        bool outputShapeCorrect{isShapeTheSame(
            outputs.at(pipelineOutputName.c_str()).tensor_shape(),
            {1, 10})};
        EXPECT_TRUE(outputShapeCorrect);
        if (!outputShapeCorrect) {
            return false;
        }
        return true;
    }
#if (MEDIAPIPE_DISABLE == 0)
    void isKFSMetadataResponseCorrect(KFSModelMetadataResponse& response, SERVABLE_TYPE servableType) {
        EXPECT_EQ(response.name(), pipelineName);
        EXPECT_EQ(response.versions().size(), 1);
        EXPECT_EQ(response.versions()[0], "1");
        EXPECT_EQ(response.platform(), "OpenVINO");
        EXPECT_EQ(response.inputs().size(), 1);
        EXPECT_EQ(response.inputs()[0].name(), "custom_dummy_input");
        EXPECT_EQ(response.inputs()[0].datatype(), "INVALID");
        EXPECT_TRUE(isShapeTheSame(response.inputs()[0].shape(), std::move(std::vector<int64_t>{})));
        EXPECT_EQ(response.outputs().size(), 1);
        EXPECT_EQ(response.outputs()[0].name(), "custom_dummy_output");
        EXPECT_EQ(response.outputs()[0].datatype(), "INVALID");
        EXPECT_TRUE(isShapeTheSame(response.outputs()[0].shape(), std::move(std::vector<int64_t>{})));
    }
#endif
    template <
        typename RequestType = tensorflow::serving::GetModelMetadataRequest,
        typename ResponseType = tensorflow::serving::GetModelMetadataResponse,
        typename ServableType = ovms::Pipeline>
    void triggerGetPipelineMetadataInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        tensorflow::serving::GetModelMetadataRequest request;
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            auto status = ovms::GetModelMetadataImpl::createGrpcRequest(pipelineName, 1, &request);
            tensorflow::serving::GetModelMetadataResponse response;
            status = ovms::GetModelMetadataImpl::getModelStatus(&request, &response, *(this->manager), ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata));
            createPipelineRetCodesCounters[status.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(status.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(status.getCode()) != allowedLoadResults.end()))
                << status.string() << "\n";
            if (!status.ok()) {
                continue;
            }
            // Check response if correct
            SERVABLE_TYPE servableType = SERVABLE_TYPE::DAG;
#if (MEDIAPIPE_DISABLE == 0)
            if (typeid(ServableType) == typeid(MediapipeGraphExecutor)) {
                servableType = SERVABLE_TYPE::MEDIAPIPE;
            }
#endif
            EXPECT_TRUE(isMetadataResponseCorrect(response, servableType));
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }
#if (MEDIAPIPE_DISABLE == 0)
    template <
        typename RequestType = KFSModelMetadataRequest,
        typename ResponseType = KFSModelMetadataResponse,
        typename ServableType = MediapipeGraphExecutor>
    void triggerKFSGetPipelineMetadataInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        KFSModelMetadataRequest request;
        request.set_name(getServableName());
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            KFSModelMetadataResponse response;
            KFSModelExtraMetadata extraMetadata;
            ovms::Server& server = ovms::Server::instance();
            KFSInferenceServiceImpl impl(server);
            auto status = impl.ModelMetadataImpl(nullptr, &request, &response, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata), extraMetadata);
            createPipelineRetCodesCounters[status.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(status.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(status.getCode()) != allowedLoadResults.end()))
                << status.string() << "\n";
            if (!status.ok()) {
                continue;
            }

            SERVABLE_TYPE servableType = SERVABLE_TYPE::DAG;
            if (typeid(ServableType) == typeid(MediapipeGraphExecutor)) {
                servableType = SERVABLE_TYPE::MEDIAPIPE;
            }
            isKFSMetadataResponseCorrect(response, servableType);
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }
#endif
    void triggerGetPipelineStatusInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        tensorflow::serving::GetModelStatusRequest request;
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        // for getModelStatus requests it must be much higher since the response time is much lower
        // as in contrast to predict/metadata requests
        auto stressIterationsCounter = stressIterationsLimit * 100000;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            auto status = ovms::GetModelStatusImpl::createGrpcRequest(getServableName(), 1, &request);
            tensorflow::serving::GetModelStatusResponse response;
            status = ovms::GetModelStatusImpl::getModelStatus(&request, &response, *(this->manager), ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelStatus));
            createPipelineRetCodesCounters[status.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(status.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(status.getCode()) != allowedLoadResults.end()))
                << status.string() << "\n";
            if (!status.ok()) {
                continue;
            }
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }

    virtual tensorflow::serving::PredictRequest preparePipelinePredictRequest(tensorflow::serving::PredictRequest) {
        tensorflow::serving::PredictRequest request;
        preparePredictRequest(request, getExpectedInputsInfo());
        auto& input = (*request.mutable_inputs())[pipelineInputName];
        input.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        return request;
    }
    virtual KFSRequest preparePipelinePredictRequest(KFSRequest&) {
        KFSRequest request;
        preparePredictRequest(request, getExpectedInputsInfo(), requestData);
        request.set_model_name(PIPELINE_1_DUMMY_NAME);
        return request;
    }
    virtual void checkPipelineResponse(const std::string& pipelineOutputName,
        TFSPredictRequest& request,
        TFSPredictResponse& response) {
        checkDummyResponse(pipelineOutputName, requestData, request, response, 1);
    }
    virtual void checkPipelineResponse(const std::string& pipelineOutputName,
        KFSRequest& request,
        KFSResponse& response) {
        checkDummyResponse(pipelineOutputName, requestData, request, response, 1, 1, pipelineName);
    }

    template <typename RequestType, typename ResponseType, typename ServableType = ovms::Pipeline>
    void triggerPredictInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        bool breakLoop = false;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (true == breakLoop) {
                SPDLOG_INFO("Ending Load");
                break;
            }
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Triggering last request");
                breakLoop = true;
            }
            std::unique_ptr<Pipeline> pipelinePtr;
#if (MEDIAPIPE_DISABLE == 0)
            std::shared_ptr<MediapipeGraphExecutor> executorPtr;
#endif
            ResponseType response;
            // little hack - we can't use noninitializad object to call function
            // and then return that object
            RequestType request2;
            RequestType request = preparePipelinePredictRequest(request2);
            ovms::Status createPipelineStatus = StatusCode::UNKNOWN_ERROR;
            if (typeid(ServableType) == typeid(ovms::Pipeline)) {
                createPipelineStatus = this->manager->createPipeline(pipelinePtr, pipelineName, &request, &response);
#if (MEDIAPIPE_DISABLE == 0)
            } else if (typeid(ServableType) == typeid(ovms::MediapipeGraphExecutor)) {
                mediacreate(executorPtr, *(this->manager), request, response, createPipelineStatus);
#endif
            }
            // we need to make sure that expected status happened and still accept
            // some that could happen but we may not hit them
            EXPECT_TRUE((requiredLoadResults.find(createPipelineStatus.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(createPipelineStatus.getCode()) != allowedLoadResults.end()))
                << createPipelineStatus.string() << "\n";
            if (!createPipelineStatus.ok()) {
                createPipelineRetCodesCounters[createPipelineStatus.getCode()]++;
                continue;
            }

            ovms::Status executePipelineStatus = StatusCode::UNKNOWN_ERROR;
            if (typeid(ServableType) == typeid(ovms::Pipeline)) {
                executePipelineStatus = pipelinePtr->execute(ovms::ExecutionContext(
                    ovms::ExecutionContext::Interface::GRPC,
                    ovms::ExecutionContext::Method::Predict));
#if (MEDIAPIPE_DISABLE == 0)
            } else if (typeid(ServableType) == typeid(ovms::MediapipeGraphExecutor)) {
                mediaexec(executorPtr, *(this->manager), request, response, executePipelineStatus);
#endif
            }
            createPipelineRetCodesCounters[executePipelineStatus.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(executePipelineStatus.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(executePipelineStatus.getCode()) != allowedLoadResults.end()))
                << executePipelineStatus.string() << " thread id:" << std::this_thread::get_id() << "\n";
            if (executePipelineStatus.ok()) {
                checkPipelineResponse(pipelineOutputName, request, response);
            }
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
        for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
            if (counter > 0) {
                SPDLOG_DEBUG("Create:[{}]={}:{}", static_cast<uint32_t>(retCode), ovms::Status(retCode).string(), counter);
            }
        }
        EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
        std::stringstream ss;
        ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
        SPDLOG_INFO(ss.str());
    }

    void isMetadataResponseCorrect(OVMS_ServableMetadata* servableMetadata) {
        ASSERT_NE(nullptr, servableMetadata);
        uint32_t inputCount = 42;
        uint32_t outputCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInputCount(servableMetadata, &inputCount));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutputCount(servableMetadata, &outputCount));
        auto expectedInputs = getExpectedInputsInfo();
        ASSERT_EQ(expectedInputs.size(), inputCount);
        ASSERT_EQ(1, outputCount);
        uint32_t id = 0;
        OVMS_DataType datatype = (OVMS_DataType)199;
        int64_t* shapeMin{nullptr};
        int64_t* shapeMax{nullptr};
        size_t dimCount = 42;
        const char* tensorName{nullptr};
        std::set<std::string> inputNames;
        std::set<std::string> outputNames;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
        EXPECT_EQ(std::string(tensorName), std::string("custom_dummy_input"));
        EXPECT_EQ(datatype, (OVMS_DataType)2);
        EXPECT_EQ(dimCount, 2);
        EXPECT_EQ(*shapeMin, {1});
        EXPECT_EQ(*shapeMax, {1});
    }

    void triggerCApiGetMetadataInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            OVMS_ServableMetadata* servableMetadata = nullptr;
            OVMS_Status* status = OVMS_GetServableMetadata(this->cserver, "pipeline1Dummy", 0, &servableMetadata);

            GET_CAPI_STATUS_CODE(status)
            createPipelineRetCodesCounters[sc]++;
            EXPECT_TRUE((requiredLoadResults.find(sc) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(sc) != allowedLoadResults.end()));
            if (sc == StatusCode::OK) {
                isMetadataResponseCorrect(servableMetadata);
                OVMS_ServableMetadataDelete(servableMetadata);
            }
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }
    void triggerCApiGetStatusInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        // for getModelStatus requests it must be much higher since the response time is much lower
        // as in contrast to predict/metadata requests
        auto stressIterationsCounter = stressIterationsLimit * 100000;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            OVMS_ServableState state;
            OVMS_Status* status = OVMS_GetServableState(this->cserver, "pipeline1Dummy", 0, &state);

            GET_CAPI_STATUS_CODE(status)
            createPipelineRetCodesCounters[sc]++;
            EXPECT_TRUE((requiredLoadResults.find(sc) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(sc) != allowedLoadResults.end()));
            if (sc != StatusCode::OK) {
                continue;
            }
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }

    virtual void checkInferResponse(OVMS_InferenceResponse* response, std::string& expectedOutputName) {
        ASSERT_NE(response, nullptr);
        uint32_t outputCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
        ASSERT_EQ(outputCount, 1);
        const void* voutputData = nullptr;
        size_t bytesize = 42;
        uint32_t outputId = 0;
        OVMS_DataType datatype = (OVMS_DataType)199;
        const int64_t* shape{nullptr};
        size_t dimCount = 42;
        OVMS_BufferType bufferType = (OVMS_BufferType)199;
        uint32_t deviceId = 42;
        const char* outputName{nullptr};
        ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
        ASSERT_EQ(std::string(expectedOutputName), outputName);
        EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
        EXPECT_EQ(dimCount, 2);
        EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
        EXPECT_EQ(deviceId, 0);
        for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
            EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
        }
        const float* outputData = reinterpret_cast<const float*>(voutputData);
        ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
        std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i] + 1, outputData[i]) << "Different at:" << i << " place.";
        }
    }

    void triggerCApiInferenceInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        bool breakLoop = false;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (true == breakLoop) {
                SPDLOG_INFO("Ending Load");
                break;
            }
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Triggering last request");
                breakLoop = true;
            }
            OVMS_InferenceRequest* request{nullptr};
            OVMS_InferenceRequestNew(&request, this->cserver, "pipeline1Dummy", 1);
            ASSERT_NE(nullptr, request);

            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "custom_dummy_input", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
            std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "custom_dummy_input", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, 0));

            OVMS_InferenceResponse* response = nullptr;
            OVMS_Status* status = OVMS_Inference(this->cserver, request, &response);
            OVMS_InferenceRequestDelete(request);

            GET_CAPI_STATUS_CODE(status)
            createPipelineRetCodesCounters[sc]++;
            EXPECT_TRUE((requiredLoadResults.find(sc) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(sc) != allowedLoadResults.end()));
            if (sc == StatusCode::OK) {
                std::string expectedOutputName = "custom_dummy_output";
                checkInferResponse(response, expectedOutputName);
            }
            OVMS_InferenceResponseDelete(response);

            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
            for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
                if (counter > 0) {
                    SPDLOG_DEBUG("Create:[{}]={}:{}", static_cast<uint32_t>(retCode), ovms::Status(retCode).string(), counter);
                }
            }

            EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
            std::stringstream ss;
            ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
            SPDLOG_INFO(ss.str());
        }
    }

    static void callbackUnblockingAndFreeingRequest(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
        SPDLOG_INFO("Using callback: callbackUnblockingAndFreeingRequest!");
        CallbackUnblockingStruct* callbackUnblockingStruct = reinterpret_cast<CallbackUnblockingStruct*>(userStruct);
        callbackUnblockingStruct->signal.set_value(42);
        callbackUnblockingStruct->response = response;
        OVMS_InferenceResponseDelete(response);
    }

    struct CallbackUnblockingStruct {
        std::promise<uint32_t> signal;
        OVMS_InferenceResponse* response;
    };

    void triggerCApiAsyncInferenceInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        bool breakLoop = false;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (true == breakLoop) {
                SPDLOG_INFO("Ending Load");
                break;
            }
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Triggering last request");
                breakLoop = true;
            }
            OVMS_InferenceRequest* request{nullptr};
            OVMS_InferenceRequestNew(&request, this->cserver, "dummy", 1);
            ASSERT_NE(nullptr, request);

            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "b", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
            std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "b", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, 0));

            OVMS_InferenceResponse* response = nullptr;
            CallbackUnblockingStruct callbackStruct;
            auto unblockSignal = callbackStruct.signal.get_future();
            callbackStruct.response = response;

            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackUnblockingAndFreeingRequest, reinterpret_cast<void*>(&callbackStruct)));

            OVMS_Status* status = OVMS_InferenceAsync(this->cserver, request);

            GET_CAPI_STATUS_CODE(status)
            // check - blocking call - expected only on success
            if (status != nullptr) {
                SPDLOG_INFO("OVMS_InferenceAsync failed with status: {}. Not waiting for callback execution.", ovms::Status(sc).string());
            } else {
                SPDLOG_INFO("OVMS_InferenceAsync GET START");
                auto callbackReturnValue = unblockSignal.get();
                SPDLOG_INFO("OVMS_InferenceAsync GET DONE");
                ASSERT_EQ(callbackReturnValue, 42);
            }

            OVMS_InferenceRequestDelete(request);

            createPipelineRetCodesCounters[sc]++;
            EXPECT_TRUE((requiredLoadResults.find(sc) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(sc) != allowedLoadResults.end()));

            if (sc == StatusCode::OK) {
                ASSERT_EQ(response, nullptr);
            }

            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
            for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
                if (counter > 0) {
                    SPDLOG_DEBUG("Create:[{}]={}:{}", static_cast<uint32_t>(retCode), ovms::Status(retCode).string(), counter);
                }
            }

            EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
            std::stringstream ss;
            ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
            SPDLOG_INFO(ss.str());
        }
    }

    void triggerCApiInferenceInALoopSingleModel(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        bool breakLoop = false;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (true == breakLoop) {
                SPDLOG_INFO("Ending Load");
                break;
            }
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Triggering last request");
                breakLoop = true;
            }
            OVMS_InferenceRequest* request{nullptr};
            OVMS_InferenceRequestNew(&request, this->cserver, "dummy", 1);
            ASSERT_NE(nullptr, request);

            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "b", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
            std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "b", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, 0));

            OVMS_InferenceResponse* response = nullptr;
            OVMS_Status* status = OVMS_Inference(this->cserver, request, &response);
            OVMS_InferenceRequestDelete(request);

            GET_CAPI_STATUS_CODE(status)
            createPipelineRetCodesCounters[sc]++;
            EXPECT_TRUE((requiredLoadResults.find(sc) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(sc) != allowedLoadResults.end()));

            if (sc == StatusCode::OK) {
                std::string expectedOutputName = "a";
                checkInferResponse(response, expectedOutputName);
            }
            OVMS_InferenceResponseDelete(response);

            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
            for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
                if (counter > 0) {
                    SPDLOG_DEBUG("Create:[{}]={}:{}", static_cast<uint32_t>(retCode), ovms::Status(retCode).string(), counter);
                }
            }

            EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
            std::stringstream ss;
            ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
            SPDLOG_INFO(ss.str());
        }
    }
};

class ConfigChangeStressTestAsync : public ConfigChangeStressTest {
    void SetUp() override {
        SetUpCAPIServerInstance(stressTestOneDummyConfig);
    }
};

class ConfigChangeStressTestAsyncStartEmpty : public ConfigChangeStressTest {
    void SetUp() override {
        SetUpCAPIServerInstance(initialClearConfig);
    }
};
