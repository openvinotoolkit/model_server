//*****************************************************************************
// Copyright 2020-2023 Intel Corporation
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
#include <regex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../config.hpp"
#include "../dags/pipeline.hpp"
#include "../dags/pipeline_factory.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#endif
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

std::string createStressTestPipelineOneDummyConfig() {
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

static const char* stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig = R"(
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

const std::string basicMediapipeConfigWithAddedGraph = R"({
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

const std::string basicMediapipeConfigWithRemovedGraph = R"({
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

const std::string basicMediapipeConfigWithRemovedModel = R"({
    "model_config_list": [
    ],
    "mediapipe_config_list": [
    {
        "name":"pipeline1Dummy",
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt"
    }
    ]
})";

const std::string basicMediapipeConfigWithReloadedModel = R"({
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

const std::string basicMediapipeConfigWithNewGraphPath = R"({
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
        "graph_path":"/ovms/src/test/mediapipe/graphdummyadapterfull_dummyinputnames2.pbtxt"
    }
    ]
})";
#if (MEDIAPIPE_DISABLE == 0)
template <typename Request, typename Response>
void mediaexec(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, Request&, Response&, ovms::Status& status) {
    throw std::string("Unsupported");
}
template <>
void mediaexec<KFSRequest, KFSResponse>(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    ServableMetricReporter* ptr{nullptr};
    status = executorPtr->infer(&request,
        &response,
        ovms::ExecutionContext(
            ovms::ExecutionContext::Interface::GRPC,
            ovms::ExecutionContext::Method::Predict),
        ptr);
}
template <typename Request, typename Response>
void mediacreate(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, Request&, Response&, ovms::Status& status) {
    throw std::string("Unsupported");
}
template <>
void mediacreate<KFSRequest, KFSResponse>(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    status = manager.createPipeline(executorPtr, request.model_name(), &request, &response);
}
#endif
class StressPipelineConfigChanges : public TestWithTempDir {
protected:
    const uint loadThreadCount = 20;
    const uint beforeConfigChangeLoadTimeMs = 30;
    const uint afterConfigChangeLoadTimeMs = 50;
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

public:
    virtual std::string getServableName() {
        return pipelineName;
    }
    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        const std::string modelPathToReplace{"/ovms/src/test/dummy"};
        auto it = ovmsConfig.find(modelPathToReplace);
        if (it != std::string::npos) {
            ovmsConfig.replace(it, modelPathToReplace.size(), modelPath);
        }
        configFilePath = directoryPath + "/ovms_config.json";
    }
    // we setup the OVMS so that it does not have any models loaded but also prepare the fixture to have ovmsConfig & configFilePath set up
    void SetUpServerInstance(const std::string& initialConfigContent) {
        TestWithTempDir::SetUp();
        std::string port = "9178";
        std::string restPort = "9178";
        modelPath = directoryPath + "/dummy/";
        SetUpConfig(initialConfigContent);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);

        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
        randomizePorts(port, restPort);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, std::stoi(restPort)));  // required for metrics
        // ideally we would want to have emptyConfigWithMetrics
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/configs/emptyConfigWithMetrics.json"));  // FIXME the content of config json is irrelevant - we just need server to be ready for C-API use in mediapipe
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(serverSettings, 0));                                      // set to 0 to reload only through test and avoid races
        OVMS_Server* cserver;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        ovms::Server& server = ovms::Server::instance();
        manager = &(dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    }
    void SetUp() override {
        SetUpServerInstance(createStressTestPipelineOneDummyConfig());
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
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/2", std::filesystem::copy_options::recursive);
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
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/2", std::filesystem::copy_options::recursive);
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
    void performStressTest(
        void (StressPipelineConfigChanges::*triggerLoadInALoop)(
            std::future<void>&,
            std::future<void>&,
            ModelManager&,
            const std::set<StatusCode>&,
            const std::set<StatusCode>&,
            std::unordered_map<StatusCode, std::atomic<uint64_t>>&),
        void (StressPipelineConfigChanges::*configChangeOperation)(),
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
        for (uint i = 0; i != static_cast<uint>(StatusCode::STATUS_CODE_END); ++i) {
            createPipelineRetCodesCounters[static_cast<StatusCode>(i)] = 0;
        }
        // create worker threads
        std::vector<std::unique_ptr<std::thread>> workerThreads;
        for (uint i = 0; i < loadThreadCount; ++i) {
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
                        *(this->manager),
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
            SPDLOG_TRACE("Create:[{}]={} -- {}", static_cast<uint>(retCode), counter, ovms::Status(retCode).string());
            if (requiredLoadResults.find(retCode) != requiredLoadResults.end()) {
                EXPECT_GT(counter, 0) << static_cast<uint>(retCode) << ":" << ovms::Status(retCode).string() << " did not occur. This may indicate fail or fail in test setup";
                continue;
            }
            if (counter == 0) {
                continue;
            }
            EXPECT_TRUE(allowedLoadResults.find(retCode) != allowedLoadResults.end()) << "Ret code:"
                                                                                      << static_cast<uint>(retCode) << " message: " << ovms::Status(retCode).string()
                                                                                      << " was not allowed in test but occured during load";
        }
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
    template <
        typename RequestType = tensorflow::serving::GetModelMetadataRequest,
        typename ResponseType = tensorflow::serving::GetModelMetadataResponse,
        typename ServableType = ovms::Pipeline>
    void triggerGetPipelineMetadataInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        ModelManager& manager,
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
            status = ovms::GetModelMetadataImpl::getModelStatus(&request, &response, manager, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata));
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
    void triggerGetPipelineStatusInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        ModelManager& manager,
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
            status = ovms::GetModelStatusImpl::getModelStatus(&request, &response, manager, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelStatus));
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
    virtual inputs_info_t getExpectedInputsInfo() {
        return {{pipelineInputName,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Precision::FP32}}};
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
        ModelManager& manager,
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
            RequestType request2;  // DIRTY HACK WARNING
            RequestType request = preparePipelinePredictRequest(request2);
            ovms::Status createPipelineStatus = StatusCode::UNKNOWN_ERROR;
            if (typeid(ServableType) == typeid(ovms::Pipeline)) {
                createPipelineStatus = manager.createPipeline(pipelinePtr, pipelineName, &request, &response);
#if (MEDIAPIPE_DISABLE == 0)
            } else if (typeid(ServableType) == typeid(ovms::MediapipeGraphExecutor)) {
                mediacreate(executorPtr, manager, request, response, createPipelineStatus);
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
                mediaexec(executorPtr, manager, request, response, executePipelineStatus);
#endif
            }
            createPipelineRetCodesCounters[executePipelineStatus.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(executePipelineStatus.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(executePipelineStatus.getCode()) != allowedLoadResults.end()))
                << executePipelineStatus.string() << "\n";
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
                SPDLOG_DEBUG("Create:[{}]={}:{}", static_cast<uint>(retCode), ovms::Status(retCode).string(), counter);
            }
        }
        EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
        std::stringstream ss;
        ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
        SPDLOG_INFO(ss.str());
    }
};

static const std::string initialClearConfig = R"(
{
    "model_config_list": [
    ]
})";

class StressModelConfigChanges : public StressPipelineConfigChanges {
    const std::string modelName = "dummy";
    const std::string modelInputName = "b";
    const std::string modelOutputName = "a";

public:
    std::string getServableName() override {
        return modelName;
    }
    void SetUp() override {
        SetUpServerInstance(initialClearConfig);
    }
};

TEST_F(StressPipelineConfigChanges, AddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, KFSAddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        // XYZ &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse>,
        &StressPipelineConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, GetMetricsDuringLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::testCurrentRequestsMetric,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemoveDefaultVersionDuringPredictLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,  // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,    // we hit default version which is unloaded already but default is not changed yet
        StatusCode::MODEL_VERSION_MISSING};              // there is no default version since all are either not loaded properly or retired
    std::set<StatusCode> allowedLoadResults = {};
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangeToShapeAutoDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::changeToAutoShape,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemovePipelineDefinitionDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE};  // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangedPipelineConnectionNameDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddedNewPipelineDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RetireSpecificVersionUsedDuringPredictLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,          // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};           // version is retired but pipeline not invalidated yet
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddNewVersionDuringGetMetadataLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemoveDefaultVersionDuringGetMetadataLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangeToShapeAutoDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::changeToAutoShape,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemovePipelineDefinitionDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE};  // when pipeline is retired
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangedPipelineConnectionNameDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddedNewPipelineDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RetireSpecificVersionUsedDuringGetMetadataLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

class StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges : public StressPipelineConfigChanges {
public:
    void checkPipelineResponse(const std::string& pipelineOutputName,
        tensorflow::serving::PredictRequest& request,
        tensorflow::serving::PredictResponse& response) override {
        std::vector<float> result(requestData.begin(), requestData.end());
        std::transform(result.begin(), result.end(), result.begin(), [this](float f) -> float { return f + 1 - 0; });
        checkDummyResponse(pipelineOutputName, result, request, response, 1);
    }
};

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RemoveCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RenameCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ChangeParamCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ReduceQueueSizeCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, IncreaseQueueSizeCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::increaseQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RemoveCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RenameCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ChangeParamCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ReduceQueueSizeCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, IncreaseQueueSizeCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::increaseQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

class StressPipelineCustomNodesConfigChanges : public StressPipelineConfigChanges {
    const int64_t differentOpsFactorsInputSize = 4;
    const std::vector<float> factorsData{1., 3, 2, 2};
    const std::string pipelineFactorsInputName{"pipeline_factors"};

public:
    tensorflow::serving::PredictRequest preparePipelinePredictRequest(tensorflow::serving::PredictRequest) override {
        tensorflow::serving::PredictRequest request;
        preparePredictRequest(request, getExpectedInputsInfo());
        auto& input = (*request.mutable_inputs())[pipelineInputName];
        input.mutable_tensor_content()->assign((char*)requestData.data(), requestData.size() * sizeof(float));
        auto& factors = (*request.mutable_inputs())[pipelineFactorsInputName];
        factors.mutable_tensor_content()->assign((char*)factorsData.data(), factorsData.size() * sizeof(float));
        return request;
    }
    inputs_info_t getExpectedInputsInfo() override {
        return {{pipelineInputName,
                    std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, DUMMY_MODEL_INPUT_SIZE}, ovms::Precision::FP32}},
            {pipelineFactorsInputName,
                std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, differentOpsFactorsInputSize}, ovms::Precision::FP32}}};
    }
    void checkPipelineResponse(const std::string& pipelineOutputName,
        tensorflow::serving::PredictRequest& request,
        tensorflow::serving::PredictResponse& response) override {
        // we need to imitate -> different ops then dummy then max
        std::vector<float> result(requestData.begin(), requestData.end());
        std::transform(result.begin(), result.end(), result.begin(), [this](float f) -> float { return f * factorsData[2]; });
        checkDummyResponse(pipelineOutputName, result, request, response, 1);
    }
};

TEST_F(StressPipelineCustomNodesConfigChanges, RemoveCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::removeCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, ChangeCustomLibraryParamDuringPredictLoad) {
    // we change used PARAM during load. This change does not effect results, but should be enough to verify
    // correctness of this operation - no segfaults etc.
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &StressPipelineConfigChanges::changeCustomLibraryParam,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, RemoveCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::removeCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, ChangeCustomLibraryParamDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuouity of operation most of the time
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::changeCustomLibraryParam,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressModelConfigChanges, AddModelDuringGetModelStatusLoad) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::MODEL_NAME_MISSING,  // until first model is loaded
        StatusCode::OK};                 // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MODEL_VERSION_MISSING  // this should be hit if test is stressing enough, sporadically does not happen
    };
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineStatusInALoop,
        &StressPipelineConfigChanges::addFirstModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

#if (MEDIAPIPE_DISABLE == 0)
class StressMediapipeChanges : public StressPipelineConfigChanges {
    const std::string modelName = "dummy";
    const std::string modelInputName = "b";
    const std::string modelOutputName = "a";

public:
    std::string getServableName() override {
        return modelName;
    }
    void SetUp() override {
        SetUpServerInstance(createStressTestPipelineOneDummyConfig());
    }
};
TEST_F(StressMediapipeChanges, AddGraphDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &StressPipelineConfigChanges::addNewMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveGraphDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE};    // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &StressPipelineConfigChanges::removeMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveModelDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::MEDIAPIPE_EXECUTION_ERROR};                  // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM,  // Can happen when OVMSSessionCalculator fails to create side input packet
    };
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &StressPipelineConfigChanges::removeMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadModelDuringPredictLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &StressPipelineConfigChanges::reloadMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadMediapipeGraphDuringPredictLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &StressPipelineConfigChanges::reloadMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
// TODO status
// TODO metadata
#endif
