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
#if (MEDIAPIPE_DISABLE == 0)
#include "../kfs_frontend/kfs_graph_executor_impl.hpp"
#endif
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
#include "stress_test_utils.hpp"
#include "test_utils.hpp"

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

#if (MEDIAPIPE_DISABLE == 0)
template <>
void mediaexec<KFSRequest, KFSResponse>(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    status = executorPtr->infer(&request,
        &response,
        ovms::ExecutionContext(
            ovms::ExecutionContext::Interface::GRPC,
            ovms::ExecutionContext::Method::Predict));
}

template <>
void mediacreate<KFSRequest, KFSResponse>(std::shared_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    status = manager.createPipeline(executorPtr, request.model_name());
}
#endif

class StressPipelineConfigChanges : public ConfigChangeStressTest {};

class StressModelConfigChanges : public StressPipelineConfigChanges {
    const std::string modelName = "dummy";
    const std::string modelInputName = "b";
    const std::string modelOutputName = "a";

public:
    std::string getServableName() override {
        return modelName;
    }
    void SetUp() override {
        SetUpCAPIServerInstance(initialClearConfig);
    }
};

TEST_F(StressPipelineConfigChanges, AddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, KFSAddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        // XYZ &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse>,
        &ConfigChangeStressTest::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
// Workaround because we cannot start http server multiple times https://github.com/drogonframework/drogon/issues/2210
TEST_F(ConfigChangeStressTest, GetMetricsDuringLoad_DROGON) {
    if (!std::getenv("DROGON_RESTART")) {
        GTEST_SKIP() << "Run with DROGON_RESTART to enable this test";
    }
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::testCurrentRequestsMetric,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemoveDefaultVersionDuringPredictLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,  // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_MISSING};              // there is no default version since all are either not loaded properly or retired
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,  // we hit default version which is unloaded already but default is not changed yet
    };
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangeToShapeAutoDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::changeToAutoShape,
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
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangedPipelineConnectionNameDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddedNewPipelineDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RetireSpecificVersionUsedDuringPredictLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,          // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};           // version is retired but pipeline not invalidated yet
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddNewVersionDuringGetMetadataLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::defaultVersionAdd,
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
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangeToShapeAutoDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::changeToAutoShape,
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
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, ChangedPipelineConnectionNameDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, AddedNewPipelineDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RetireSpecificVersionUsedDuringGetMetadataLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::retireSpecificVersionUsed,
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
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RenameCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ChangeParamCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ReduceQueueSizeCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, IncreaseQueueSizeCustomLibraryDuringPredictLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::increaseQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RemoveCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, RenameCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ChangeParamCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, ReduceQueueSizeCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersConfigChanges, IncreaseQueueSizeCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::increaseQueueSizePreallocatedCustomLibraryUsed,
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
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::removeCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, ChangeCustomLibraryParamDuringPredictLoad) {
    // we change used PARAM during load. This change does not effect results, but should be enough to verify
    // correctness of this operation - no segfaults etc.
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<TFSPredictRequest, TFSPredictResponse>,
        &ConfigChangeStressTest::changeCustomLibraryParam,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, RemoveCustomLibraryDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::removeCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineCustomNodesConfigChanges, ChangeCustomLibraryParamDuringGetMetadataLoad) {
    SetUpConfig(stressPipelineCustomNodeDifferentOperationsThenDummyThenChooseMaximumConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation most of the time
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::changeCustomLibraryParam,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressModelConfigChanges, AddModelDuringGetModelStatusLoad) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::MODEL_NAME_MISSING,  // until first model is loaded
        StatusCode::OK};                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MODEL_VERSION_MISSING  // this should be hit if test is stressing enough, sporadically does not happen
    };
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::addFirstModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

#if (MEDIAPIPE_DISABLE == 0)
class StressMediapipeChanges : public StressPipelineConfigChanges {
    const std::string modelName = PIPELINE_1_DUMMY_NAME;
    const std::string modelInputName = "b";
    const std::string modelOutputName = "a";

public:
    std::string getServableName() override {
        return modelName;
    }
    void SetUp() override {
        SetUpCAPIServerInstance(createStressTestPipelineOneDummyConfig());
    }
};
TEST_F(StressMediapipeChanges, AddGraphDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::addNewMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveGraphDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE};    // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::removeMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveModelDuringPredictLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK,  // we expect full continuity of operation
        StatusCode::MEDIAPIPE_PRECONDITION_FAILED,
    };  // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MEDIAPIPE_EXECUTION_ERROR,
        StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM,  // Can happen when OVMSSessionCalculator fails to create side input packet
    };
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::removeMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadModelDuringPredictLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::reloadMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadMediapipeGraphDuringPredictLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::reloadMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressMediapipeChanges, AddGraphDuringStatusLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::addNewMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveGraphDuringStatusLoad) {
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::removeMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveModelDuringStatusLoad) {
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::removeMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadModelDuringStatusLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::reloadMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadMediapipeGraphDuringStatusLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerGetPipelineStatusInALoop,
        &ConfigChangeStressTest::reloadMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, AddGraphDuringMetadataLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::addNewMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveGraphDuringMetadataLoad) {
    // we add another definition during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE};
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::removeMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, RemoveModelDuringMetadataLoad) {
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::removeMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadModelDuringMetadataLoad) {
    // we change nireq during load
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::reloadMediapipeGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeChanges, ReloadMediapipeGraphDuringMetadataLoad) {
    SetUpConfig(basicMediapipeConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::reloadMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
#endif
