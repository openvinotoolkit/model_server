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
#include "../kfs_frontend/kfs_graph_executor_impl.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "src/filesystem/localfilesystem.hpp"
#include "../logging.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "stress_test_utils.hpp"
#include "test_models.hpp"
#include "test_utils.hpp"

#if (MEDIAPIPE_DISABLE == 0)
template <>
void mediaexec<KFSRequest, KFSResponse>(std::unique_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    status = executorPtr->infer(&request,
        &response,
        ovms::ExecutionContext(
            ovms::ExecutionContext::Interface::GRPC,
            ovms::ExecutionContext::Method::Predict));
}

template <>
void mediacreate<KFSRequest, KFSResponse>(std::unique_ptr<MediapipeGraphExecutor>& executorPtr, ovms::ModelManager& manager, KFSRequest& request, KFSResponse& response, ovms::Status& status) {
    status = manager.createPipeline(executorPtr, request.model_name());
}
#endif

class StressPipelineConfigChanges : public ConfigChangeStressTest {
public:
    static void SetUpTestSuite() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test on Windows, sporadic";  // CVS-176244
#endif
    }
};

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
    // Graph path change triggers real reload, briefly entering NOT_LOADED_YET state
    std::set<StatusCode> allowedLoadResults = {StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerKFSGetPipelineMetadataInALoop,
        &ConfigChangeStressTest::reloadMediapipeGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

class StressMediapipeQueueChanges : public StressPipelineConfigChanges {
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
TEST_F(StressMediapipeQueueChanges, AddGraphDuringPredictLoad) {
    // we add another graph definition during load (queue-enabled graph)
    SetUpConfig(basicMediapipeQueueConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::addNewMediapipeQueueGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeQueueChanges, RemoveGraphDuringPredictLoad) {
    SetUpConfig(basicMediapipeQueueConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE};
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::removeMediapipeQueueGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeQueueChanges, RemoveModelDuringPredictLoad) {
    SetUpConfig(basicMediapipeQueueConfig);
    bool performWholeConfigReload = true;
    // With queue path, pre-initialized graphs may keep working with cached sessions
    // even after model removal, so MEDIAPIPE_PRECONDITION_FAILED may not occur
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK,
    };
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MEDIAPIPE_EXECUTION_ERROR,
        StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM,
        StatusCode::MEDIAPIPE_PRECONDITION_FAILED,
    };
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::removeMediapipeQueueGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeQueueChanges, ReloadModelDuringPredictLoad) {
    SetUpConfig(basicMediapipeQueueConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::reloadMediapipeQueueGraphUsedModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressMediapipeQueueChanges, ReloadMediapipeGraphDuringPredictLoad) {
    SetUpConfig(basicMediapipeQueueConfig);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerPredictInALoop<KFSRequest, KFSResponse, ovms::MediapipeGraphExecutor>,
        &ConfigChangeStressTest::reloadMediapipeQueueGraph,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
// Status and metadata tests are not duplicated for queue fixture because
// neither status nor metadata operations exercise the graph queue path.
#endif
