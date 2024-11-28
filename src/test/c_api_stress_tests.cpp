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
#include <regex>

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
#include "c_api_test_utils.hpp"
#include "stress_test_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

class StressCapiConfigChanges : public ConfigChangeStressTest {};

class ConfigChangeStressTestSingleModel : public ConfigChangeStressTestAsync {};

class StressModelCapiConfigChanges : public StressCapiConfigChanges {
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

TEST_F(ConfigChangeStressTestSingleModel, ChangeToEmptyConfigInference) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK,
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerCApiInferenceInALoopSingleModel,
        &ConfigChangeStressTest::changeToEmptyConfig,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(ConfigChangeStressTestAsync, ChangeToEmptyConfigAsyncInference) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK,
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &ConfigChangeStressTest::triggerCApiAsyncInferenceInALoop,
        &ConfigChangeStressTest::changeToEmptyConfig,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(ConfigChangeStressTestAsync, ChangeToWrongShapeAsyncInference) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::INVALID_SHAPE};
    performStressTest(
        &ConfigChangeStressTest::triggerCApiAsyncInferenceInALoop,
        &ConfigChangeStressTest::changeToWrongShapeOneModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(ConfigChangeStressTestAsync, ChangeToAutoShapeDuringAsyncInference) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MODEL_VERSION_NOT_LOADED_YET};
    performStressTest(
        &ConfigChangeStressTest::triggerCApiAsyncInferenceInALoop,
        &ConfigChangeStressTest::changeToAutoShapeOneModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(ConfigChangeStressTestAsyncStartEmpty, ChangeToLoadedModelDuringAsyncInference) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::PIPELINE_DEFINITION_NAME_MISSING,
        StatusCode::MODEL_NAME_MISSING,
        StatusCode::MODEL_VERSION_MISSING};
    performStressTest(
        &ConfigChangeStressTest::triggerCApiAsyncInferenceInALoop,
        &ConfigChangeStressTest::addFirstModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressCapiConfigChanges, AddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, KFSAddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, GetMetricsDuringLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::testCurrentRequestsMetric,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RemoveDefaultVersionDuringPredictLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,  // we hit default version which is unloaded already but default is not changed yet
        StatusCode::MODEL_VERSION_MISSING};              // there is no default version since all are either not loaded properly or retired
    std::set<StatusCode> allowedLoadResults = {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, ChangeToShapeAutoDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::changeToAutoShape,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RemovePipelineDefinitionDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE};  // we expect to stop creating pipelines
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, ChangedPipelineConnectionNameDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, AddedNewPipelineDuringPredictLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RetireSpecificVersionUsedDuringPredictLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    std::set<StatusCode> allowedLoadResults = {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, AddNewVersionDuringGetMetadataLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RemoveDefaultVersionDuringGetMetadataLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, ChangeToShapeAutoDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::changeToAutoShape,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RemovePipelineDefinitionDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE};  // when pipeline is retired
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::removePipelineDefinition,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, ChangedPipelineConnectionNameDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::changeConnectionName,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, AddedNewPipelineDuringGetMetadataLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::addNewPipeline,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressCapiConfigChanges, RetireSpecificVersionUsedDuringGetMetadataLoad) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressCapiConfigChanges, AddModelDuringGetModelStatusLoad) {
    bool performWholeConfigReload = true;  // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK};  // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {
        StatusCode::MODEL_VERSION_MISSING  // this should be hit if test is stressing enough, sporadically does not happen
    };
    performStressTest(
        &ConfigChangeStressTest::triggerCApiGetStatusInALoop,
        &ConfigChangeStressTest::addFirstModel,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

class StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges : public StressCapiConfigChanges {
public:
    void checkInferResponse(OVMS_InferenceResponse* response, std::string& expectedOutputName) override {
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
        ASSERT_EQ(std::string("custom_dummy_output"), outputName);
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
            EXPECT_EQ(data[i] + 2, outputData[i]) << "Different at:" << i << " place.";
        }
    }
};

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, RemoveCustomLibraryDuringPredictLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};
    std::set<StatusCode> allowedLoadResults = {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, RenameCustomLibraryDuringPredictLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, ChangeParamCustomLibraryDuringPredictLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, ReduceQueueSizeCustomLibraryDuringPredictLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, IncreaseQueueSizeCustomLibraryDuringPredictLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::increaseQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, RemoveCustomLibraryDuringGetMetadataLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};         // we hit when all config changes finish to propagate
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::removePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, RenameCustomLibraryDuringGetMetadataLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::renamePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, ChangeParamCustomLibraryDuringGetMetadataLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::changeParamPreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, ReduceQueueSizeCustomLibraryDuringGetMetadataLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::reduceQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressPipelineCustomNodesWithPreallocatedBuffersCapiConfigChanges, IncreaseQueueSizeCustomLibraryDuringGetMetadataLoad) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows";
#endif
    SetUpConfig(stressPipelineCustomNodeAddOneThenDummy);
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};                                 // we expect full continuity of operation
    std::set<StatusCode> allowedLoadResults = {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};  // might hit reload phase
    performStressTest(
        &StressCapiConfigChanges::triggerCApiGetMetadataInALoop,
        &StressCapiConfigChanges::increaseQueueSizePreallocatedCustomLibraryUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
