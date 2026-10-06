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
#include "src/filesystem/localfilesystem.hpp"
#include "../logging.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "src/servable_management/servablemanagermodule.hpp"
#include "../server.hpp"
#include "src/status.hpp"
#include "../stringutils.hpp"
#include "c_api_test_utils.hpp"
#include "stress_test_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::_;
using testing::Return;

class StressCapiConfigChanges : public ConfigChangeStressTest {
public:
    static void SetUpTestSuite() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test on Windows, sporadic";  // CVS-176244
#endif
    }
};

class ConfigChangeStressTestSingleModel : public ConfigChangeStressTestAsync {
public:
    static void SetUpTestSuite() {
#ifdef _WIN32
        GTEST_SKIP() << "Skipping test on Windows, sporadic";  // CVS-176244
#endif
    }
};

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
        &ConfigChangeStressTest::triggerCApiInferenceInALoop,
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
    bool performWholeConfigReload = false;
    const std::set<StatusCode> requiredLoadResults{StatusCode::OK};
    const std::set<StatusCode> allowedLoadResults{StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,
        StatusCode::MODEL_VERSION_MISSING};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressCapiConfigChanges, KFSAddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;
    const std::set<StatusCode> requiredLoadResults{StatusCode::OK};
    const std::set<StatusCode> allowedLoadResults{StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,
        StatusCode::MODEL_VERSION_MISSING};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionAdd,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}

TEST_F(StressCapiConfigChanges, RemoveDefaultVersionDuringPredictLoad) {
    bool performWholeConfigReload = true;  // we need whole config reload since there is no other way to dispose the model version other than removing it from config
    std::set<StatusCode> requiredLoadResults = {
        StatusCode::OK,
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};  // model gets retired once removed from config
    std::set<StatusCode> allowedLoadResults = {StatusCode::MODEL_VERSION_MISSING};
    performStressTest(
        &StressCapiConfigChanges::triggerCApiInferenceInALoop,
        &StressCapiConfigChanges::defaultVersionRemove,
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
