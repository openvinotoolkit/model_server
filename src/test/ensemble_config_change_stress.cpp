
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

#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

static const std::string PIPELINE_1_DUMMY_NAME = "pipeline1Dummy";

static const char* stressTestPipelineOneDummyConfig = R"(
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

class StressPipelineConfigChanges : public TestWithTempDir {
    const uint loadThreadCount = 10;
    const uint beforeConfigChangeLoadTimeMs = 30;
    const uint afterConfigChangeLoadTimeMs = 50;
    const int stressIterationsLimit = 5000;

    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;

    const std::string pipelineInputName = "custom_dummy_input";
    const std::string pipelineOutputName = "custom_dummy_output";

public:
    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        const std::string modelPathToReplace{"/ovms/src/test/dummy"};
        ovmsConfig.replace(ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), modelPath);
        configFilePath = directoryPath + "/ovms_config.json";
    }
    void TearDown() override {}
    void SetUp() override {
        TestWithTempDir::SetUp();
        modelPath = directoryPath + "/dummy/";
        SetUpConfig(stressTestPipelineOneDummyConfig);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    }
    void defaultVersionRemove() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        ovmsConfig = stressTestPipelineOneDummyRemovedConfig;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void defaultVersionAdd() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/2", std::filesystem::copy_options::recursive);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeToAutoShape() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        ovmsConfig = stressTestPipelineOneDummyConfigChangedToAuto;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void removePipelineDefinition() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        ovmsConfig = stressTestPipelineOneDummyConfigPipelineRemoved;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void changeConnectionName() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        ovmsConfig = stressTestPipelineOneDummyConfigChangeConnectionName;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void addNewPipeline() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        ovmsConfig = stressTestPipelineOneDummyConfigAddNewPipeline;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void retireSpecificVersionUsed() {
        SPDLOG_INFO("{} start", __FUNCTION__);
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/2", std::filesystem::copy_options::recursive);
        SPDLOG_INFO("{} end", __FUNCTION__);
    }
    void performStressTest(void (StressPipelineConfigChanges::*configChangeOperation)(),
        bool reloadWholeConfig,
        std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK},
        std::set<StatusCode> requiredExecuteResults = {StatusCode::OK},
        std::set<StatusCode> allowedCreatePDResults = {StatusCode::OK},
        std::set<StatusCode> allowedExecuteResults = {StatusCode::OK}) {
        ConstructorEnabledModelManager manager;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        auto status = manager.loadConfig(configFilePath);
        ASSERT_TRUE(status.ok());

        // setup helper variables for managing threads
        const std::vector<float> requestData{1., 2., 3., 7., 5., 6., 4., 9., 10., 8.};
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
        std::unordered_map<StatusCode, std::atomic<uint64_t>> executePipelineRetCodesCounters;
        for (uint i = 0; i != static_cast<uint>(StatusCode::STATUS_CODE_END); ++i) {
            createPipelineRetCodesCounters[static_cast<StatusCode>(i)] = 0;
        }
        for (uint i = 0; i != static_cast<uint>(StatusCode::STATUS_CODE_END); ++i) {
            executePipelineRetCodesCounters[static_cast<StatusCode>(i)] = 0;
        }
        // create worker threads
        std::vector<std::unique_ptr<std::thread>> workerThreads;
        for (uint i = 0; i < loadThreadCount; ++i) {
            workerThreads.emplace_back(std::make_unique<std::thread>(
                [this,
                    &futureStartSignals,
                    &futureStopSignals,
                    &manager,
                    &requestData,
                    &requiredCreatePDResults,
                    &requiredExecuteResults,
                    &allowedCreatePDResults,
                    &allowedExecuteResults,
                    &createPipelineRetCodesCounters,
                    &executePipelineRetCodesCounters,
                    i]() {
                    this->triggerPredictInALoop(futureStartSignals[i],
                        futureStopSignals[i],
                        manager,
                        PIPELINE_1_DUMMY_NAME,
                        requestData,
                        requiredCreatePDResults,
                        requiredExecuteResults,
                        allowedCreatePDResults,
                        allowedExecuteResults,
                        createPipelineRetCodesCounters,
                        executePipelineRetCodesCounters);
                }));
        }
        // start initial load
        std::for_each(startSignals.begin(), startSignals.end(), [](auto& startSignal) { startSignal.set_value(); });
        // sleep to allow all load threads to stress ovms during config changes
        std::this_thread::sleep_for(std::chrono::milliseconds(beforeConfigChangeLoadTimeMs));
        ((*this).*configChangeOperation)();
        if (reloadWholeConfig) {
            manager.loadConfig(configFilePath);
        } else {
            manager.updateConfigurationWithoutConfigFile();
        }
        // wait to work strictly on config operations after change
        std::this_thread::sleep_for(std::chrono::milliseconds(afterConfigChangeLoadTimeMs));
        std::for_each(stopSignals.begin(), stopSignals.end(), [](auto& stopSignal) { stopSignal.set_value(); });
        std::for_each(workerThreads.begin(), workerThreads.end(), [](auto& t) { t->join(); });

        for (auto& [retCode, counter] : createPipelineRetCodesCounters) {
            SPDLOG_TRACE("Create:[{}]={} -- {}", static_cast<uint>(retCode), counter, ovms::Status(retCode).string());
            if (requiredCreatePDResults.find(retCode) != requiredCreatePDResults.end()) {
                EXPECT_GT(counter, 0) << static_cast<uint>(retCode) << ":" << ovms::Status(retCode).string() << " did not occur. This may indicate fail or fail in test setup";
                continue;
            }
            if (counter == 0) {
                continue;
            }
            EXPECT_TRUE(allowedCreatePDResults.find(retCode) != allowedCreatePDResults.end()) << "Ret code:"
                                                                                              << static_cast<uint>(retCode) << " message: " << ovms::Status(retCode).string()
                                                                                              << " was not allowed in test but occured during load";
        }
        for (auto& [retCode, counter] : executePipelineRetCodesCounters) {
            SPDLOG_TRACE("Execute:[{}]={} -- {}", static_cast<uint>(retCode), counter, ovms::Status(retCode).string());
            if (requiredExecuteResults.find(retCode) != requiredExecuteResults.end()) {
                EXPECT_GT(counter, 0) << static_cast<uint>(retCode) << ":\"" << ovms::Status(retCode).string() << "\" did not occur. This may indicate fail or fail in test setup";
                continue;
            }
            if (counter == 0) {
                continue;
            }
            EXPECT_TRUE(allowedExecuteResults.find(retCode) != allowedExecuteResults.end()) << "Ret code:"
                                                                                            << static_cast<uint>(retCode) << " message: " << ovms::Status(retCode).string()
                                                                                            << " was not allowed in test but occured during load";
        }
    }
    void triggerPredictInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        ModelManager& manager,
        const std::string& pipelineName,
        const std::vector<float>& requestData,
        const std::set<StatusCode>& requiredCreatePDResults,
        const std::set<StatusCode>& requiredExecuteResults,
        const std::set<StatusCode>& allowedCreatePDResults,
        const std::set<StatusCode>& allowedExecuteResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& executePipelineRetCodesCounters) {
        startSignal.get();
        // stressIterationsCounter is additional safety measure
        auto stressIterationsCounter = stressIterationsLimit;
        while (stressIterationsCounter-- > 0) {
            auto futureWaitResult = stopSignal.wait_for(std::chrono::milliseconds(0));
            if (futureWaitResult == std::future_status::ready) {
                SPDLOG_INFO("Got stop signal. Ending Load");
                break;
            }
            std::unique_ptr<Pipeline> pipelinePtr;
            tensorflow::serving::PredictRequest request = preparePredictRequest(
                {{pipelineInputName,
                    std::tuple<ovms::shape_t, tensorflow::DataType>{{1, DUMMY_MODEL_INPUT_SIZE}, tensorflow::DataType::DT_FLOAT}}});
            auto& input = (*request.mutable_inputs())[pipelineInputName];
            std::vector<float> reqData = requestData;
            input.mutable_tensor_content()->assign((char*)reqData.data(), reqData.size() * sizeof(float));
            tensorflow::serving::PredictResponse response;
            auto createPipelineStatus = manager.createPipeline(pipelinePtr, pipelineName, &request, &response);
            createPipelineRetCodesCounters[createPipelineStatus.getCode()]++;

            // then we need to make sure that all expected statuses happened and still accept
            // some that could happen but we may not hit them
            EXPECT_TRUE((requiredCreatePDResults.find(createPipelineStatus.getCode()) != requiredCreatePDResults.end()) ||
                        (allowedCreatePDResults.find(createPipelineStatus.getCode()) != allowedCreatePDResults.end()))
                << createPipelineStatus.string() << "\n";
            if (!createPipelineStatus.ok()) {
                continue;
            }
            ovms::Status executePipelineStatus = StatusCode::UNKNOWN_ERROR;
            executePipelineStatus = pipelinePtr->execute();
            executePipelineRetCodesCounters[executePipelineStatus.getCode()]++;
            EXPECT_TRUE((requiredExecuteResults.find(executePipelineStatus.getCode()) != requiredExecuteResults.end()) ||
                        (allowedExecuteResults.find(executePipelineStatus.getCode()) != allowedExecuteResults.end()))
                << executePipelineStatus.string() << "\n";
            if (executePipelineStatus.ok()) {
                checkDummyResponse(pipelineOutputName, requestData, request, response, 1);
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
        for (auto& [retCode, counter] : executePipelineRetCodesCounters) {
            if (counter > 0) {
                SPDLOG_DEBUG("Execute:[{}]={}:{}", static_cast<uint>(retCode), ovms::Status(retCode).string(), counter);
            }
        }
        EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
        std::stringstream ss;
        ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
        SPDLOG_INFO(ss.str());
    }
};

TEST_F(StressPipelineConfigChanges, AddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                            // we just need to have all model versions rechecked
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK};   // we expect full continuouity of operation
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, RemoveDefaultVersionDuringPredictLoad) {
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};            // we hit when all config changes finish to propagate
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK,  // the model did not unload yet
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,               // we hit default version which is unloaded already but default is not changed yet
        StatusCode::MODEL_VERSION_MISSING};                         // there is no default version since all are either not loaded properly or retired
    // if we do not hit different execute results than OK it means that the stress was not really stress
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    // we need whole config reload since there is no other way to dispose
    // all model versions different than removing model from config
    bool performWholeConfigReload = true;
    performStressTest(&StressPipelineConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, ChangeToShapeAutoDuringPredictionLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK};   // we expect full continuouity of operation
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::changeToAutoShape,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, RemovePipelineDefinitionDuringLoad) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE};         // we expect to stop creating pipelines
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::removePipelineDefinition,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, ChangedPipelineConnectionName) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK};   // we expect full continuouity of operation
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::changeConnectionName,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, AddedNewPipeline) {
    bool performWholeConfigReload = true;
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK};   // we expect full continuouity of operation
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::addNewPipeline,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
TEST_F(StressPipelineConfigChanges, RetireSpecificVersionUsed) {
    // we declare specific version used (1) and latest model version policy with count=1
    // then we add version 2 causing previous default to be retired
    SetUpConfig(stressTestPipelineOneDummyConfigSpecificVersionUsed);
    bool performWholeConfigReload = false;
    std::set<StatusCode> requiredCreatePDResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET};             // we hit when all config changes finish to propagate
    std::set<StatusCode> requiredExecuteResults = {StatusCode::OK,
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};  // version is retired but pipeline not invalidated yet
    std::set<StatusCode> allowedCreatePDResults = {};
    std::set<StatusCode> allowedExecuteResults = {};
    performStressTest(&StressPipelineConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredCreatePDResults,
        requiredExecuteResults,
        allowedCreatePDResults,
        allowedExecuteResults);
}
