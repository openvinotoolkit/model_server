
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

#include "../get_model_metadata_impl.hpp"
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

    const std::vector<float> requestData{1., 2., 3., 7., 5., 6., 4., 9., 10., 8.};

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
        ConstructorEnabledModelManager manager;
        createConfigFileWithContent(ovmsConfig, configFilePath);
        auto status = manager.loadConfig(configFilePath);
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
                    &triggerLoadInALoop,
                    &futureStartSignals,
                    &futureStopSignals,
                    &manager,
                    &requiredLoadResults,
                    &allowedLoadResults,
                    &createPipelineRetCodesCounters,
                    i]() {
                    ((*this).*triggerLoadInALoop)(futureStartSignals[i],
                        futureStopSignals[i],
                        manager,
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
    bool isMetadataResponseCorrect(tensorflow::serving::GetModelMetadataResponse& response) {
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

        bool inputsSizeCorrect{inputs.size() == 1};
        EXPECT_TRUE(inputsSizeCorrect) << "Expected: " << 1 << " actual: " << inputs.size();
        bool outputsSizeCorrect{outputs.size() == 1};
        EXPECT_TRUE(outputsSizeCorrect) << "Expected: " << 1 << " actual: " << outputs.size();
        if (!inputsSizeCorrect || !outputsSizeCorrect) {
            return false;
        }
        bool inputNameExist{inputs.find(pipelineInputName.c_str()) != inputs.end()};
        EXPECT_TRUE(inputNameExist);
        bool outputNameExist{outputs.find(pipelineOutputName.c_str()) != outputs.end()};
        EXPECT_TRUE(outputNameExist);
        if (!inputNameExist || !outputNameExist) {
            return false;
        }
        bool inputNameCorrect{inputs.at(pipelineInputName.c_str()).name() == pipelineInputName};
        EXPECT_TRUE(inputNameCorrect);
        bool outputNameCorrect{outputs.at(pipelineOutputName.c_str()).name() == pipelineOutputName};
        EXPECT_TRUE(outputNameCorrect);
        if (!inputNameCorrect || !outputNameCorrect) {
            return false;
        }
        bool inputTypeCorrect{inputs.at(pipelineInputName.c_str()).dtype() == tensorflow::DT_FLOAT};
        EXPECT_TRUE(inputTypeCorrect);
        bool outputTypeCorrect{outputs.at(pipelineOutputName.c_str()).dtype() == tensorflow::DT_FLOAT};
        EXPECT_TRUE(outputTypeCorrect);
        if (!inputTypeCorrect || !outputTypeCorrect) {
            return false;
        }
        auto isShape = [](
                           const tensorflow::TensorShapeProto& actual,
                           const std::vector<size_t>&& expected) -> bool {
            if (static_cast<unsigned int>(actual.dim_size()) != expected.size()) {
                return false;
            }
            for (int i = 0; i < actual.dim_size(); i++) {
                if (static_cast<unsigned int>(actual.dim(i).size()) != expected[i]) {
                    return false;
                }
            }
            return true;
        };
        bool inputShapeCorrect{isShape(
            inputs.at(pipelineInputName.c_str()).tensor_shape(),
            {1, 10})};
        EXPECT_TRUE(inputShapeCorrect);
        bool outputShapeCorrect{isShape(
            outputs.at(pipelineOutputName.c_str()).tensor_shape(),
            {1, 10})};
        EXPECT_TRUE(outputShapeCorrect);
        if (!inputShapeCorrect || !outputShapeCorrect) {
            return false;
        }
        return true;
    }
    void triggerGetPipelineMetadataInALoop(
        std::future<void>& startSignal,
        std::future<void>& stopSignal,
        ModelManager& manager,
        const std::set<StatusCode>& requiredLoadResults,
        const std::set<StatusCode>& allowedLoadResults,
        std::unordered_map<StatusCode, std::atomic<uint64_t>>& createPipelineRetCodesCounters) {
        tensorflow::serving::GetModelMetadataRequest request;
        tensorflow::serving::GetModelMetadataResponse response;
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
            status = ovms::GetModelMetadataImpl::getModelStatus(&request, &response, manager);
            createPipelineRetCodesCounters[status.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(status.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(status.getCode()) != allowedLoadResults.end()))
                << status.string() << "\n";
            if (!status.ok()) {
                continue;
            }
            // Check response if correct
            EXPECT_TRUE(isMetadataResponseCorrect(response));
            if (::testing::Test::HasFailure()) {
                SPDLOG_INFO("Earlier fail detected. Stopping execution");
                break;
            }
        }
    }
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
            executePipelineStatus = pipelinePtr->execute();
            createPipelineRetCodesCounters[executePipelineStatus.getCode()]++;
            EXPECT_TRUE((requiredLoadResults.find(executePipelineStatus.getCode()) != requiredLoadResults.end()) ||
                        (allowedLoadResults.find(executePipelineStatus.getCode()) != allowedLoadResults.end()))
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
        EXPECT_GT(stressIterationsCounter, 0) << "Reaching 0 means that we might not test enough \"after config change\" operation was applied";
        std::stringstream ss;
        ss << "Executed: " << stressIterationsLimit - stressIterationsCounter << " inferences by thread id: " << std::this_thread::get_id() << std::endl;
        SPDLOG_INFO(ss.str());
    }
};

TEST_F(StressPipelineConfigChanges, AddNewVersionDuringPredictLoad) {
    bool performWholeConfigReload = false;                        // we just need to have all model versions rechecked
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK};  // we expect full continuouity of operation
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerPredictInALoop,
        &StressPipelineConfigChanges::defaultVersionRemove,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::triggerPredictInALoop,
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
        &StressPipelineConfigChanges::defaultVersionRemove,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
TEST_F(StressPipelineConfigChanges, RemoveDefaultVersionDuringGetMetadataLoad) {
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,  // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,    // we hit default version which is unloaded already but default is not changed yet
        StatusCode::MODEL_MISSING};                      // model instance not found during metadata request
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
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE,  // when pipeline is retired
        StatusCode::MODEL_VERSION_NOT_LOADED_YET};           // we do not wait for models durign get metadata
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
    std::set<StatusCode> requiredLoadResults = {StatusCode::OK,  // we expect full continuouity of operation
        StatusCode::MODEL_VERSION_NOT_LOADED_YET};               // we do not wait for models durign get metadata
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
        StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET,          // we hit when all config changes finish to propagate
        StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE};           // version is retired but pipeline not invalidated yet
    std::set<StatusCode> allowedLoadResults = {};
    performStressTest(
        &StressPipelineConfigChanges::triggerGetPipelineMetadataInALoop,
        &StressPipelineConfigChanges::retireSpecificVersionUsed,
        performWholeConfigReload,
        requiredLoadResults,
        allowedLoadResults);
}
