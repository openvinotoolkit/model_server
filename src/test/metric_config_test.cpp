
//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include "../metric_config.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::_;
using testing::Return;

static const char* modelDefaultConfig = R"(
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

static const char* modelMetricsChangedConfig = R"(
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
    "monitoring":
        {
            "metrics":
            {
                "enable" : true,
                "metrics_list": ["requestSuccessGrpcPredict", "requestFailRestModelStatus"],
                "endpoint_path": "/newmetrics" 
            }
        }
})";

static const char* modelMetricsAllEnabledConfig = R"(
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
    "monitoring":
        {
            "metrics":
            {
                "enable" : true
            }
        }
})";

class MetricsConfigTest : public TestWithTempDir {
public:
    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;
    std::string dummyModelName;

    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        dummyModelName = "dummy";
        const std::string modelPathToReplace{"/ovms/src/test/dummy"};
        ovmsConfig.replace(ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), modelPath);
        configFilePath = directoryPath + "/ovms_config.json";
    }
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        modelPath = directoryPath + "/dummy/";
        SetUpConfig(modelDefaultConfig);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    }
};

TEST_F(MetricsConfigTest, DefaultValues) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto modelConfig = modelInstance->getModelConfig();

    auto metricConfig = modelConfig.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.requestSuccessGrpcPredict, false);
    ASSERT_EQ(metricConfig.requestFailRestModelStatus, false);
}

TEST_F(MetricsConfigTest, ChangedValues) {
    SetUpConfig(modelMetricsChangedConfig);
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto modelConfig = modelInstance->getModelConfig();

    auto metricConfig = modelConfig.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/newmetrics");
    ASSERT_EQ(metricConfig.requestSuccessGrpcPredict, true);
    ASSERT_EQ(metricConfig.requestFailRestModelStatus, true);
    ASSERT_EQ(metricConfig.requestSuccessGrpcModelStatus, false);
}

TEST_F(MetricsConfigTest, MetricsAllEnabledTest) {
    SetUpConfig(modelMetricsAllEnabledConfig);
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto modelConfig = modelInstance->getModelConfig();

    auto metricConfig = modelConfig.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.requestFailGrpcModelInfer, true);
    ASSERT_EQ(metricConfig.requestFailRestModelInfer, true);
    ASSERT_EQ(metricConfig.requestFailRestModelMetadata, true);
}
