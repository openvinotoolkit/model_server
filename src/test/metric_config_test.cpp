
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
#include "../metric_registry.hpp"
#include "../model_metric_reporter.hpp"
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
                "metrics_list": ["request_success_grpc_predict", "request_fail_rest_model_ready"],
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

static const char* modelMetricsBadEndpoint = R"(
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
                "endpoint_path": "/new..metrics"
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
    }
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        modelPath = directoryPath + "/dummy/";
        configFilePath = directoryPath + "/ovms_config.json";
    }
};

TEST_F(MetricsConfigTest, DefaultValues) {
    SetUpConfig(modelDefaultConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    const auto& metricConfig = manager.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.requestSuccessGrpcPredict, false);
    ASSERT_EQ(metricConfig.requestFailRestModelReady, false);
}

TEST_F(MetricsConfigTest, ChangedValues) {
    SetUpConfig(modelMetricsChangedConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    const auto& metricConfig = manager.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, true);

    // ASSERT_EQ(metricConfig.endpointsPath, "/newmetrics");
    ASSERT_EQ(metricConfig.requestSuccessGrpcPredict, true);
    ASSERT_EQ(metricConfig.requestFailRestModelReady, true);
    ASSERT_EQ(metricConfig.requestSuccessGrpcModelReady, false);
}

TEST_F(MetricsConfigTest, InitOnce) {
    SetUpConfig(modelMetricsChangedConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    TearDown();
    SetUp();
    SetUpConfig(modelDefaultConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    status = manager.loadConfig(configFilePath);

    const auto& metricConfig2 = manager.getMetricConfig();
    ASSERT_EQ(metricConfig2.metricsEnabled, true);

    // ASSERT_EQ(metricConfig2.endpointsPath, "/newmetrics");
    ASSERT_EQ(metricConfig2.requestSuccessGrpcPredict, true);
    ASSERT_EQ(metricConfig2.requestFailRestModelReady, true);
    ASSERT_EQ(metricConfig2.requestSuccessGrpcModelReady, false);
}

TEST_F(MetricsConfigTest, MetricsAllEnabledTest) {
    SetUpConfig(modelMetricsAllEnabledConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    const auto& metricConfig = manager.getMetricConfig();
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.requestFailGrpcModelInfer, true);
    ASSERT_EQ(metricConfig.requestFailRestModelInfer, true);
    ASSERT_EQ(metricConfig.requestFailRestModelMetadata, true);
}

TEST_F(MetricsConfigTest, DISABLED_MetricsBadEndpoint) {
    SetUpConfig(modelMetricsBadEndpoint);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_EQ(status, StatusCode::INVALID_METRICS_ENDPOINT) << status.string();
}

class ModelMetricReporterTest : public ::testing::Test {};

TEST_F(ModelMetricReporterTest, MetricReporterConstructorTest) {
    MetricRegistry registry;
    ModelMetricReporter reporter1(nullptr, nullptr, "example_pipeline_name", 1);
    ASSERT_EQ(reporter1.requestFailGrpcGetModelMetadata, nullptr);

    ModelMetricReporter reporter2(nullptr, &registry, "example_pipeline_name", 1);
    auto metrics = registry.collect();
    ASSERT_EQ(metrics, "");
    ASSERT_EQ(reporter2.requestFailGrpcGetModelMetadata, nullptr);

    MetricConfig metricConfig;
    ModelMetricReporter reporter3(&metricConfig, &registry, "example_pipeline_name", 1);
    metrics = registry.collect();
    ASSERT_EQ(metrics, "");
    ASSERT_EQ(reporter3.requestFailGrpcGetModelMetadata, nullptr);

    metricConfig.setAllMetricsTo(true);
    ModelMetricReporter reporter4(&metricConfig, &registry, "example_pipeline_name", 1);
    metrics = registry.collect();
    ASSERT_EQ(metrics, "");
    ASSERT_EQ(reporter4.requestFailGrpcGetModelMetadata, nullptr);

    metricConfig.metricsEnabled = true;
    ModelMetricReporter reporter5(&metricConfig, &registry, "example_pipeline_name", 1);
    metrics = registry.collect();
    ASSERT_NE(metrics, "");
    ASSERT_TRUE(reporter5.requestFailGrpcGetModelMetadata != nullptr);
}
