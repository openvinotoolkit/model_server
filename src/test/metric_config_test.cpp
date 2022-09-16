
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

#include "../config.hpp"
#include "../metric_config.hpp"
#include "../metric_registry.hpp"
#include "../model_metric_reporter.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::_;
using testing::Return;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"

class PublicMetricConfig : public MetricConfig {
public:
    friend class MetricConfig;
    const std::unordered_set<std::string>& getEnabledFamiliesList() const {
        return this->enabledFamiliesList;
    }
};

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
        char* n_argv[] = {"ovms", "--model_path", "/path/to/model", "--model_name", "some_name", "--rest_port", "8080"};
        int arg_count = 9;
        static bool parseOnce = [&arg_count, &n_argv]() {
            ovms::Config::instance().parse(arg_count, n_argv);
            return true;
        }();

        if (parseOnce) {
            // Prepare manager
            modelPath = directoryPath + "/dummy/";
            configFilePath = directoryPath + "/ovms_config.json";
        }
    }
};

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

TEST_F(MetricsConfigTest, DefaultValues) {
    SetUpConfig(modelDefaultConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto& metricConfig = static_cast<const PublicMetricConfig&>(manager.getMetricConfig());
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.getEnabledFamiliesList().size(), 0);
}

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
                "metrics_list": ["ovms_requests_success", "ovms_infer_req_queue_size"]
            }
        }
})";

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
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_success"));
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"));
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_fail"), false);
}

static const char* modelMetricsBadListConfig = R"(
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
                "metrics_list": ["ovms_request_success", "ovms_infer_req_queue_size"]
            }
        }
})";

TEST_F(MetricsConfigTest, BadFamilyConfig) {
    SetUpConfig(modelMetricsBadListConfig);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);

    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_EQ(status, StatusCode::INVALID_METRICS_FAMILY_NAME);
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
    ASSERT_TRUE(metricConfig2.isFamilyEnabled("ovms_requests_success"));
    ASSERT_TRUE(metricConfig2.isFamilyEnabled("ovms_infer_req_queue_size"));
    ASSERT_EQ(metricConfig2.isFamilyEnabled("ovms_requests_fail"), false);
}

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
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_success"));
    // Non default metric
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"), false);
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_fail"));
}

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

TEST_F(MetricsConfigTest, DISABLED_MetricsBadEndpoint) {
    SetUpConfig(modelMetricsBadEndpoint);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_EQ(status, StatusCode::INVALID_METRICS_ENDPOINT) << status.string();
}

static const char* MetricsNegativeAdditionalMember = R"(
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
                "something" : "else"
            }
        }
})";

TEST_F(MetricsConfigTest, MetricsNegativeAdditionalMember) {
    SetUpConfig(MetricsNegativeAdditionalMember);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_FALSE(status.ok());
}

static const char* MetricsNegativeBadMember = R"(
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
                "enabled" : true
            }
        }
})";

TEST_F(MetricsConfigTest, MetricsNegativeBadMember) {
    SetUpConfig(MetricsNegativeBadMember);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_FALSE(status.ok());
}

static const char* MetricsNegativeBadJson = R"(
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
                "enable" : {},
            }
        }
})";

TEST_F(MetricsConfigTest, MetricsNegativeBadJson) {
    SetUpConfig(MetricsNegativeBadJson);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_FALSE(status.ok());
}

static const char* MetricsNegativeBadType = R"(
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
                "enable" : 1,
            }
        }
})";

TEST_F(MetricsConfigTest, MetricsNegativeBadType) {
    SetUpConfig(MetricsNegativeBadType);
    std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    ConstructorEnabledModelManager manager;

    auto status = manager.loadConfig(configFilePath);
    ASSERT_FALSE(status.ok());
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

    metricConfig.setDefaultMetricsTo(true);
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

class MetricsCli : public ::testing::Test {
public:
    // 1024-65536
    uint64_t restPort = 10000;
};

TEST_F(MetricsCli, DefaultCliReading) {
    MetricConfig metricConfig;
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_success"), false);
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"), false);
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_fail"), false);

    auto status = metricConfig.loadFromCLIString(true, "ovms_requests_success, ovms_requests_fail", this->restPort);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_success"));
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"), false);
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_fail"));
}

TEST_F(MetricsCli, WorkingCliReading) {
    MetricConfig metricConfig;
    auto status = metricConfig.loadFromCLIString(true, "ovms_requests_success, ovms_infer_req_queue_size", this->restPort);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_success"));
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"));
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_fail"), false);
}

TEST_F(MetricsCli, DefaultEmptyList) {
    PublicMetricConfig metricConfig;
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.getEnabledFamiliesList().size(), 0);

    auto status = metricConfig.loadFromCLIString(true, "", this->restPort);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_success"));
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"), false);
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_fail"));
}

TEST_F(MetricsCli, BadCliReading) {
    PublicMetricConfig metricConfig;
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");

    auto status = metricConfig.loadFromCLIString(true, "badrequest_success_grpc_predict, $$$_fail_rest_model_ready", this->restPort);

    ASSERT_EQ(status, StatusCode::INVALID_METRICS_FAMILY_NAME);
    ASSERT_EQ(metricConfig.metricsEnabled, true);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.getEnabledFamiliesList().size(), 0);
}

TEST_F(MetricsCli, DisabledMetrics) {
    MetricConfig metricConfig;
    auto status = metricConfig.loadFromCLIString(false, "ovms_infer_req_queue_size, ovms_requests_fail", this->restPort);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_success"), false);
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"));
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_fail"));
}

TEST_F(MetricsCli, MetricsDisabledCliRestPort) {
    MetricConfig metricConfig;
    auto status = metricConfig.loadFromCLIString(false, "ovms_infer_req_queue_size, ovms_requests_fail", 0);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(metricConfig.metricsEnabled, false);
    ASSERT_EQ(metricConfig.endpointsPath, "/metrics");
    ASSERT_EQ(metricConfig.isFamilyEnabled("ovms_requests_success"), false);
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_infer_req_queue_size"));
    ASSERT_TRUE(metricConfig.isFamilyEnabled("ovms_requests_fail"));
}

TEST_F(MetricsCli, MetricsEnabledCliRestPort) {
    MetricConfig metricConfig;
    auto status = metricConfig.loadFromCLIString(true, "ovms_infer_req_queue_size, ovms_requests_fail", 0);

    ASSERT_EQ(status, StatusCode::CONFIG_FILE_INVALID);
}

TEST_F(MetricsCli, MetricsEnabledCliRestPortDefault) {
    MetricConfig metricConfig;
    auto status = metricConfig.loadFromCLIString(true, "ovms_infer_req_queue_size, ovms_requests_fail");

    ASSERT_EQ(status, StatusCode::CONFIG_FILE_INVALID);
}

#pragma GCC diagnostic pop
