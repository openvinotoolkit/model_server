
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

#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "test_utils.hpp"
#include "light_test_utils.hpp"
#include "platform_utils.hpp"
#include "test_with_temp_dir.hpp"

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

static const char* modelStatefulChangedConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"latest": {"num_versions":1}},
                "nireq": 100,
                "stateful": true,
                "low_latency_transformation": true,
                "max_sequence_number": 1000,
                "shape": {"b": "(1,10) "}
            }
        }
    ]
})";

class StatefulConfigTest : public TestWithTempDir {
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
        std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/dummy"), modelPath, std::filesystem::copy_options::recursive);
    }
};

TEST_F(StatefulConfigTest, DefaultValues) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto modelConfig = modelInstance->getModelConfig();

    auto is = modelConfig.isLowLatencyTransformationUsed();
    ASSERT_EQ(is, false);
    is = modelConfig.isStateful();
    ASSERT_EQ(is, false);
    auto maxSequenceNumber = modelConfig.getMaxSequenceNumber();
    ASSERT_EQ(maxSequenceNumber, 500);
    auto idleSequenceCleanup = modelConfig.getIdleSequenceCleanup();
    ASSERT_EQ(idleSequenceCleanup, true);
}

TEST_F(StatefulConfigTest, ChangedValues) {
    SetUpConfig(modelStatefulChangedConfig);
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto modelConfig = modelInstance->getModelConfig();

    auto is = modelConfig.isLowLatencyTransformationUsed();
    ASSERT_EQ(is, true);
    is = modelConfig.isStateful();
    ASSERT_EQ(is, true);
    auto maxSequenceNumber = modelConfig.getMaxSequenceNumber();
    ASSERT_EQ(maxSequenceNumber, 1000);
    auto idleSequenceCleanup = modelConfig.getIdleSequenceCleanup();
    ASSERT_EQ(idleSequenceCleanup, true);
}
