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
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../get_model_metadata_impl.hpp"
#include "../statefulmodelinstance.hpp"
#include "test_utils.hpp"

using testing::Return;

namespace {
static const char* modelStatefulConfig = R"(
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
                "sequence_timeout_seconds": 120,
                "max_sequence_number": 1000,
                "shape": {"b": "(1,10) "}
            }
        }
    ]
})";

constexpr const char* DUMMY_MODEL_INPUT_NAME = "b";
class StatefulModelInstance : public TestWithTempDir {
public:
    std::string configFilePath;
    std::string ovmsConfig;
    std::string modelPath;
    std::string dummyModelName;
    inputs_info_t modelInput;
    std::pair sequenceId;
    std::pair sequenceControlStart;

    void SetUpConfig(const std::string& configContent) {
        ovmsConfig = configContent;
        dummyModelName = "dummy";
        const std::string modelPathToReplace{ "/ovms/src/test/dummy" };
        ovmsConfig.replace(ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), modelPath);
        configFilePath = directoryPath + "/ovms_config.json";
    }
    void SetUp() override {
        TestWithTempDir::SetUp();
        // Prepare manager
        modelPath = directoryPath + "/dummy/";
        SetUpConfig(modelStatefulConfig);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
        modelInput = { {DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{ {1, 10}, tensorflow::DataType::DT_FLOAT}} };

        sequenceInput = std::make_pair("sequence_id", std::tuple<ovms::shape_t, tensorflow::DataType>{ {1, 1}, tensorflow::DataType::DT::UINT64});
        sequenceControlStart = std::make_pair("sequence_control_input", std::tuple<ovms::shape_t, tensorflow::DataType>{ {1, 1}, tensorflow::DataType::DT::UINT32});
    }

    void TearDown() override {
        TestWithTempDir::TearDown();
        modelInput.clear();
    }
};

TEST_F(StatefulModelInstance, positiveValidate) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());

    auto modelInstance = manager.findModelInstance(dummyModelName);

    modelInput.insert(sequenceControlStart);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);

    status = modelInstance->validate(&request);
    ASSERT_TRUE(status.ok());
}
}
