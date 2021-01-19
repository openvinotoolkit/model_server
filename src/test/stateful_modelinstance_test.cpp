//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "../processing_spec.hpp"
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
    std::pair<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>> sequenceId;
    std::pair<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>> sequenceControlStart;

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
        SetUpConfig(modelStatefulConfig);
        std::filesystem::copy("/ovms/src/test/dummy", modelPath, std::filesystem::copy_options::recursive);
        modelInput = {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}};
    }

    void TearDown() override {
        TestWithTempDir::TearDown();
        modelInput.clear();
    }
};

class MockedStatefulModelInstance : public ovms::StatefulModelInstance {
public:
    MockedStatefulModelInstance(const std::string& name, ovms::model_version_t version) :
        StatefulModelInstance(name, version) {}

    void injectSequence(uint64_t sequenceId, ovms::model_memory_state_t state) {
        getSequenceManager().addSequence(sequenceId);
        getSequenceManager().updateSequenceMemoryState(sequenceId, state);
    }
};

TEST_F(StatefulModelInstance, positiveValidate) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    auto modelInstance = manager.findModelInstance(dummyModelName);
    uint64_t seqId = 1;
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, SEQUENCE_START);

    status = modelInstance->validate(&request, &spec);
    ASSERT_TRUE(status.ok());
}

TEST_F(StatefulModelInstance, PreprocessingFirstRequest) {
    // Prepare model instance and processing spec
    MockedStatefulModelInstance modelInstance("model", 1);
    uint32_t sequenceControlInput = SEQUENCE_START;
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec sequenceProcessingSpec(sequenceControlInput, sequenceId);

    // Prepare states blob desc
    std::vector<size_t> shape{ 1, 10 };
    size_t elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    const Precision precision{ Precision::FP32 };
    const Layout layout{ Layout::NC };
    const TensorDesc desc{ precision, shape, layout };

    // Prepare default state blob
    std::vector<float> defaultState(elementsCount);
    std::iota(defaultState.begin(), defaultState.end(), 0);
    Blob::Ptr defaultBlob = make_shared_blob<float>(desc, defaultState.data());

    // Prepare new state blob
    std::vector<float> currentState(elementsCount);
    std::iota(currentState.begin(), currentState.end(), 10);
    Blob::Ptr currentBlob = make_shared_blob<float>(desc, currentState.data());

    // Initialize InferRequest with current state and default state
    std::shared_ptr<IInferRequest> iireqPtr = std::make_shared<MockIInferRequestStateful>("state", currentBlob, defaultBlob);
    InferRequest inferRequest(iireqPtr);

    // Check if InferRequest has been initialized properly
    const ovms::model_memory_state_t& irMemoryState = inferRequest.QueryState();
    EXPECT_EQ(irMemoryState.size(), 1);
    EXPECT_EQ(irMemoryState[0].GetName(), "state");

    InferenceEngine::Blob::Ptr stateCloneBlob = nullptr;
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);

    std::vector<float> currentBlobIrData;
    currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
    EXPECT_EQ(currentBlobIrData, currentState);

    // Perform preprocessing (load state from sequence to infer request)
    modelInstance.preInferenceProcessing(inferRequest, sequenceProcessingSpec);

    // Check if InferRequest memory state has been reset to default
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
    currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
    EXPECT_EQ(currentBlobIrData, defaultState);

}  // namespace
