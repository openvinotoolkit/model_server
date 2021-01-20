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
#include <typeinfo>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../get_model_metadata_impl.hpp"
#include "../ov_utils.hpp"
#include "../processing_spec.hpp"
#include "../statefulmodelinstance.hpp"
#include "sequence_test_utils.hpp"
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
class StatefulModelInstanceTempDir : public TestWithTempDir {
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
        const std::string modelPathToReplace{"/ovms/src/test/dummy"};
        ovmsConfig.replace(ovmsConfig.find(modelPathToReplace), modelPathToReplace.size(), modelPath);
        configFilePath = directoryPath + "/ovms_config.json";
        dummyModelName = "dummy";
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

class StatefulModelInstanceInputValidation : public ::testing::Test {
public:
    inputs_info_t modelInput;
    std::pair<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>> sequenceId;
    std::pair<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>> sequenceControlStart;

    void SetUp() override {
        modelInput = {};
    }

    void TearDown() override {
        modelInput.clear();
    }
};

class MockedValidateStatefulModelInstance : public ovms::StatefulModelInstance {
public:
    MockedValidateStatefulModelInstance(const std::string& name, ovms::model_version_t version) :
        StatefulModelInstance(name, version) {}

    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request, ovms::ProcessingSpec& processingSpecPtr) {
        return validate(request, processingSpecPtr);
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

class StatefulModelInstanceTest : public ::testing::Test {
public:
    std::shared_ptr<MockedStatefulModelInstance> modelInstance;
    std::vector<size_t> shape;
    Blob::Ptr defaultBlob;
    Blob::Ptr currentBlob;
    Blob::Ptr newBlob;

    std::vector<float> defaultState;
    std::vector<float> currentState;
    std::vector<float> newState;

    size_t elementsCount;

    void SetUp() override {
        modelInstance = std::make_shared<MockedStatefulModelInstance>("model", 1);
        // Prepare states blob desc
        shape = std::vector<size_t>{1, 10};
        elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        const Precision precision{Precision::FP32};
        const Layout layout{Layout::NC};
        const TensorDesc desc{precision, shape, layout};

        // Prepare default state blob
        defaultState = std::vector<float>(elementsCount);
        std::iota(defaultState.begin(), defaultState.end(), 0);
        defaultBlob = make_shared_blob<float>(desc, defaultState.data());

        // Prepare new state blob
        currentState = std::vector<float>(elementsCount);
        std::iota(currentState.begin(), currentState.end(), 10);
        currentBlob = make_shared_blob<float>(desc, currentState.data());

        newState = std::vector<float>(elementsCount);
        std::iota(newState.begin(), newState.end(), 10);
        newBlob = make_shared_blob<float>(desc, newState.data());
    }
};

TEST_F(StatefulModelInstanceTempDir, modelInstanceFactory) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    ASSERT_TRUE(typeid(*modelInstance) == typeid(ovms::StatefulModelInstance));
}

TEST_F(StatefulModelInstanceInputValidation, positiveValidate) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    uint64_t seqId = 1;
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, SEQUENCE_START);

    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());

    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, SEQUENCE_END);

    status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());

    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, NO_CONTROL_INPUT);

    status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());
}

TEST_F(StatefulModelInstanceInputValidation, missingSeqId) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, SEQUENCE_END);

    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongSeqIdEnd) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, SEQUENCE_END);

    uint64_t seqId = 0;
    setRequestSequenceId(&request, seqId);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongSeqIdNoControl) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, NO_CONTROL_INPUT);

    uint64_t seqId = 0;
    setRequestSequenceId(&request, seqId);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongProtoKeywords) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    auto& input = (*request.mutable_inputs())["sequenceid"];
    input.add_uint64_val(12);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, badControlInput) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    request = preparePredictRequest(modelInput);
    auto& input = (*request.mutable_inputs())["sequence_control_input"];
    input.add_uint32_val(999);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::INVALID_SEQUENCE_CONTROL_INPUT);
}

TEST_F(StatefulModelInstanceInputValidation, invalidProtoTypes) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::ProcessingSpec spec = ovms::ProcessingSpec();
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    auto& input = (*request.mutable_inputs())["sequence_id"];
    input.add_uint32_val(12);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_BAD_TYPE);

    request = preparePredictRequest(modelInput);
    input = (*request.mutable_inputs())["sequence_control_input"];
    input.add_uint64_val(1);
    status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE);
}

TEST_F(StatefulModelInstanceTest, PreprocessingFirstRequest) {
    // Prepare model instance and processing spec
    uint32_t sequenceControlInput = SEQUENCE_START;
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec sequenceProcessingSpec(sequenceControlInput, sequenceId);

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
    modelInstance->preInferenceProcessing(inferRequest, sequenceProcessingSpec);

    // Check if InferRequest memory state has been reset to default
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
    currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
    EXPECT_EQ(currentBlobIrData, defaultState);
}

TEST_F(StatefulModelInstanceTest, PreprocessingIntermediateRequest) {
    for (uint32_t sequenceControlInput : {NO_CONTROL_INPUT, SEQUENCE_END}) {
        // Prepare model instance and processing spec
        uint64_t sequenceId = 42;
        ovms::SequenceProcessingSpec sequenceProcessingSpec(sequenceControlInput, sequenceId);

        // Initialize InferRequest with default state as both default and current states
        std::shared_ptr<IInferRequest> iireqPtr = std::make_shared<MockIInferRequestStateful>("state", defaultBlob, defaultBlob);
        InferRequest inferRequest(iireqPtr);

        // Check if InferRequest has been initialized properly
        const ovms::model_memory_state_t& irMemoryState = inferRequest.QueryState();
        EXPECT_EQ(irMemoryState.size(), 1);
        EXPECT_EQ(irMemoryState[0].GetName(), "state");

        InferenceEngine::Blob::Ptr stateCloneBlob = nullptr;
        EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);

        std::vector<float> currentBlobIrData;
        currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
        EXPECT_EQ(currentBlobIrData, defaultState);

        // Inject sequence with newState as the last state written to sequence memory state
        ovms::model_memory_state_t memoryState;
        addState(memoryState, "state", shape, newState);
        modelInstance->injectSequence(sequenceId, memoryState);

        // Perform preprocessing (load state from sequence to infer request)
        modelInstance->preInferenceProcessing(inferRequest, sequenceProcessingSpec);

        // Check if InferRequest memory state has been updated to sequence memory state
        EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
        currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
        EXPECT_EQ(currentBlobIrData, newState);
    }
}

TEST_F(StatefulModelInstanceTest, PostprocessingLastRequest) {
    // Prepare model instance and processing spec
    uint32_t sequenceControlInput = SEQUENCE_END;
    uint64_t sequenceId = 42;
    ovms::SequenceProcessingSpec sequenceProcessingSpec(sequenceControlInput, sequenceId);

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

    tensorflow::serving::PredictResponse response;

    modelInstance->postInferenceProcessing(&response, inferRequest, sequenceProcessingSpec);

    auto it = response.mutable_outputs()->find("sequence_id");
    EXPECT_FALSE(it == response.mutable_outputs()->end());
    auto& output = (*response.mutable_outputs())["sequence_id"];
    EXPECT_EQ(output.uint64_val_size(), 1);
    EXPECT_EQ(output.uint64_val(0), sequenceId);

    // Check if InferRequest memory state has been reset to default
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
    currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
    EXPECT_EQ(currentBlobIrData, defaultState);
}

TEST_F(StatefulModelInstanceTest, PostprocessingStartAndNoControl) {
    for (uint32_t sequenceControlInput : {NO_CONTROL_INPUT, SEQUENCE_START}) {
        // Prepare model instance and processing spec
        uint64_t sequenceId = 33;
        ovms::SequenceProcessingSpec sequenceProcessingSpec(sequenceControlInput, sequenceId);

        // Initialize InferRequest with default state as both default and current states
        std::shared_ptr<IInferRequest> iireqPtr = std::make_shared<MockIInferRequestStateful>("state", defaultBlob, defaultBlob);
        InferRequest inferRequest(iireqPtr);

        // Check if InferRequest has been initialized properly
        const ovms::model_memory_state_t& irMemoryState = inferRequest.QueryState();
        EXPECT_EQ(irMemoryState.size(), 1);
        EXPECT_EQ(irMemoryState[0].GetName(), "state");

        InferenceEngine::Blob::Ptr stateCloneBlob = nullptr;
        EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);

        std::vector<float> currentBlobIrData;
        currentBlobIrData.assign((float*)stateCloneBlob->buffer(), ((float*)stateCloneBlob->buffer()) + elementsCount);
        EXPECT_EQ(currentBlobIrData, defaultState);

        // Inject sequence with newState as the last state written to sequence memory state
        ovms::model_memory_state_t memoryState;
        addState(memoryState, "state", shape, newState);
        modelInstance->injectSequence(sequenceId, memoryState);

        // Sanity check for new state
        const ovms::sequence_memory_state_t& currentSequenceMemoryState = modelInstance->getSequenceManager().getSequenceMemoryState(sequenceProcessingSpec.sequenceId);
        EXPECT_TRUE(currentSequenceMemoryState.count("state"));
        InferenceEngine::Blob::Ptr sanityBlob = currentSequenceMemoryState.at("state");
        std::vector<float> sanityBlobIrData;
        sanityBlobIrData.assign((float*)sanityBlob->buffer(), ((float*)sanityBlob->buffer()) + elementsCount);
        EXPECT_EQ(sanityBlobIrData, newState);

        tensorflow::serving::PredictResponse response;

        modelInstance->postInferenceProcessing(&response, inferRequest, sequenceProcessingSpec);

        // Check if sequence memory state is the same as InferRequest memory state
        const ovms::sequence_memory_state_t& updatedSequenceMemoryState = modelInstance->getSequenceManager().getSequenceMemoryState(sequenceProcessingSpec.sequenceId);
        EXPECT_TRUE(updatedSequenceMemoryState.count("state"));
        InferenceEngine::Blob::Ptr chengedBlob = updatedSequenceMemoryState.at("state");
        std::vector<float> sequenceBlobIrData;
        sequenceBlobIrData.assign((float*)chengedBlob->buffer(), ((float*)chengedBlob->buffer()) + elementsCount);
        EXPECT_EQ(sequenceBlobIrData, defaultState);

        auto it = response.mutable_outputs()->find("sequence_id");
        EXPECT_FALSE(it == response.mutable_outputs()->end());

        auto& output = (*response.mutable_outputs())["sequence_id"];
        EXPECT_EQ(output.uint64_val_size(), 1);
        EXPECT_EQ(output.uint64_val(0), sequenceId);
    }
}

}  // namespace
