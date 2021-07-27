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
#include <future>
#include <thread>
#include <typeinfo>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "../deserialization.hpp"
#include "../executingstreamidguard.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../global_sequences_viewer.hpp"
#include "../ov_utils.hpp"
#include "../sequence_processing_spec.hpp"
#include "../serialization.hpp"
#include "../statefulmodelinstance.hpp"
#include "../timer.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "stateful_test_utils.hpp"
#include "test_utils.hpp"

using testing::Return;

namespace {
static bool testWarningPrinted = false;

enum SequenceTimeoutScenarios {
    WAIT_BEFORE_MANAGER_LOCKED,
    WAIT_BEFORE_SEQUENCE_LOCKED,
    WAIT_BEFORE_SEQUENCE_UNLOCKED,
    WAIT_AFTER_SEQUENCE_UNLOCKED,
    UNKNOWN
};

static const char* ToString(SequenceTimeoutScenarios scenario) {
    switch (scenario) {
    case WAIT_BEFORE_MANAGER_LOCKED:
        return "WAIT_BEFORE_MANAGER_LOCKED";
    case WAIT_BEFORE_SEQUENCE_LOCKED:
        return "WAIT_BEFORE_SEQUENCE_LOCKED";
    case WAIT_BEFORE_SEQUENCE_UNLOCKED:
        return "WAIT_BEFORE_SEQUENCE_UNLOCKED";
    case WAIT_AFTER_SEQUENCE_UNLOCKED:
        return "WAIT_AFTER_SEQUENCE_UNLOCKED";
    default:
        return "[ERROR] Unknown sequence timeout scenario type";
    }
}

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
    ovms::model_version_t modelVersion;
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
        modelVersion = 1;
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
    ovms::GlobalSequencesViewer sequencesViewer;
    MockedValidateStatefulModelInstance(const std::string& name, ovms::model_version_t version) :
        StatefulModelInstance(name, version, &sequencesViewer) {}

    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request, ovms::SequenceProcessingSpec& processingSpec) {
        return validate(request, processingSpec);
    }
};

class MockedStatefulModelInstance : public ovms::StatefulModelInstance {
    class MockedGlobalSequencesViewer : public ovms::GlobalSequencesViewer {
    public:
        ovms::Status removeIdleSequences() {
            return ovms::GlobalSequencesViewer::removeIdleSequences();
        }
    };

public:
    ovms::GlobalSequencesViewer sequencesViewer;
    std::unique_ptr<MockedSequenceManager> mockedSequenceManager = std::make_unique<MockedSequenceManager>(60, "dummy", 1);

    MockedStatefulModelInstance(const std::string& name, ovms::model_version_t version) :
        StatefulModelInstance(name, version, &sequencesViewer) {}

    const std::unique_ptr<MockedSequenceManager>& getMockedSequenceManager() const {
        return this->mockedSequenceManager;
    }

    MockedGlobalSequencesViewer* getSequencesViewer() {
        return static_cast<MockedGlobalSequencesViewer*>(this->globalSequencesViewer);
    }

    void injectSequence(uint64_t sequenceId, ovms::model_memory_state_t state) {
        ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, sequenceId);
        getMockedSequenceManager()->mockCreateSequence(spec);
        getMockedSequenceManager()->getSequence(sequenceId).updateMemoryState(state);
    }

    // This method must be kept up to date with StatefulModelInstance::infer for tests to function properly.
    ovms::Status infer(const tensorflow::serving::PredictRequest* requestProto,
        tensorflow::serving::PredictResponse* responseProto,
        std::unique_ptr<ovms::ModelInstanceUnloadGuard>& modelUnloadGuardPtr,
        std::future<void>* waitBeforeManagerLock,
        std::future<void>* waitBeforeSequenceLock,
        std::future<void>* waitBeforeSequenceUnlocked) {
        if (!testWarningPrinted) {
            std::cout << "[WARNING] This method must be kept up to date with StatefulModelInstance::infer for tests to function properly." << std::endl;
            testWarningPrinted = true;
        }
        ovms::Timer timer;
        using std::chrono::microseconds;
        ovms::SequenceProcessingSpec sequenceProcessingSpec;
        auto status = validate(requestProto, sequenceProcessingSpec);
        if (!status.ok())
            return status;

        if (waitBeforeManagerLock) {
            std::cout << "Waiting before sequenceManagerLock" << std::endl;
            waitBeforeManagerLock->get();
        }
        std::unique_lock<std::mutex> sequenceManagerLock(sequenceManager->getMutex());
        status = sequenceManager->processRequestedSpec(sequenceProcessingSpec);
        if (!status.ok())
            return status;
        const uint64_t sequenceId = sequenceProcessingSpec.getSequenceId();
        if (!sequenceManager->sequenceExists(sequenceId))
            return ovms::StatusCode::INTERNAL_ERROR;
        ovms::Sequence& sequence = sequenceManager->getSequence(sequenceId);

        if (waitBeforeSequenceLock) {
            std::cout << "Waiting before waitBeforeSequenceLock" << std::endl;
            waitBeforeSequenceLock->get();
        }

        std::unique_lock<std::mutex> sequenceLock(sequence.getMutex());
        sequenceManagerLock.unlock();

        timer.start("get infer request");
        ovms::ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue());

        InferenceEngine::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
        timer.stop("get infer request");

        timer.start("preprocess");
        status = preInferenceProcessing(inferRequest, sequence, sequenceProcessingSpec);
        timer.stop("preprocess");
        if (!status.ok())
            return status;

        timer.start("deserialize");
        ovms::InputSink<InferRequest&> inputSink(inferRequest);
        bool isPipeline = false;
        status = ovms::deserializePredictRequest<ovms::ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inputSink, isPipeline);
        timer.stop("deserialize");
        if (!status.ok())
            return status;

        timer.start("prediction");
        status = performInference(inferRequest);
        timer.stop("prediction");
        if (!status.ok())
            return status;

        timer.start("serialize");
        status = serializePredictResponse(inferRequest, getOutputsInfo(), responseProto);
        timer.stop("serialize");
        if (!status.ok())
            return status;

        timer.start("postprocess");
        status = postInferenceProcessing(responseProto, inferRequest, sequence, sequenceProcessingSpec);
        timer.stop("postprocess");
        if (!status.ok())
            return status;

        if (waitBeforeSequenceUnlocked) {
            std::cout << "Waiting before waitBeforeSequenceUnlocked" << std::endl;
            waitBeforeSequenceUnlocked->get();
        }

        sequenceLock.unlock();
        if (sequenceProcessingSpec.getSequenceControlInput() == ovms::SEQUENCE_END) {
            sequenceManagerLock.lock();
            status = sequenceManager->removeSequence(sequenceId);
            if (!status.ok())
                return status;
        }

        return ovms::StatusCode::OK;
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

void RunStatefulPredict(const std::shared_ptr<ovms::ModelInstance> modelInstance, inputs_info_t modelInput, uint64_t seqId, uint32_t sequenceControl) {
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, sequenceControl);

    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    tensorflow::serving::PredictResponse response;
    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::OK);
    // Check response
    EXPECT_TRUE(CheckSequenceIdResponse(response, seqId));
}

void RunStatefulPredicts(const std::shared_ptr<ovms::ModelInstance> modelInstance, inputs_info_t modelInput, int numberOfNoControlRequests, uint64_t seqId,
    std::future<void>* waitBeforeSequenceStarted,
    std::future<void>* waitAfterSequenceStarted,
    std::future<void>* waitBeforeSequenceFinished) {
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;

    if (waitBeforeSequenceStarted) {
        std::cout << "Waiting before StartStatefulPredict" << std::endl;
        waitBeforeSequenceStarted->get();
    }

    RunStatefulPredict(modelInstance, modelInput, seqId, ovms::SEQUENCE_START);
    if (waitAfterSequenceStarted) {
        std::cout << "Waiting after StartStatefulPredict" << std::endl;
        waitAfterSequenceStarted->get();
    }
    for (int i = 1; i < numberOfNoControlRequests; i++) {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        setRequestSequenceId(&request, seqId);
        setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

        tensorflow::serving::PredictResponse response;
        // Do the inference
        ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::OK);
        // Check response
        EXPECT_TRUE(CheckSequenceIdResponse(response, seqId));
    }

    if (waitBeforeSequenceFinished) {
        std::cout << "Waiting before EndStatefulPredict" << std::endl;
        waitBeforeSequenceFinished->get();
    }

    RunStatefulPredict(modelInstance, modelInput, seqId, ovms::SEQUENCE_END);
}

void RunStatefulPredictsOnMockedInferStart(const std::shared_ptr<MockedStatefulModelInstance> modelInstance, inputs_info_t modelInput, uint64_t seqId, SequenceTimeoutScenarios sequenceTimeoutScenario) {
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    std::promise<void> waitBeforeSequenceStarted, waitAfterSequenceStarted, waitBeforeSequenceFinished;

    // Cleaner scan before tested inference
    modelInstance->getSequencesViewer()->removeIdleSequences();

    // START
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_START);

    std::cout << "Executing timeout scenario : " << ToString(sequenceTimeoutScenario) << std::endl;
    tensorflow::serving::PredictResponse response;
    // Do the inference
    std::thread t1(
        [&waitBeforeSequenceStarted, &waitAfterSequenceStarted, &waitBeforeSequenceFinished, seqId, &modelInstance, &unload_guard, &response, &request]() {
            std::future<void> fut1 = waitBeforeSequenceStarted.get_future();
            std::future<void> fut2 = waitAfterSequenceStarted.get_future();
            std::future<void> fut3 = waitBeforeSequenceFinished.get_future();
            modelInstance->infer(&request, &response, unload_guard, &fut1, &fut2, &fut3);
        });

    std::promise<void> cleanerStartPromise, cleanerEndPromise;
    std::future<void> cleanerStartFuture = cleanerStartPromise.get_future();
    std::future<void> cleanerEndFuture = cleanerEndPromise.get_future();
    // Cleaner scan mid inference
    std::thread cleanerThread([&modelInstance, &cleanerStartFuture, &cleanerEndPromise]() {
        cleanerStartFuture.get();
        modelInstance->getSequencesViewer()->removeIdleSequences();
        cleanerEndPromise.set_value();
    });

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitAfterSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceFinished.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    cleanerEndFuture.get();
    cleanerThread.join();

    // Wait for sequence timeout
    t1.join();

    // Cleaner scan after tested inference
    modelInstance->getSequencesViewer()->removeIdleSequences();

    // END
    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    tensorflow::serving::PredictResponse response2;

    auto status = modelInstance->infer(&request, &response2, unload_guard, nullptr, nullptr, nullptr);

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_MISSING);
    }
}

void RunStatefulPredictsOnMockedInferMiddle(const std::shared_ptr<MockedStatefulModelInstance> modelInstance, inputs_info_t modelInput, uint64_t seqId, SequenceTimeoutScenarios sequenceTimeoutScenario) {
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    std::promise<void> waitBeforeSequenceStarted, waitAfterSequenceStarted, waitBeforeSequenceFinished;

    // START
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_START);

    tensorflow::serving::PredictResponse response3;

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response3, unload_guard, nullptr, nullptr, nullptr), ovms::StatusCode::OK);

    // Cleaner scan before tested inference
    modelInstance->getSequencesViewer()->removeIdleSequences();

    // NO CONTROL
    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

    std::cout << "Executing timeout scenario : " << ToString(sequenceTimeoutScenario) << std::endl;
    tensorflow::serving::PredictResponse response;
    // Do the inference
    std::thread t1(
        [&waitBeforeSequenceStarted, &waitAfterSequenceStarted, &waitBeforeSequenceFinished, seqId, &modelInstance, &unload_guard, &response, &request]() {
            std::future<void> fut1 = waitBeforeSequenceStarted.get_future();
            std::future<void> fut2 = waitAfterSequenceStarted.get_future();
            std::future<void> fut3 = waitBeforeSequenceFinished.get_future();
            modelInstance->infer(&request, &response, unload_guard, &fut1, &fut2, &fut3);
        });

    std::promise<void> cleanerStartPromise, cleanerEndPromise;
    std::future<void> cleanerStartFuture = cleanerStartPromise.get_future();
    std::future<void> cleanerEndFuture = cleanerEndPromise.get_future();
    // Cleaner scan mid inference
    std::thread cleanerThread([&modelInstance, &cleanerStartFuture, &cleanerEndPromise]() {
        cleanerStartFuture.get();
        modelInstance->getSequencesViewer()->removeIdleSequences();
        cleanerEndPromise.set_value();
    });

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitAfterSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceFinished.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    cleanerEndFuture.get();
    cleanerThread.join();

    // Wait for sequence timeout
    t1.join();

    // Cleaner scan after tested inference
    modelInstance->getSequencesViewer()->removeIdleSequences();

    // END
    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    tensorflow::serving::PredictResponse response2;

    // Do the inference
    auto status = modelInstance->infer(&request, &response2, unload_guard, nullptr, nullptr, nullptr);

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_MISSING);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
    }
    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
        ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_MISSING);
    }
}

void RunStatefulPredictsOnMockedInferEnd(const std::shared_ptr<MockedStatefulModelInstance> modelInstance, inputs_info_t modelInput, uint64_t seqId, SequenceTimeoutScenarios sequenceTimeoutScenario) {
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    std::promise<void> waitBeforeSequenceStarted, waitAfterSequenceStarted, waitBeforeSequenceFinished;

    // START
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_START);

    tensorflow::serving::PredictResponse response3;

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response3, unload_guard, nullptr, nullptr, nullptr), ovms::StatusCode::OK);

    // Cleaner scan before tested inference
    modelInstance->getSequencesViewer()->removeIdleSequences();

    // END
    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    std::cout << "Executing timeout scenario : " << ToString(sequenceTimeoutScenario) << std::endl;
    tensorflow::serving::PredictResponse response;
    // Do the inference
    std::thread t1(
        [&waitBeforeSequenceStarted, &waitAfterSequenceStarted, &waitBeforeSequenceFinished, seqId, sequenceTimeoutScenario, &modelInstance, &unload_guard, &response, &request]() {
            std::future<void> fut1 = waitBeforeSequenceStarted.get_future();
            std::future<void> fut2 = waitAfterSequenceStarted.get_future();
            std::future<void> fut3 = waitBeforeSequenceFinished.get_future();
            auto status = modelInstance->infer(&request, &response, unload_guard, &fut1, &fut2, &fut3);
            if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
                ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_MISSING);
            }
            if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
                ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
            }
            if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
                ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
            }
            if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
                ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
            }
        });

    std::promise<void> cleanerStartPromise, cleanerEndPromise;
    std::future<void> cleanerStartFuture = cleanerStartPromise.get_future();
    std::future<void> cleanerEndFuture = cleanerEndPromise.get_future();
    // Cleaner scan mid inference
    std::thread cleanerThread([&modelInstance, &cleanerStartFuture, &cleanerEndPromise]() {
        cleanerStartFuture.get();
        modelInstance->getSequencesViewer()->removeIdleSequences();
        cleanerEndPromise.set_value();
    });

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_MANAGER_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_LOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitAfterSequenceStarted.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_BEFORE_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    waitBeforeSequenceFinished.set_value();

    if (sequenceTimeoutScenario == SequenceTimeoutScenarios::WAIT_AFTER_SEQUENCE_UNLOCKED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cleanerStartPromise.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    cleanerEndFuture.get();
    cleanerThread.join();

    // Wait for sequence timeout
    t1.join();
}

TEST_F(StatefulModelInstanceTempDir, idleSequencesCleanup) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);
    uint64_t sequenceCounter = 10;

    for (uint64_t i = 1; i < sequenceCounter + 1; i++) {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        setRequestSequenceId(&request, i);
        setRequestSequenceControl(&request, ovms::SEQUENCE_START);

        tensorflow::serving::PredictResponse response3;

        // Do the inference
        ASSERT_EQ(stetefulMockedModelInstance->infer(&request, &response3, unload_guard, nullptr, nullptr, nullptr), ovms::StatusCode::OK);
    }

    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), sequenceCounter);
    // First sequence cleaner check sets idle flag on each sequence to true
    stetefulMockedModelInstance->getSequencesViewer()->removeIdleSequences();
    uint64_t activeSequencesNumber = 7;
    for (uint64_t i = 1; i < activeSequencesNumber + 1; i++) {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        setRequestSequenceId(&request, i);
        setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

        tensorflow::serving::PredictResponse response;

        // Do the inference -> set idle flag to false
        ASSERT_EQ(stetefulMockedModelInstance->infer(&request, &response, unload_guard, nullptr, nullptr, nullptr), ovms::StatusCode::OK);
    }

    // Second sequence cleaner check removes sequences with idle flag == true
    stetefulMockedModelInstance->getSequencesViewer()->removeIdleSequences();
    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), activeSequencesNumber);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferManagerMutexTest) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);
    uint64_t sequenceCounter = 10;

    for (uint64_t i = 1; i < sequenceCounter + 1; i++) {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        setRequestSequenceId(&request, i);
        setRequestSequenceControl(&request, ovms::SEQUENCE_START);

        tensorflow::serving::PredictResponse response;

        // Do the inference
        ASSERT_EQ(stetefulMockedModelInstance->infer(&request, &response, unload_guard, nullptr, nullptr, nullptr), ovms::StatusCode::OK);
    }

    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), sequenceCounter);

    std::promise<void> cleanerStartPromise, cleanerEndPromise;
    std::future<void> cleanerStartFuture = cleanerStartPromise.get_future();
    std::future<void> cleanerEndFuture = cleanerEndPromise.get_future();
    std::thread cleanerThread([&stetefulMockedModelInstance, &cleanerStartFuture, &cleanerEndPromise]() {
        cleanerStartFuture.get();
        stetefulMockedModelInstance->getSequencesViewer()->removeIdleSequences();
        cleanerEndPromise.set_value();
    });

    stetefulMockedModelInstance->getSequencesViewer()->removeIdleSequences();
    std::unique_lock<std::mutex> sequenceManagerLock(stetefulMockedModelInstance->getSequenceManager()->getMutex());
    cleanerStartPromise.set_value();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), sequenceCounter);
    sequenceManagerLock.unlock();
    cleanerEndFuture.get();
    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), 0);
    cleanerThread.join();
}

TEST_F(StatefulModelInstanceTempDir, statefulInferMultipleThreadsSequenceTimeout) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);

    uint64_t seqId = 0;
    uint16_t numberOfThreads = 8;

    std::vector<std::thread> inferThreads;
    for (auto i = 0u; i < numberOfThreads; ++i) {
        seqId++;
        inferThreads.emplace_back(
            std::thread(
                [this, seqId, stetefulMockedModelInstance]() {
                    RunStatefulPredictsOnMockedInferEnd(stetefulMockedModelInstance, modelInput, seqId, WAIT_BEFORE_MANAGER_LOCKED);
                }));
        seqId++;
        inferThreads.emplace_back(
            std::thread(
                [this, seqId, stetefulMockedModelInstance]() {
                    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId, WAIT_AFTER_SEQUENCE_UNLOCKED);
                }));
        seqId++;
        inferThreads.emplace_back(
            std::thread(
                [this, seqId, stetefulMockedModelInstance]() {
                    RunStatefulPredictsOnMockedInferStart(stetefulMockedModelInstance, modelInput, seqId, WAIT_AFTER_SEQUENCE_UNLOCKED);
                }));
        seqId++;
        inferThreads.emplace_back(
            std::thread(
                [this, seqId, stetefulMockedModelInstance]() {
                    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId, WAIT_BEFORE_MANAGER_LOCKED);
                }));
    }

    for (auto& thread : inferThreads) {
        thread.join();
    }

    ASSERT_EQ(stetefulMockedModelInstance->getSequenceManager()->getSequencesCount(), 0);
}

TEST_F(StatefulModelInstanceTempDir, modelInstanceFactory) {
    ConstructorEnabledModelManager manager;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    ASSERT_TRUE(typeid(*modelInstance) == typeid(ovms::StatefulModelInstance));
}

TEST_F(StatefulModelInstanceTempDir, statefulInferMultipleInferences) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);

    uint64_t seqId = 1;

    RunStatefulPredicts(modelInstance, modelInput, 100, seqId, nullptr, nullptr, nullptr);
    auto stetefulModelInstance = std::static_pointer_cast<ovms::StatefulModelInstance>(modelInstance);

    ASSERT_EQ(stetefulModelInstance->getSequenceManager()->getSequencesCount(), 0);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferMultipleThreads) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);

    uint64_t startingSequenceId = 0;
    uint16_t numberOfThreadsWaitingOnStart = 30;
    uint16_t numberOfThreadsWaitingOnEnd = 30;

    std::vector<std::promise<void>> releaseWaitBeforeSequenceStarted(numberOfThreadsWaitingOnStart + numberOfThreadsWaitingOnEnd), releaseWaitAfterSequenceStarted(numberOfThreadsWaitingOnStart), releaseWaitBeforeSequenceFinished(numberOfThreadsWaitingOnEnd);
    std::vector<std::thread> inferThreads;

    for (auto i = 0u; i < numberOfThreadsWaitingOnStart; ++i) {
        startingSequenceId++;

        inferThreads.emplace_back(
            std::thread(
                [this, &releaseWaitBeforeSequenceStarted, &releaseWaitAfterSequenceStarted, i, startingSequenceId, modelInstance]() {
                    std::future<void> fut1 = releaseWaitBeforeSequenceStarted[i].get_future();
                    std::future<void> fut2 = releaseWaitAfterSequenceStarted[i].get_future();
                    RunStatefulPredicts(modelInstance, modelInput, 100, startingSequenceId,
                        &fut1,
                        &fut2,
                        nullptr);
                }));
    }

    for (auto i = 0u; i < numberOfThreadsWaitingOnEnd; ++i) {
        startingSequenceId++;
        inferThreads.emplace_back(
            std::thread(
                [this, &releaseWaitBeforeSequenceStarted, &releaseWaitBeforeSequenceFinished, i, startingSequenceId, modelInstance, numberOfThreadsWaitingOnStart]() {
                    std::future<void> fut1 = releaseWaitBeforeSequenceStarted[numberOfThreadsWaitingOnStart + i].get_future();
                    std::future<void> fut2 = releaseWaitBeforeSequenceFinished[i].get_future();
                    RunStatefulPredicts(modelInstance, modelInput, 100, startingSequenceId,
                        &fut1,
                        nullptr,
                        &fut2);
                }));
    }

    // Sleep to allow all threads to initialize
    for (auto& promise : releaseWaitBeforeSequenceStarted) {
        promise.set_value();
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));

    auto stetefulModelInstance = std::static_pointer_cast<ovms::StatefulModelInstance>(modelInstance);

    ASSERT_EQ(stetefulModelInstance->getSequenceManager()->getSequencesCount(), numberOfThreadsWaitingOnEnd + numberOfThreadsWaitingOnStart);

    for (auto& promise : releaseWaitAfterSequenceStarted) {
        promise.set_value();
    }

    // Sleep to allow half threads to work
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_EQ(stetefulModelInstance->getSequenceManager()->getSequencesCount(), numberOfThreadsWaitingOnEnd);

    for (auto& promise : releaseWaitBeforeSequenceFinished) {
        promise.set_value();
    }

    for (auto& thread : inferThreads) {
        thread.join();
    }

    ASSERT_EQ(stetefulModelInstance->getSequenceManager()->getSequencesCount(), 0);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceStartTimeout) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    uint64_t seqId = 1;

    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);

    RunStatefulPredictsOnMockedInferStart(stetefulMockedModelInstance, modelInput, seqId, WAIT_BEFORE_MANAGER_LOCKED);
    RunStatefulPredictsOnMockedInferStart(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_LOCKED);
    RunStatefulPredictsOnMockedInferStart(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_UNLOCKED);
    RunStatefulPredictsOnMockedInferStart(stetefulMockedModelInstance, modelInput, seqId++, WAIT_AFTER_SEQUENCE_UNLOCKED);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceNoControlTimeout) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    uint64_t seqId = 1;

    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);

    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId, WAIT_BEFORE_MANAGER_LOCKED);
    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_LOCKED);
    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_UNLOCKED);
    RunStatefulPredictsOnMockedInferMiddle(stetefulMockedModelInstance, modelInput, seqId++, WAIT_AFTER_SEQUENCE_UNLOCKED);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceEndTimeout) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    SetUpConfig(modelStatefulConfig);
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    uint64_t seqId = 1;
    auto stetefulMockedModelInstance = std::static_pointer_cast<MockedStatefulModelInstance>(modelInstance);

    RunStatefulPredictsOnMockedInferEnd(stetefulMockedModelInstance, modelInput, seqId, WAIT_BEFORE_MANAGER_LOCKED);
    RunStatefulPredictsOnMockedInferEnd(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_LOCKED);
    RunStatefulPredictsOnMockedInferEnd(stetefulMockedModelInstance, modelInput, seqId++, WAIT_BEFORE_SEQUENCE_UNLOCKED);
    RunStatefulPredictsOnMockedInferEnd(stetefulMockedModelInstance, modelInput, seqId++, WAIT_AFTER_SEQUENCE_UNLOCKED);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceMissing) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    tensorflow::serving::PredictResponse response;
    uint64_t seqId = 1;

    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_MISSING);

    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_MISSING);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceIdNotProvided) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    tensorflow::serving::PredictResponse response;

    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);

    request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferSequenceIdNotProvided2) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    tensorflow::serving::PredictResponse response;

    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);

    request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, 99);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::INVALID_SEQUENCE_CONTROL_INPUT);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferIdAlreadyExists) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    tensorflow::serving::PredictResponse response;
    uint64_t seqId = 1;

    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_START);

    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::OK);
    ASSERT_EQ(modelInstance->infer(&request, &response, unload_guard), ovms::StatusCode::SEQUENCE_ALREADY_EXISTS);
}

TEST_F(StatefulModelInstanceTempDir, statefulInferStandardFlow) {
    ConstructorEnabledModelManager manager;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    createConfigFileWithContent(ovmsConfig, configFilePath);
    auto status = manager.loadConfig(configFilePath);
    ASSERT_TRUE(status.ok());
    auto modelInstance = manager.findModelInstance(dummyModelName);
    tensorflow::serving::PredictResponse firstResponse;
    uint64_t seqId = 1;

    tensorflow::serving::PredictRequest firstRequest = preparePredictRequest(modelInput);
    setRequestSequenceId(&firstRequest, seqId);
    setRequestSequenceControl(&firstRequest, ovms::SEQUENCE_START);

    // Do the inference
    ASSERT_EQ(modelInstance->infer(&firstRequest, &firstResponse, unload_guard), ovms::StatusCode::OK);

    // Check response
    EXPECT_TRUE(CheckSequenceIdResponse(firstResponse, seqId));

    tensorflow::serving::PredictRequest intermediateRequest = preparePredictRequest(modelInput);
    setRequestSequenceId(&intermediateRequest, seqId);
    setRequestSequenceControl(&intermediateRequest, ovms::NO_CONTROL_INPUT);

    tensorflow::serving::PredictResponse intermediateResponse;
    // Do the inference
    ASSERT_EQ(modelInstance->infer(&intermediateRequest, &intermediateResponse, unload_guard), ovms::StatusCode::OK);
    // Check response
    EXPECT_TRUE(CheckSequenceIdResponse(intermediateResponse, seqId));

    tensorflow::serving::PredictRequest lastRequest = preparePredictRequest(modelInput);
    setRequestSequenceId(&lastRequest, seqId);
    setRequestSequenceControl(&lastRequest, ovms::SEQUENCE_END);

    tensorflow::serving::PredictResponse lastResponse;
    // Do the inference
    ASSERT_EQ(modelInstance->infer(&lastRequest, &lastResponse, unload_guard), ovms::StatusCode::OK);
    // Check response
    EXPECT_TRUE(CheckSequenceIdResponse(lastResponse, seqId));
}

TEST_F(StatefulModelInstanceTempDir, loadModel) {
    ovms::GlobalSequencesViewer sequencesViewer;
    ovms::StatefulModelInstance modelInstance(dummyModelName, modelVersion, &sequencesViewer);

    const ovms::ModelConfig config1{
        dummyModelName,
        modelPath,     // base path
        "CPU",         // target device
        "1",           // batchsize
        1,             // NIREQ
        true,          // is stateful
        true,          // idle sequence cleanup enabled
        false,         // low latency transformation enabled
        44,            // steteful sequence max number
        modelVersion,  // version
        modelPath,     // local path
    };
    auto status = modelInstance.loadModel(config1);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();

    EXPECT_EQ(modelInstance.getSequenceManager()->getMaxSequenceNumber(), 44);
    EXPECT_EQ(modelInstance.getModelConfig().isLowLatencyTransformationUsed(), false);

    const ovms::ModelConfig config2{
        dummyModelName,
        modelPath,     // base path
        "CPU",         // target device
        "1",           // batchsize
        1,             // NIREQ
        true,          // is stateful
        true,          // idle sequence cleanup enabled
        true,          // low latency transformation enabled
        11,            // steteful sequence max number
        modelVersion,  // version
        modelPath,     // local path
    };
    status = modelInstance.reloadModel(config2);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();

    EXPECT_EQ(modelInstance.getSequenceManager()->getMaxSequenceNumber(), 11);
    EXPECT_EQ(modelInstance.getModelConfig().isLowLatencyTransformationUsed(), true);
}

TEST_F(StatefulModelInstanceInputValidation, positiveValidate) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    uint64_t seqId = 1;
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, seqId);

    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_START);

    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());

    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());

    request = preparePredictRequest(modelInput);
    setRequestSequenceId(&request, seqId);
    setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

    status = modelInstance->mockValidate(&request, spec);
    ASSERT_TRUE(status.ok());
}

TEST_F(StatefulModelInstanceInputValidation, missingSeqId) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongSeqIdEnd) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, ovms::SEQUENCE_END);

    uint64_t seqId = 0;
    setRequestSequenceId(&request, seqId);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongSeqIdNoControl) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    setRequestSequenceControl(&request, ovms::NO_CONTROL_INPUT);

    uint64_t seqId = 0;
    setRequestSequenceId(&request, seqId);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, wrongProtoKeywords) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    auto& input = (*request.mutable_inputs())["sequenceid"];
    input.set_dtype(tensorflow::DataType::DT_UINT64);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
    input.add_uint64_val(12);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_NOT_PROVIDED);
}

TEST_F(StatefulModelInstanceInputValidation, badControlInput) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
    request = preparePredictRequest(modelInput);
    auto& input = (*request.mutable_inputs())["sequence_control_input"];
    input.set_dtype(tensorflow::DataType::DT_UINT32);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
    input.add_uint32_val(999);
    auto status = modelInstance->mockValidate(&request, spec);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::INVALID_SEQUENCE_CONTROL_INPUT);
}

TEST_F(StatefulModelInstanceInputValidation, invalidProtoTypes) {
    std::shared_ptr<MockedValidateStatefulModelInstance> modelInstance = std::make_shared<MockedValidateStatefulModelInstance>("model", 1);
    ovms::SequenceProcessingSpec spec(ovms::SEQUENCE_START, 1);
    {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        auto& input = (*request.mutable_inputs())["sequence_id"];
        input.set_dtype(tensorflow::DataType::DT_UINT32);
        input.mutable_tensor_shape()->add_dim()->set_size(1);
        input.add_uint32_val(12);
        auto status = modelInstance->mockValidate(&request, spec);
        ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_ID_BAD_TYPE);
    }
    {
        tensorflow::serving::PredictRequest request = preparePredictRequest(modelInput);
        auto& input = (*request.mutable_inputs())["sequence_control_input"];
        input.set_dtype(tensorflow::DataType::DT_UINT64);
        input.mutable_tensor_shape()->add_dim()->set_size(1);
        input.add_uint64_val(1);
        auto status = modelInstance->mockValidate(&request, spec);
        ASSERT_EQ(status.getCode(), ovms::StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE);
    }
}

TEST_F(StatefulModelInstanceTest, PreprocessingFirstRequest) {
    // Prepare model instance and processing spec
    uint32_t sequenceControlInput = ovms::SEQUENCE_START;
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
    currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
    EXPECT_EQ(currentBlobIrData, currentState);

    // Perform preprocessing (load state from sequence to infer request)
    ovms::Sequence sequence(sequenceId);
    modelInstance->preInferenceProcessing(inferRequest, sequence, sequenceProcessingSpec);

    // Check if InferRequest memory state has been reset to default
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
    currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
    EXPECT_EQ(currentBlobIrData, defaultState);
}

TEST_F(StatefulModelInstanceTest, PreprocessingIntermediateRequest) {
    for (uint32_t sequenceControlInput : {ovms::NO_CONTROL_INPUT, ovms::SEQUENCE_END}) {
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
        currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
        EXPECT_EQ(currentBlobIrData, defaultState);

        // Inject sequence with newState as the last state written to sequence memory state
        ovms::model_memory_state_t memoryState;
        addState(memoryState, "state", shape, newState);
        modelInstance->injectSequence(sequenceId, memoryState);

        // Perform preprocessing (load state from sequence to infer request)
        ovms::Sequence& sequence = modelInstance->getMockedSequenceManager()->getSequence(sequenceId);
        modelInstance->preInferenceProcessing(inferRequest, sequence, sequenceProcessingSpec);

        // Check if InferRequest memory state has been updated to sequence memory state
        EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
        currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
        EXPECT_EQ(currentBlobIrData, newState);
    }
}

TEST_F(StatefulModelInstanceTest, PostprocessingLastRequest) {
    // Prepare model instance and processing spec
    uint32_t sequenceControlInput = ovms::SEQUENCE_END;
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
    currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
    EXPECT_EQ(currentBlobIrData, currentState);

    tensorflow::serving::PredictResponse response;
    ovms::Sequence sequence(sequenceId);
    modelInstance->postInferenceProcessing(&response, inferRequest, sequence, sequenceProcessingSpec);

    EXPECT_TRUE(CheckSequenceIdResponse(response, sequenceId));

    // Check if InferRequest memory state has been reset to default
    EXPECT_EQ(ovms::blobClone(stateCloneBlob, irMemoryState[0].GetState()), ovms::StatusCode::OK);
    currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
    EXPECT_EQ(currentBlobIrData, defaultState);
}

TEST_F(StatefulModelInstanceTest, PostprocessingStartAndNoControl) {
    for (uint32_t sequenceControlInput : {ovms::NO_CONTROL_INPUT, ovms::SEQUENCE_START}) {
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
        currentBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(stateCloneBlob)->rmap().as<float*>() + elementsCount);
        EXPECT_EQ(currentBlobIrData, defaultState);

        // Inject sequence with newState as the last state written to sequence memory state
        ovms::model_memory_state_t memoryState;
        addState(memoryState, "state", shape, newState);
        modelInstance->injectSequence(sequenceId, memoryState);
        ovms::Sequence& sequence = modelInstance->getMockedSequenceManager()->getSequence(sequenceId);
        // Sanity check for new state
        const ovms::sequence_memory_state_t& currentSequenceMemoryState = sequence.getMemoryState();
        EXPECT_TRUE(currentSequenceMemoryState.count("state"));
        InferenceEngine::Blob::Ptr sanityBlob = currentSequenceMemoryState.at("state");
        std::vector<float> sanityBlobIrData;
        sanityBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(sanityBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(sanityBlob)->rmap().as<float*>() + elementsCount);
        EXPECT_EQ(sanityBlobIrData, newState);

        tensorflow::serving::PredictResponse response;
        modelInstance->postInferenceProcessing(&response, inferRequest, sequence, sequenceProcessingSpec);

        // Check if sequence memory state is the same as InferRequest memory state
        const ovms::sequence_memory_state_t& updatedSequenceMemoryState = sequence.getMemoryState();
        EXPECT_TRUE(updatedSequenceMemoryState.count("state"));
        InferenceEngine::Blob::Ptr chengedBlob = updatedSequenceMemoryState.at("state");
        std::vector<float> sequenceBlobIrData;
        sequenceBlobIrData.assign(InferenceEngine::as<InferenceEngine::MemoryBlob>(chengedBlob)->rmap().as<float*>(), InferenceEngine::as<InferenceEngine::MemoryBlob>(chengedBlob)->rmap().as<float*>() + elementsCount);
        EXPECT_EQ(sequenceBlobIrData, defaultState);
        EXPECT_TRUE(CheckSequenceIdResponse(response, sequenceId));
    }
}

TEST_F(StatefulModelInstanceTest, extractSequenceId_OK) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT64);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.add_uint64_val(5000000000);
    uint64_t sequenceId = 0;
    EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::OK);
    EXPECT_EQ(sequenceId, 5000000000);
}

TEST_F(StatefulModelInstanceTest, extractSequenceId_NoTensorShape) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT64);
    proto.add_uint64_val(5000000000);
    uint64_t sequenceId = 0;
    EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE);
    EXPECT_EQ(sequenceId, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceId_WrongTensorShapeDims) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT64);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.add_uint64_val(5000000000);
    uint64_t sequenceId = 0;
    EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
    EXPECT_EQ(sequenceId, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceId_WrongTensorShape) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT64);
    proto.mutable_tensor_shape()->add_dim()->set_size(2);
    proto.add_uint64_val(5000000000);
    uint64_t sequenceId = 0;
    EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::INVALID_SHAPE);
    EXPECT_EQ(sequenceId, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceId_WrongValField) {
    {
        // Data stored in the wrong val field
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_UINT64);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.add_uint32_val(1);
        uint64_t sequenceId = 0;
        EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::SEQUENCE_ID_BAD_TYPE);
        EXPECT_EQ(sequenceId, 0);
    }
    {
        // Good val field, but too many values
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_UINT64);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.add_uint64_val(1);
        proto.add_uint64_val(2);
        uint64_t sequenceId = 0;
        EXPECT_EQ(modelInstance->extractSequenceId(proto, sequenceId), ovms::StatusCode::SEQUENCE_ID_BAD_TYPE);
        EXPECT_EQ(sequenceId, 0);
    }
}

TEST_F(StatefulModelInstanceTest, extractSequenceControlInput_OK) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT32);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.add_uint32_val(1);
    uint32_t sequenceControlInput = 0;
    EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::OK);
    EXPECT_EQ(sequenceControlInput, 1);
}

TEST_F(StatefulModelInstanceTest, extractSequenceControlInput_NoTensorShape) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT32);
    proto.add_uint32_val(1);
    uint32_t sequenceControlInput = 0;
    EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE);
    EXPECT_EQ(sequenceControlInput, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceControlInput_WrongTensorShapeDims) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT32);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.add_uint32_val(1);
    uint32_t sequenceControlInput = 0;
    EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS);
    EXPECT_EQ(sequenceControlInput, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceControlInput_WrongTensorShape) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_UINT32);
    proto.mutable_tensor_shape()->add_dim()->set_size(2);
    proto.add_uint32_val(1);
    uint32_t sequenceControlInput = 0;
    EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::INVALID_SHAPE);
    EXPECT_EQ(sequenceControlInput, 0);
}

TEST_F(StatefulModelInstanceTest, extractSequenceControlInput_WrongValField) {
    {
        // Data stored in the wrong val field
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_UINT32);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.add_uint64_val(1);
        uint32_t sequenceControlInput = 0;
        EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE);
        EXPECT_EQ(sequenceControlInput, 0);
    }
    {
        // Good val field, but too many values
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_UINT32);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.add_uint32_val(1);
        proto.add_uint32_val(2);
        uint32_t sequenceControlInput = 0;
        EXPECT_EQ(modelInstance->extractSequenceControlInput(proto, sequenceControlInput), ovms::StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE);
        EXPECT_EQ(sequenceControlInput, 0);
    }
}

#pragma GCC diagnostic pop
}  // namespace
