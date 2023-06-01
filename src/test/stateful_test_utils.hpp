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
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../sequence.hpp"
#include "../sequence_manager.hpp"
#include "../tensorinfo.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

const std::string SEQUENCE_ID_INPUT = "sequence_id";
const std::string SEQUENCE_CONTROL_INPUT = "sequence_control_input";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void setRequestSequenceId(tensorflow::serving::PredictRequest* request, uint64_t sequenceId) {
    auto& input = (*request->mutable_inputs())[SEQUENCE_ID_INPUT];
    input.set_dtype(tensorflow::DataType::DT_UINT64);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
    input.add_uint64_val(sequenceId);
}

static void setRequestSequenceControl(tensorflow::serving::PredictRequest* request, uint32_t sequenceControl) {
    auto& input = (*request->mutable_inputs())[SEQUENCE_CONTROL_INPUT];
    input.set_dtype(tensorflow::DataType::DT_UINT32);
    input.mutable_tensor_shape()->add_dim()->set_size(1);
    input.add_uint32_val(sequenceControl);
}

static bool CheckSequenceIdResponse(tensorflow::serving::PredictResponse& response, uint64_t seqId) {
    // Check response
    auto it = response.mutable_outputs()->find("sequence_id");
    if (it == response.mutable_outputs()->end())
        return false;
    auto& output = (*response.mutable_outputs())["sequence_id"];
    if (output.uint64_val_size() != 1)
        return false;
    if (output.uint64_val(0) != seqId)
        return false;

    return true;
}

class DummyStatefulModel {
private:
    ov::Core ieCore;
    const std::string MODEL_PATH = std::filesystem::current_path().u8string() + "/src/test/summator/1/summator.xml";

    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::CompiledModel> compiledModel;

    const std::string stateName = "state";

public:
    DummyStatefulModel() {
        model = ieCore.read_model(MODEL_PATH);
        compiledModel = std::make_shared<ov::CompiledModel>(ieCore.compile_model(model, "CPU"));
    }

    ov::InferRequest createInferRequest() {
        return compiledModel->create_infer_request();
    }

    const std::string getStateName() {
        return stateName;
    }

    static ov::VariableState getVariableState(ov::InferRequest& inferRequest) {
        std::vector<ov::VariableState> memoryState = inferRequest.query_state();
        return memoryState[0];
    }

    static void resetVariableState(ov::InferRequest& inferRequest) {
        std::vector<ov::VariableState> memoryState = inferRequest.query_state();
        memoryState[0].reset();
    }

    static void setVariableState(ov::InferRequest& inferRequest, std::vector<float> values) {
        DummyStatefulModel::resetVariableState(inferRequest);
        std::vector<size_t> shape{1, 1};

        ov::Tensor tensor(
            ov::element::Type_t::f32,
            shape,
            values.data());

        inferRequest.set_tensor("input", tensor);
        inferRequest.infer();
    }
};

#pragma GCC diagnostic pop

class MockedSequenceManager : public ovms::SequenceManager {
public:
    MockedSequenceManager(uint32_t maxSequenceNumber, std::string name, ovms::model_version_t version) :
        ovms::SequenceManager(maxSequenceNumber, name, version) {}

    void setSequenceIdCounter(uint64_t newValue) {
        this->sequenceIdCounter = newValue;
    }

    uint64_t mockGetUniqueSequenceId() {
        return ovms::SequenceManager::getUniqueSequenceId();
    }

    ovms::Status mockHasSequence(const uint64_t& sequenceId) {
        return ovms::SequenceManager::hasSequence(sequenceId);
    }

    ovms::Status mockCreateSequence(ovms::SequenceProcessingSpec& sequenceProcessingSpec) {
        return ovms::SequenceManager::createSequence(sequenceProcessingSpec);
    }

    ovms::Status mockTerminateSequence(const uint64_t& sequenceId) {
        return ovms::SequenceManager::terminateSequence(sequenceId);
    }
};
