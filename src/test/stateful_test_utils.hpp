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

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>

#include "../sequence.hpp"
#include "../sequence_manager.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "mock_iinferrequest.hpp"

#include <gmock/gmock-generated-function-mockers.h>

using namespace InferenceEngine;

const std::string SEQUENCE_ID_INPUT = "sequence_id";
const std::string SEQUENCE_CONTROL_INPUT = "sequence_control_input";

class MockIVariableState : public IVariableState {
public:
    MOCK_METHOD(StatusCode, GetName, (char* name, size_t len, ResponseDesc* resp), (const, noexcept, override));
    MOCK_METHOD(StatusCode, Reset, (ResponseDesc * resp), (noexcept, override));
    MOCK_METHOD(StatusCode, SetState, (Blob::Ptr newState, ResponseDesc* resp), (noexcept, override));
    MOCK_METHOD(StatusCode, GetState, (Blob::CPtr & state, ResponseDesc* resp), (const, noexcept, override));
};

class MockIVariableStateWithData : public MockIVariableState {
public:
    std::string stateName;
    Blob::Ptr currentBlob;
    const Blob::Ptr defaultBlob;

    MockIVariableStateWithData(std::string name, Blob::Ptr currentBlob) :
        stateName(name),
        currentBlob(currentBlob) {}

    MockIVariableStateWithData(std::string name, Blob::Ptr currentBlob, Blob::Ptr defaultBlob) :
        stateName(name),
        currentBlob(currentBlob),
        defaultBlob(defaultBlob) {}

    StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept override {
        snprintf(name, sizeof(stateName), stateName.c_str());
        return StatusCode::OK;
    }

    StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept override {
        state = currentBlob;
        return StatusCode::OK;
    }

    StatusCode Reset(ResponseDesc* resp) noexcept override {
        currentBlob = defaultBlob;
        return StatusCode::OK;
    }

    StatusCode SetState(Blob::Ptr newState, ResponseDesc* resp) noexcept override {
        currentBlob = newState;
        return StatusCode::OK;
    }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void setRequestSequenceId(tensorflow::serving::PredictRequest* request, uint64_t sequenceId) {
    auto& input = (*request->mutable_inputs())[SEQUENCE_ID_INPUT];
    input.add_uint64_val(sequenceId);
}

static void setRequestSequenceControl(tensorflow::serving::PredictRequest* request, uint32_t sequenceControl) {
    auto& input = (*request->mutable_inputs())[SEQUENCE_CONTROL_INPUT];
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

static void addState(ovms::model_memory_state_t& states, std::string name, std::vector<size_t>& shape, std::vector<float>& values) {
    const Precision precision{Precision::FP32};
    const Layout layout{Layout::NC};
    const TensorDesc desc{precision, shape, layout};

    Blob::Ptr stateBlob = make_shared_blob<float>(desc, values.data());
    std::shared_ptr<IVariableState> ivarPtr = std::make_shared<MockIVariableStateWithData>(name, stateBlob);
    states.push_back(VariableState(ivarPtr));
}

#pragma GCC diagnostic pop

class MockIInferRequestStateful : public MockIInferRequest {
public:
    IVariableState::Ptr memoryState;

    MockIInferRequestStateful(std::string name, Blob::Ptr currentBlob, Blob::Ptr defaultBlob) {
        this->memoryState = std::make_shared<MockIVariableStateWithData>(name, currentBlob, defaultBlob);
    }

    InferenceEngine::StatusCode QueryState(IVariableState::Ptr& pState, size_t idx, ResponseDesc* resp) noexcept override {
        if (idx == 0) {
            pState = memoryState;
            return InferenceEngine::StatusCode::OK;
        }
        return InferenceEngine::StatusCode::OUT_OF_BOUNDS;
    }
};
