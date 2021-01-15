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

#include <gmock/gmock-generated-function-mockers.h>

using namespace InferenceEngine;

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
    Blob::Ptr stateBlob;

    MockIVariableStateWithData(std::string name, Blob::Ptr blob) {
        stateName = name;
        stateBlob = blob;
    }

    StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept override {
        snprintf(name, sizeof(stateName), stateName.c_str());
        return StatusCode::OK;
    }

    StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept override {
        state = stateBlob;
        return StatusCode::OK;
    }
};

static void addState(ovms::model_memory_state_t& states, std::string name, std::vector<size_t>& shape, std::vector<float>& values) {
    const Precision precision{Precision::FP32};
    const Layout layout{Layout::NC};
    const TensorDesc desc{precision, shape, layout};

    Blob::Ptr stateBlob = make_shared_blob<float>(desc, values.data());
    std::shared_ptr<IVariableState> ivarPtr = std::make_shared<MockIVariableStateWithData>(name, stateBlob);
    states.push_back(VariableState(ivarPtr));
}
