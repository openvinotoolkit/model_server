//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <optional>
#include <string>

#include <gmock/gmock.h>

#include "src/kfs_frontend/validation.hpp"
#include "src/modelinstance.hpp"

class MockedMetadataModelIns : public ovms::ModelInstance {
public:
    MockedMetadataModelIns(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", 42, ieCore) {}
    MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
    MOCK_METHOD(const ovms::tensor_map_t&, getOutputsInfo, (), (const, override));
    MOCK_METHOD(std::optional<ovms::Dimension>, getBatchSize, (), (const, override));
    MOCK_METHOD(const ovms::ModelConfig&, getModelConfig, (), (const, override));
    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
        return validate(request);
    }
    const ovms::Status mockValidate(const ::KFSRequest* request) {
        return validate(request);
    }
    const ovms::Status mockValidate(const ovms::InferenceRequest* request) {
        return validate(request);
    }
    template <typename RequestType>
    ovms::Status validate(const RequestType* request) {
        return ovms::request_validation_utils::validate(
            *request,
            this->getInputsInfo(),
            this->getOutputsInfo(),
            this->getName(),
            this->getVersion(),
            this->getOptionalInputNames(),
            this->getModelConfig().getBatchingMode(),
            this->getModelConfig().getShapes());
    }
};
