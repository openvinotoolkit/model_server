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
#pragma once

#include <memory>
#include <string>
#include <utility>

#include "../model.hpp"
#include "../modelinstance.hpp"
#include "../status.hpp"

class MockModelInstanceChangingStates : public ovms::ModelInstance {
    static const ovms::model_version_t UNUSED_VERSION = 987789;

public:
    MockModelInstanceChangingStates() {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, ovms::ModelVersionState::START);
    }
    virtual ~MockModelInstanceChangingStates() {}
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        this->status = ovms::ModelVersionStatus(config.getName(), config.getVersion());
        this->status.setLoading();
        this->name = config.getName();
        this->version = config.getVersion();
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
    ovms::Status reloadModel(const ovms::ModelConfig& config) override {
        version = config.getVersion();
        status.setLoading();
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
    void unloadModel() override {
        status.setUnloading();
        status.setEnd();
    }
};

class MockModelWithInstancesJustChangingStates : public ovms::Model {
public:
    MockModelWithInstancesJustChangingStates(const std::string& name = "UNUSED_NAME") :
        Model(name) {}
    virtual ~MockModelWithInstancesJustChangingStates() {}

protected:
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory() override {
        return std::move(std::make_shared<MockModelInstanceChangingStates>());
    }
};
