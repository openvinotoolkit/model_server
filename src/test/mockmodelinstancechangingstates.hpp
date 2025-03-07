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
#include "../modelversionstatus.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

class MockModelInstanceChangingStates : public ovms::ModelInstance {
public:
    MockModelInstanceChangingStates(const std::string& modelName, const ovms::model_version_t modelVersion, ov::Core& ieCore, ovms::MetricRegistry* registry = nullptr, const ovms::MetricConfig* metricConfig = nullptr) :
        ModelInstance(modelName, modelVersion, ieCore, registry, metricConfig) {
        status = ovms::ModelVersionStatus(modelName, modelVersion, ovms::ModelVersionState::START);
    }
    virtual ~MockModelInstanceChangingStates() {}
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        this->status = ovms::ModelVersionStatus(config.getName(), config.getVersion());
        this->status.setLoading();
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
    ovms::Status reloadModel(const ovms::ModelConfig& config, const ovms::DynamicModelParameter& parameter = ovms::DynamicModelParameter()) override {
        status.setLoading();
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
    void retireModel(bool isPermanent = true) override {
        if (isPermanent) {
            status.setUnloading();
            status.setEnd();
        } else {
            status.setLoading();
        }
    }
    void cleanupFailedLoad() override {
        status.setLoading(ovms::ModelVersionStatusErrorCode::UNKNOWN);
    }
    void setState(ovms::ModelVersionState state) {
        status.setState(state);
    }
};

class MockModelWithInstancesJustChangingStates : public ovms::Model {
public:
    MockModelWithInstancesJustChangingStates(const std::string& name = "UNUSED_NAME") :
        Model(name, false, nullptr) {}
    virtual ~MockModelWithInstancesJustChangingStates() {}

protected:
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(ov::Core& ieCore) { return modelInstanceFactory("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore, nullptr, nullptr); }
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t version, ov::Core& ieCore, ovms::MetricRegistry* registry = nullptr, const ovms::MetricConfig* metricConfig = nullptr) override {
        return std::move(std::make_shared<MockModelInstanceChangingStates>(modelName, version, ieCore, registry, metricConfig));
    }
};
