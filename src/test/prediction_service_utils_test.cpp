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
#include <chrono>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../model.hpp"
#include "../modelmanager.hpp"
#include "../prediction_service_utils.hpp"
#include "test_utils.hpp"

class GetModelInstanceTest : public ::testing::Test {};

class MockModel : public ovms::Model {};

std::shared_ptr<ovms::Model> model;

class MockModelManagerWith1Model : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name) override {
        return model;
    }
};

TEST_F(GetModelInstanceTest, WithRequestedNameShouldReturnModelNameMissing) {
    MockModelManagerWith1Model manager;
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, "SOME", 0, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_NAME_MISSING) << "Should fail with no model with such name registered";
}

TEST_F(GetModelInstanceTest, WithRequestedUnexistingVersionShouldReturnModelVersionMissing) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_unique<ovms::Model>(config.getName());
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, config.getName(), 2, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_MISSING) << "Should fail with no model with such name registered";
}

class MockModelInstanceFakeLoad : public ovms::ModelInstance {
public:
    MockModelInstanceFakeLoad() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        status = ovms::ModelVersionStatus(name, version);
        status.setAvailable();
        return ovms::StatusCode::OK;
    }
};

class ModelWithModelInstanceFakeLoad : public ovms::Model {
public:
    ModelWithModelInstanceFakeLoad(const std::string& name) :
        Model(name) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t) override {
        return std::make_shared<MockModelInstanceFakeLoad>();
    }
};

std::shared_ptr<ModelWithModelInstanceFakeLoad> modelWithModelInstanceFakeLoad;

TEST_F(GetModelInstanceTest, WithRequestedDefaultVersionUnloadedShouldReturnModelVersionMissing) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_unique<ovms::Model>(config.getName());
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    std::shared_ptr<ovms::model_versions_t> versionsToRetire = std::make_shared<ovms::model_versions_t>();
    versionsToRetire->emplace_back(1);
    model->retireVersions(versionsToRetire);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, config.getName(), 0, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(GetModelInstanceTest, WithRequestedVersion1ShouldReturnModelVersionNotLoadedAnymore) {
    MockModelManagerWith1Model manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    model = std::make_shared<ovms::Model>(config.getName());
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    std::shared_ptr<ovms::model_versions_t> versionsToRetire = std::make_shared<ovms::model_versions_t>();
    versionsToRetire->emplace_back(1);
    model->retireVersions(versionsToRetire);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
}

class ModelInstanceLoadedStuckInLoadingState : public ovms::ModelInstance {
public:
    ModelInstanceLoadedStuckInLoadingState() :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        status = ovms::ModelVersionStatus(name, version);
        status.setLoading();
        return ovms::StatusCode::OK;
    }
};

class ModelWithModelInstanceLoadedStuckInLoadingState : public ovms::Model {
public:
    ModelWithModelInstanceLoadedStuckInLoadingState(const std::string& name) :
        Model(name) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t) override {
        return std::make_shared<ModelInstanceLoadedStuckInLoadingState>();
    }
};

std::shared_ptr<ModelWithModelInstanceLoadedStuckInLoadingState> modelWithModelInstanceLoadedStuckInLoadingState;

class ModelManagerWithModelInstanceLoadedStuckInLoadingState : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name) override {
        return modelWithModelInstanceLoadedStuckInLoadingState;
    }
};

const int AVAILABLE_STATE_DELAY_MILLISECONDS = 5;

class ModelInstanceLoadedWaitInLoadingState : public ovms::ModelInstance {
public:
    ModelInstanceLoadedWaitInLoadingState(const uint modelInstanceLoadDelayInMilliseconds) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION),
        modelInstanceLoadDelayInMilliseconds(modelInstanceLoadDelayInMilliseconds) {}

protected:
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        this->status = ovms::ModelVersionStatus(name, version);
        this->status.setLoading();
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(modelInstanceLoadDelayInMilliseconds));
            status.setAvailable();
            modelLoadedNotify.notify_all();
        })
            .detach();
        return ovms::StatusCode::OK;
    }

private:
    const uint modelInstanceLoadDelayInMilliseconds;
};

class ModelWithModelInstanceLoadedWaitInLoadingState : public ovms::Model {
public:
    ModelWithModelInstanceLoadedWaitInLoadingState(const std::string& name, const uint modelInstanceLoadDelayInMilliseconds) :
        Model(name),
        modelInstanceLoadDelayInMilliseconds(modelInstanceLoadDelayInMilliseconds) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t) override {
        return std::make_shared<ModelInstanceLoadedWaitInLoadingState>(modelInstanceLoadDelayInMilliseconds);
    }

private:
    const uint modelInstanceLoadDelayInMilliseconds;
};

std::shared_ptr<ModelWithModelInstanceLoadedWaitInLoadingState> modelWithModelInstanceLoadedWaitInLoadingState;

class ModelManagerWithModelInstanceLoadedWaitInLoadingState : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name) override {
        return modelWithModelInstanceLoadedWaitInLoadingState;
    }
};

class ModelInstanceModelLoadedNotify : public ::testing::Test {};

TEST_F(ModelInstanceModelLoadedNotify, WhenChangedStateFromLoadingToAvailableInNotReachingTimeoutShouldSuceed) {
    // Need unit tests for modelInstance load first
    ModelManagerWithModelInstanceLoadedWaitInLoadingState manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    modelWithModelInstanceLoadedWaitInLoadingState = std::make_shared<ModelWithModelInstanceLoadedWaitInLoadingState>(
        config.getName(), ovms::WAIT_FOR_MODEL_LOADED_TIMEOUT_MS / 4);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(ovms::ModelVersionState::AVAILABLE, modelInstance->getStatus().getState());
}

TEST_F(ModelInstanceModelLoadedNotify, WhenChangedStateFromLoadingToAvailableInReachingTimeoutShouldReturnModelNotLoadedYet) {
    // Need unit tests for modelInstance load first
    ModelManagerWithModelInstanceLoadedWaitInLoadingState manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    const auto MODEL_LOADING_LONGER_THAN_WAIT_FOR_LOADED_TIMEOUT_MS = 1.2 * ovms::WAIT_FOR_MODEL_LOADED_TIMEOUT_MS;
    modelWithModelInstanceLoadedWaitInLoadingState = std::make_shared<ModelWithModelInstanceLoadedWaitInLoadingState>(
        config.getName(), MODEL_LOADING_LONGER_THAN_WAIT_FOR_LOADED_TIMEOUT_MS);
    ASSERT_EQ(manager.reloadModelWithVersions(config), ovms::StatusCode::OK);
    ASSERT_EQ(ovms::ModelVersionState::LOADING, modelWithModelInstanceLoadedWaitInLoadingState->getModelInstanceByVersion(1)->getStatus().getState());
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto status = ovms::getModelInstance(manager, config.getName(), 1, modelInstance, modelInstanceUnloadGuardPtr);
    spdlog::error("State: {}", (int)modelInstance->getStatus().getState());
    EXPECT_EQ(ovms::ModelVersionState::LOADING, modelInstance->getStatus().getState()) << "State:" << (int)modelInstance->getStatus().getState();
    spdlog::error("State: {}", (int)modelInstance->getStatus().getState());
    EXPECT_EQ(status, ovms::StatusCode::MODEL_VERSION_NOT_LOADED_YET);
}
