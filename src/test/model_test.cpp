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
#include <deque>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../model.hpp"
#include "test_utils.hpp"

namespace {
class MockModelInstanceChangingStates : public ovms::ModelInstance {
   static const ovms::model_version_t UNUSED_VERSION = 987789;
public:
   MockModelInstanceChangingStates() {
       status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_VERSION, ovms::ModelVersionState::START);
   }
   virtual ~MockModelInstanceChangingStates() {}
   ovms::Status loadModel(const ovms::ModelConfig& config) override {
        version = config.getVersion();
        status.setAvailable();
        return ovms::StatusCode::OK;
   }
   void unloadModel() override {
       status.setEnd();
   }
};

class MockModel : public ovms::Model {
public:
    MockModel() : Model("UNUSED_NAME") {}
    virtual ~MockModel() {}

protected:
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory() override {
        return std::move(std::make_shared<MockModelInstanceChangingStates>());
    }
};

class ModelDefaultVersions: public ::testing::Test { };
}  //  namespace

TEST_F(ModelDefaultVersions, DefaultVersionNullWhenNoVersionAdded) {
    MockModel mockModel;
    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_EQ(nullptr, defaultInstance);
}

TEST_F(ModelDefaultVersions, DefaultVersionNullWhenVersionRetired) {
    MockModel mockModel;
    std::shared_ptr<ovms::model_versions_t> versionsToChange = std::make_shared<ovms::model_versions_t>();
    versionsToChange->push_back(1);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    mockModel.addVersions(versionsToChange, config);
    mockModel.retireVersions(versionsToChange);

    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_EQ(nullptr, defaultInstance);
}

TEST_F(ModelDefaultVersions, DefaultVersionShouldReturnValidWhen1Added) {
    MockModel mockModel;
    std::shared_ptr<ovms::model_versions_t> versionsToChange = std::make_shared<ovms::model_versions_t>();
    versionsToChange->push_back(1);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);

    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_TRUE(nullptr != defaultInstance);
    EXPECT_EQ(1, defaultInstance->getVersion());
}

TEST_F(ModelDefaultVersions, DefaultVersionShouldReturnHighest) {
    MockModel mockModel;
    std::shared_ptr<ovms::model_versions_t> versionsToChange = std::make_shared<ovms::model_versions_t>();
    versionsToChange->push_back(1);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();
    versionsToChange->push_back(2);
    config.setVersion(2);
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);

    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_TRUE(nullptr != defaultInstance);
    EXPECT_EQ(2, defaultInstance->getVersion());
}

TEST_F(ModelDefaultVersions, DefaultVersionShouldReturnHighestNonRetired) {
    MockModel mockModel;
    std::shared_ptr<ovms::model_versions_t> versionsToChange = std::make_shared<ovms::model_versions_t>();
    versionsToChange->push_back(1);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();

    versionsToChange->push_back(2);
    config.setVersion(2);
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();

    versionsToChange->push_back(2);
    mockModel.retireVersions(versionsToChange);
    versionsToChange->clear();

    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_TRUE(nullptr != defaultInstance);
    EXPECT_EQ(1, defaultInstance->getVersion());
}

TEST_F(ModelDefaultVersions, DefaultVersionShouldReturnHighestWhenVersionReloaded) {
    MockModel mockModel;
    std::shared_ptr<ovms::model_versions_t> versionsToChange = std::make_shared<ovms::model_versions_t>();
    versionsToChange->push_back(1);
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();

    versionsToChange->push_back(2);
    config.setVersion(2);
    ASSERT_EQ(mockModel.addVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();

    versionsToChange->push_back(2);
    mockModel.retireVersions(versionsToChange);
    versionsToChange->clear();

    versionsToChange->push_back(2);
    config.setVersion(2);
    ASSERT_EQ(mockModel.reloadVersions(versionsToChange, config), ovms::StatusCode::OK);
    versionsToChange->clear();

    std::shared_ptr<ovms::ModelInstance> defaultInstance;
    defaultInstance = mockModel.getDefaultModelInstance();
    EXPECT_TRUE(nullptr != defaultInstance);
    EXPECT_EQ(2, defaultInstance->getVersion());
}
