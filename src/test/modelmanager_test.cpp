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
#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../model.hpp"
#include "../modelmanager.hpp"

using ::testing::_;
using ::testing::Return;

namespace {
// returns path to a file.
std::string createConfigFileWithContent(const std::string& content, std::string filename = "/tmp/ovms_config_file.json") {
    std::ofstream configFile;
    configFile.open(filename);
    configFile << content << std::endl;
    configFile.close();

    return filename;
}

const char* config_1_model = R"({
   "model_config_list": [
    {
      "config": {
        "name": "resnet",
        "base_path": "/tmp/models/dummy1",
        "target_device": "CPU"
      }
   }]
})";

const char* config_2_models = R"({
   "model_config_list": [
    {
      "config": {
        "name": "resnet",
        "base_path": "/tmp/models/dummy1",
        "target_device": "CPU"
      }
    },
    {
      "config": {
        "name": "alpha",
        "base_path": "/tmp/models/dummy2",
        "target_device": "CPU"
      }
    }]
})";

const std::string model_1_path = "/tmp/models/dummy1/1";
const std::string model_2_path = "/tmp/models/dummy2/2";

const std::chrono::duration SLEEP_TIME_S = std::chrono::seconds(2);

}  // namespace

class MockModel : public ovms::Model {
    public:
        MOCK_METHOD(ovms::Status, addVersion, (const ovms::ModelConfig&), (override));
};

std::shared_ptr<MockModel> modelMock;

class MockModelManager : public ovms::ModelManager {
    public:
    std::shared_ptr<ovms::Model> modelFactory() {
        return modelMock;
    }
};

TEST(ModelManager, ConfigParseNoModels) {
    std::string configFile = createConfigFileWithContent("{ \"model_config_list\": [ ] }\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    ovms::Status status = manager.start(configFile);
    EXPECT_EQ(status, ovms::Status::OK);
}

TEST(ModelManager, WrongConfigFile) {
    std::string configFile = "123/tmp/not_a_valid_file_name";
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    ovms::Status status = manager.start(configFile);
    EXPECT_EQ(status, ovms::Status::FILE_INVALID);
}

TEST(ModelManager, ConfigParseEmpty) {
    std::string configFile = createConfigFileWithContent("\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    ovms::Status status = manager.start(configFile);
    EXPECT_EQ(status, ovms::Status::JSON_INVALID);
}

TEST(ModelManager, ConfigNotAJson) {
    std::string configFile = createConfigFileWithContent("abcdfgh");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    ovms::Status status = manager.start(configFile);
    EXPECT_EQ(status, ovms::Status::JSON_INVALID);
}

TEST(ModelManager, ConfigParseEmptyJson) {
    std::string configFile = createConfigFileWithContent("{}\n");
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    ovms::Status status = manager.start(configFile);
    EXPECT_EQ(status, ovms::Status::JSON_INVALID);
}

TEST(ModelManager, ReadsVersionsFromDisk) {
    const std::string path = "/tmp/test_model/";

    for (auto i : {1, 5, 8, 10}) {
        std::filesystem::create_directories(path + std::to_string(i));
    }

    std::filesystem::create_directories(path + "unknown_dir11");  // invalid version directory

    std::vector<ovms::model_version_t> versions;
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.readAvailableVersions(path, versions);

    EXPECT_EQ(status, ovms::Status::OK);
    EXPECT_THAT(versions, ::testing::UnorderedElementsAre(1, 5, 8, 10));
}

TEST(ModelManager, ReadVersionsInvalidPath) {
    const std::string path = "/tmp/inexisting_path/8bt4kv";

    try {
        std::filesystem::remove(path);
    } catch (const std::filesystem::filesystem_error&) {
    }

    std::vector<ovms::model_version_t> versions;
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    auto status = manager.readAvailableVersions(path, versions);

    EXPECT_EQ(status, ovms::Status::PATH_INVALID);
}

TEST(ModelManager, StartFromFile) {
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file1.json";
    createConfigFileWithContent(config_1_model, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    EXPECT_CALL(*modelMock, addVersion(_))
        .Times(1)
        .WillRepeatedly(Return(ovms::Status::OK));
    ovms::Status status = manager.start(fileToReload);
    EXPECT_EQ(status, ovms::Status::OK);
    manager.join();
    modelMock.reset();
}

TEST(ModelManager, ConfigReloading) {
    std::filesystem::create_directories(model_1_path);
    std::filesystem::create_directories(model_2_path);
    std::string fileToReload = "/tmp/ovms_config_file2.json";
    createConfigFileWithContent(config_1_model, fileToReload);
    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;
    EXPECT_CALL(*modelMock, addVersion(_))
        .WillRepeatedly(Return(ovms::Status::OK));

    ovms::Status status = manager.start(fileToReload);
    auto models = manager.getModels().size();
    EXPECT_EQ(models, 1);
    EXPECT_EQ(status, ovms::Status::OK);
    std::thread t([](){
        std::this_thread::sleep_for(SLEEP_TIME_S);
    });
    t.join();

    createConfigFileWithContent(config_2_models, fileToReload);
    std::thread s([](){
        std::this_thread::sleep_for(SLEEP_TIME_S);
    });
    s.join();
    models = manager.getModels().size();
    EXPECT_EQ(models, 2);
    manager.join();
    modelMock.reset();
}
