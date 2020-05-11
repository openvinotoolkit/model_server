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

#include <iostream>
#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../modelmanager.hpp"


namespace {
// returns path to a file.
std::string createConfigFileWithContent(const std::string& content) {
   const char* configFilePath = "/tmp/ovms_config_file.json";
   std::ofstream configFile;
   configFile.open(configFilePath);
   configFile << content << std::endl;
   configFile.close();
   return configFilePath;
}

}  // namespace

TEST(ModelManager, ConfigParseNoModels) {
   std::string configFile = createConfigFileWithContent("{ \"model_config_list\": [ ] }\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(configFile);
   EXPECT_EQ(status, ovms::Status::OK);
}

TEST(ModelManager, ConfigParseEmpty) {
   std::string configFile = createConfigFileWithContent("\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(configFile);
   EXPECT_NE(status, ovms::Status::OK);
}

TEST(ModelManager, ConfigParseEmptyJson) {
   std::string configFile = createConfigFileWithContent("{}\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(configFile);
   EXPECT_NE(status, ovms::Status::OK);
}

TEST(ModelManager, ReadsVersionsFromDisk) {
   const std::string path = "/tmp/test_model/";

   for (auto i : {1, 5, 8, 10}) {
      std::filesystem::create_directories(path + std::to_string(i));
   }

   std::filesystem::create_directories(path + "unknown_dir11");  // invalid version directory

   std::vector<ovms::model_version_t> versions;
   auto status = ovms::ModelManager::readAvailableVersions(path, versions);

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
   auto status = ovms::ModelManager::readAvailableVersions(path, versions);

   EXPECT_EQ(status, ovms::Status::PATH_INVALID);
}
