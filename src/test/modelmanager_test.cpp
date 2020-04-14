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

#include "gtest/gtest.h"

#include "../modelmanager.h"

#include <iostream>
#include <fstream>

namespace {
// returns path to a file.
std::string create_config_file_with_content(const std::string& content) {
   const char* config_file_path = "/tmp/ovms_config_file.json";
   std::ofstream config_file;
   config_file.open(config_file_path);
   config_file << content << std::endl;
   config_file.close(); 
   return config_file_path;
}

} // namespace anon.

TEST(ModelManager,config_parse_no_models)
{
   std::string config_file = create_config_file_with_content("{ \"models\": [ ] }\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(config_file);
   EXPECT_EQ(status,ovms::Status::OK);
}

TEST(ModelManager,config_parse_empty)
{
   std::string config_file = create_config_file_with_content("\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(config_file);
   EXPECT_NE(status,ovms::Status::OK);
}
TEST(ModelManager,config_parse_empty_json)
{
   std::string config_file = create_config_file_with_content("{}\n");
   ovms::ModelManager& manager = ovms::ModelManager::getInstance();
   ovms::Status status = manager.start(config_file);
   EXPECT_NE(status,ovms::Status::OK);
}

