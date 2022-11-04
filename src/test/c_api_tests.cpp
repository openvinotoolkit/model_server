//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <gtest/gtest.h>

#include "../config.hpp"
#include "../poc_api_impl.hpp"

using namespace ovms;

class CapiConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(CapiConfigTest, Parse) {
    GeneralOptionsImpl go;
    MultiModelOptionsImpl mmo;

    go.grpcPort = 123;
    go.restPort = 234;
    mmo.configPath = "/path/config.json";

    ovms::Config::instance().parse(&go, &mmo);
    EXPECT_EQ(ovms::Config::instance().port(), 123);
    EXPECT_EQ(ovms::Config::instance().restPort(), 234);
    EXPECT_EQ(ovms::Config::instance().configPath(), "/path/config.json");
}
