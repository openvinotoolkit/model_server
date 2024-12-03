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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <drogon/drogon.h>

#include <thread>
#include <chrono>


// Disabled due to drogon issue https://github.com/drogonframework/drogon/issues/2210
TEST(Drogon, DISABLED_basic) {
    for (int i = 0; i < 2; i++) {
        std::thread k([] {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            drogon::app().quit();
        });
        drogon::app()
                    .setThreadNum(3)
                    .setIdleConnectionTimeout(0)
                    .addListener("0.0.0.0", 11933)
                    .run();
        k.join();
    }
}