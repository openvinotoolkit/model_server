//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include "../status.hpp"
#include "../systeminfo.hpp"
#include "../systeminfo_impl.hpp"

using namespace testing;
using ovms::getCoreCount;
using ovms::getCoreCountImpl;
using ovms::getCPUSetFile;
using ovms::StatusCode;

TEST(SystemInfoImpl, getCoreCountImplPositive) {
    uint16_t coreCount = 42;
    EXPECT_EQ(getCoreCountImpl("1", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 1);
    EXPECT_EQ(getCoreCountImpl("3", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 1);

    EXPECT_EQ(getCoreCountImpl("0-1", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 2);
    EXPECT_EQ(getCoreCountImpl("1-3", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 3);

    EXPECT_EQ(getCoreCountImpl("0,2-4", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 4);
    EXPECT_EQ(getCoreCountImpl("2-4,9", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 4);

    EXPECT_EQ(getCoreCountImpl("2-4,9-12", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 7);
    EXPECT_EQ(getCoreCountImpl("2-4,9-12,123-125", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 10);

    EXPECT_EQ(getCoreCountImpl("3,8,124,1096,1098", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 5);
    EXPECT_EQ(getCoreCountImpl("3,8,124,1096,1098,1099-1101", coreCount), StatusCode::OK);
    EXPECT_EQ(coreCount, 8);
    uint64_t nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    EXPECT_EQ(nprocs, getCoreCount()) << nprocs;
}
TEST(SystemInfoImpl, getCoreCountImplNegative) {
    uint16_t coreCount = 42;
    EXPECT_EQ(getCoreCountImpl("-1", coreCount), StatusCode::FILESYSTEM_ERROR);
    EXPECT_EQ(coreCount, 42);
    EXPECT_EQ(getCoreCountImpl("-33", coreCount), StatusCode::FILESYSTEM_ERROR);
    EXPECT_EQ(coreCount, 42);

    EXPECT_EQ(getCoreCountImpl("35-33", coreCount), StatusCode::FILESYSTEM_ERROR);
    EXPECT_EQ(coreCount, 42);

    EXPECT_EQ(getCoreCountImpl("33-35-37", coreCount), StatusCode::FILESYSTEM_ERROR);
    EXPECT_EQ(coreCount, 42);

    EXPECT_EQ(getCoreCountImpl("1234567890123456789012345678901234567890", coreCount), StatusCode::FILESYSTEM_ERROR);
    EXPECT_EQ(coreCount, 42);
    // TODO test against maximum values
    std::ifstream ifs;
    EXPECT_EQ(getCPUSetFile(ifs, "/sys/fs/illegal_file"), StatusCode::FILESYSTEM_ERROR);
}

TEST(SystemInfo, getCoreCount) {
    uint16_t cpuCount = getCoreCount();
    EXPECT_GE(cpuCount, 1);
    EXPECT_LE(cpuCount, std::thread::hardware_concurrency());
}
