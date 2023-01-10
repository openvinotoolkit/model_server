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

TEST(SystemInfoImpl, getCoreCountImplPositive) {
    EXPECT_EQ(1, getCoreCountImpl("1"));
    EXPECT_EQ(1, getCoreCountImpl("3"));

    EXPECT_EQ(2, getCoreCountImpl("0-1"));
    EXPECT_EQ(3, getCoreCountImpl("1-3"));

    EXPECT_EQ(4, getCoreCountImpl("0,2-4"));
    EXPECT_EQ(4, getCoreCountImpl("2-4,9"));

    EXPECT_EQ(7, getCoreCountImpl("2-4,9-12"));
    EXPECT_EQ(10, getCoreCountImpl("2-4,9-12,123-125"));

    EXPECT_EQ(5, getCoreCountImpl("3,8,124,1096,1098"));
    EXPECT_EQ(8, getCoreCountImpl("3,8,124,1096,1098,1099-1101"));
    uint64_t nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    EXPECT_EQ(nprocs, getCoreCount()) << nprocs;
}
TEST(SystemInfoImpl, getCoreCountImplNegative) {
    EXPECT_EQ(1, getCoreCountImpl("-1"));
    EXPECT_EQ(1, getCoreCountImpl("-33"));

    EXPECT_EQ(1, getCoreCountImpl("35-33"));

    EXPECT_EQ(1, getCoreCountImpl("33-35-37"));

    EXPECT_EQ(1, getCoreCountImpl("1234567890123456789012345678901234567890"));
    // TODO test against maximum values
    std::ifstream ifs;
    EXPECT_EQ(getCPUSetFile(ifs, "/sys/fs/illegal_file"), ovms::StatusCode::FILESYSTEM_ERROR);
}

TEST(SystemInfo, getCoreCount) {
    uint16_t cpuCount = getCoreCount();
    EXPECT_GE(cpuCount, 1);
    EXPECT_LE(cpuCount, std::thread::hardware_concurrency());
}
