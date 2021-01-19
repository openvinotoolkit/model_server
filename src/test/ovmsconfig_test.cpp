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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sysexits.h>

#include "spdlog/spdlog.h"

#include "../config.hpp"

using testing::_;
using testing::ContainerEq;
using testing::Return;
using testing::ReturnRef;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"
class DISABLED_OvmsConfigTest : public ::testing::Test {
public:
    void SetUp() override {
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }
    void TearDown() override {
        std::cout.rdbuf(sbuf);
        sbuf = nullptr;
    }

    ::testing::AssertionResult AssertRegexMessageInOutput(std::string regexMessage) {
        std::string stdOut{buffer.str()};
        std::regex re(regexMessage.c_str());
        std::smatch m;
        bool found = std::regex_search(stdOut, m, re);

        return found ? ::testing::AssertionSuccess() : testing::AssertionFailure() << "message not found.";
    }

    std::stringstream buffer{};
    std::streambuf* sbuf;
};

TEST_F(DISABLED_OvmsConfigTest, bufferTest) {
    std::string input{"Test buffer"};
    std::cout << input;
    std::string check{buffer.str()};
    EXPECT_EQ(input, check);
}

TEST_F(DISABLED_OvmsConfigTest, emptyInput) {
    char* n_argv[] = {"ovms"};
    int arg_count = 1;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(DISABLED_OvmsConfigTest, helpInput) {
    char* n_argv[] = {"ovms", "help"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(DISABLED_OvmsConfigTest, badInput) {
    char* n_argv[] = {"ovms", "--bad_option"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options");
}

TEST_F(DISABLED_OvmsConfigTest, negativeTwoParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--model_name", "some_name"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use either config_path or model_path");
}

TEST_F(DISABLED_OvmsConfigTest, negativeMissingPathAndName) {
    char* n_argv[] = {"ovms", "--rest_port", "8080"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(DISABLED_OvmsConfigTest, negativeMissingName) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(DISABLED_OvmsConfigTest, negativeMissingPath) {
    char* n_argv[] = {"ovms", "--model_name", "model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(DISABLED_OvmsConfigTest, negativeSamePorts) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8080", "--port", "8080"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port and rest_port cannot");
}

TEST_F(DISABLED_OvmsConfigTest, negativeMultiParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--batch_size", "10"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Model parameters in CLI are exclusive");
}

TEST_F(DISABLED_OvmsConfigTest, missingParams) {
    char* n_argv[] = {"ovms", "--batch_size", "10"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(DISABLED_OvmsConfigTest, negativePortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "-1"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options: Argument ‘-1’");
}

TEST_F(DISABLED_OvmsConfigTest, negativeRestPortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "-1"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options: Argument ‘-1’ ");
}

TEST_F(DISABLED_OvmsConfigTest, negativePortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range from 0");
}

TEST_F(DISABLED_OvmsConfigTest, negativeRestPortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range from 0");
}

TEST_F(DISABLED_OvmsConfigTest, negativePortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range");
}

TEST_F(DISABLED_OvmsConfigTest, negativeRestPortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port number out of range");
}

TEST_F(DISABLED_OvmsConfigTest, negativeGrpcWorkersMax) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--grpc_workers", "10000"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "grpc_workers count should be from 1");
}

TEST_F(DISABLED_OvmsConfigTest, negativeUint64Max) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "0xffffffffffffffff"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port number out of range from 0 to 65535");
}

class OvmsParamsTest : public ::testing::Test {
};

TEST_F(OvmsParamsTest, hostname_ip_regex) {
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("0.0.0.0"), true);
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("127.0.0.1"), true);
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("localhost"), true);
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("example.com"), true);
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("    "), false);
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip("(%$#*F"), false);
    std::string too_long(256, 'a');
    EXPECT_EQ(ovms::Config::instance().check_hostname_or_ip(too_long), false);
}

#pragma GCC diagnostic pop
