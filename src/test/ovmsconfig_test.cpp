//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
class OvmsConfigDeathTest : public ::testing::Test {
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

TEST_F(OvmsConfigDeathTest, bufferTest) {
    std::string input{"Test buffer"};
    std::cout << input;
    std::string check{buffer.str()};
    EXPECT_EQ(input, check);
}

TEST_F(OvmsConfigDeathTest, emptyInput) {
    char* n_argv[] = {"ovms"};
    int arg_count = 1;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(OvmsConfigDeathTest, helpInput) {
    char* n_argv[] = {"ovms", "--help"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(OvmsConfigDeathTest, versionInput) {
    char* n_argv[] = {"ovms", "--version"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_OK), "");
}

TEST_F(OvmsConfigDeathTest, badInput) {
    char* n_argv[] = {"ovms", "--bad_option"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options");
}

TEST_F(OvmsConfigDeathTest, negativeTwoParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--model_name", "some_name"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use either config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeMissingPathAndName) {
    char* n_argv[] = {"ovms", "--rest_port", "8080"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, metricMissingPort) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model", "--model_name", "some_name", "--metrics_enable"};
    int arg_count = 6;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port setting is missing, metrics are enabled on rest port");
}

TEST_F(OvmsConfigDeathTest, metricEnableMissing) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model", "--model_name", "some_name", "--metrics_list", "metric1,metric2"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "metrics_enable setting is missing, required when metrics_list is provided");
}

TEST_F(OvmsConfigDeathTest, metricEnablingInCli) {
    char* n_argv[] = {"ovms", "--config_path", "/path/to/config", "--metrics_enable"};
    int arg_count = 4;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port setting is missing, metrics are enabled on rest port");
}

TEST_F(OvmsConfigDeathTest, metricListInCli) {
    char* n_argv[] = {"ovms", "--config_path", "/path/to/config", "--metrics_list", "metric1,metric2"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "metrics_enable or metrics_list and config_path cant be used together. Use json config file to enable metrics when using config_path.");
}

TEST_F(OvmsConfigDeathTest, metricEnablingInCliWithPort) {
    char* n_argv[] = {"ovms", "--config_path", "/path/to/config", "--metrics_enable", "--rest_port", "8080"};
    int arg_count = 6;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "metrics_enable or metrics_list and config_path cant be used together. Use json config file to enable metrics when using config_path.");
}

TEST_F(OvmsConfigDeathTest, metricListAndEnableInCli) {
    char* n_argv[] = {"ovms", "--config_path", "/path/to/config", "--metrics_list", "metric1,metric2", "--metrics_enable", "--rest_port", "8080"};
    int arg_count = 8;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "metrics_enable or metrics_list and config_path cant be used together. Use json config file to enable metrics when using config_path.");
}

TEST_F(OvmsConfigDeathTest, negativeMissingName) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeMissingPath) {
    char* n_argv[] = {"ovms", "--model_name", "model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeSamePorts) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8080", "--port", "8080"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port and rest_port cannot");
}

TEST_F(OvmsConfigDeathTest, restWorkersTooLarge) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8080", "--port", "8080", "--rest_workers", "100001"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_workers count should be from 2 to ");
}

TEST_F(OvmsConfigDeathTest, restWorkersDefinedRestPortUndefined) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "8080", "--rest_workers", "60"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_workers is set but rest_port is not set");
}

TEST_F(OvmsConfigDeathTest, invalidRestBindAddress) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8081", "--port", "8080", "--rest_bind_address", "192.0.2"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_bind_address has invalid format");
}

TEST_F(OvmsConfigDeathTest, invalidGrpcBindAddress) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "8080", "--grpc_bind_address", "192.0.2"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "grpc_bind_address has invalid format");
}

TEST_F(OvmsConfigDeathTest, negativeMultiParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--batch_size", "10"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Model parameters in CLI are exclusive");
}

TEST_F(OvmsConfigDeathTest, missingParams) {
    char* n_argv[] = {"ovms", "--batch_size", "10"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativePortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "-1"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options: Argument ‘-1’");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "-1"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "error parsing options: Argument ‘-1’ ");
}

TEST_F(OvmsConfigDeathTest, negativePortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range from 0");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range from 0");
}

TEST_F(OvmsConfigDeathTest, negativePortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "port number out of range");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port number out of range");
}

TEST_F(OvmsConfigDeathTest, negativeGrpcWorkersMax) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--grpc_workers", "10000"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "grpc_workers count should be from 1");
}

TEST_F(OvmsConfigDeathTest, cpuExtensionMissingPath) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--cpu_extension", "/wrong/dir"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "File path provided as an --cpu_extension parameter does not exists in the filesystem");
}

TEST_F(OvmsConfigDeathTest, nonExistingLogLevel) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--log_level", "WRONG"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "log_level should be one of");
}

TEST_F(OvmsConfigDeathTest, lowLatencyUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--low_latency_transformation"};
    int arg_count = 6;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, maxSequenceNumberUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--max_sequence_number", "325"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, idleSequenceCleanupUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--idle_sequence_cleanup"};
    int arg_count = 6;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, negativeUint64Max) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "0xffffffffffffffff"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(EX_USAGE), "rest_port number out of range from 0 to 65535");
}

class OvmsParamsTest : public ::testing::Test {
};

TEST_F(OvmsParamsTest, hostname_ip_regex) {
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("0.0.0.0"), true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("127.0.0.1"), true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("localhost"), true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("example.com"), true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("    "), false);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("(%$#*F"), false);
    std::string too_long(256, 'a');
    EXPECT_EQ(ovms::Config::check_hostname_or_ip(too_long), false);
}

TEST(OvmsConfigTest, positive) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "44", "--rest_port", "45"};
    int arg_count = 7;
    ovms::Config::instance().parse(arg_count, n_argv);
    EXPECT_EQ(ovms::Config::instance().port(), 44);
    EXPECT_EQ(ovms::Config::instance().restPort(), 45);
    EXPECT_EQ(ovms::Config::instance().configPath(), "/path1");
}

#pragma GCC diagnostic pop
