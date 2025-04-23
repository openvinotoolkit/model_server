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

#include "spdlog/spdlog.h"

#include "../config.hpp"
#include "../ovms_exit_codes.hpp"
#include "../systeminfo.hpp"
#include "test_utils.hpp"

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
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(OvmsConfigDeathTest, helpInput) {
    char* n_argv[] = {"ovms", "--help"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_OK), "");

    // EXPECT_TRUE(AssertRegexMessageInOutput(std::string("config_path")));
}

TEST_F(OvmsConfigDeathTest, versionInput) {
    char* n_argv[] = {"ovms", "--version"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_OK), "");
}

TEST_F(OvmsConfigDeathTest, badInput) {
    char* n_argv[] = {"ovms", "--bad_option"};
    int arg_count = 2;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "error parsing options");
}

TEST_F(OvmsConfigDeathTest, negativeTwoParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--model_name", "some_name"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Use either config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithBatchSize) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--batch_size", "5"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithShape) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--shape", "(1,2)"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithNireq) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--nireq", "3"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithModelVersionPolicy) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--model_version_policy", "policy"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithTargetDevice) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--target_device", "GPU"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, OVMSDuplicatedMetricsConfig) {
    char* n_argv[] = {"ovms", "--config_path", "/path/to/config", "--metrics_enable", "--rest_port", "8080"};
    int arg_count = 6;
    ovms::Config::instance().parse(arg_count, n_argv);
    EXPECT_TRUE(ovms::Config::instance().validate());
}

TEST_F(OvmsConfigDeathTest, negativeConfigPathWithPluginConfig) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--plugin_config", "setting"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive with the config file");
}

TEST_F(OvmsConfigDeathTest, negativeMissingPathAndName) {
    char* n_argv[] = {"ovms", "--rest_port", "8080"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, metricMissingRestPort) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model", "--model_name", "some_name", "--metrics_enable"};
    int arg_count = 6;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_port setting is missing, metrics are enabled on rest port");
}

TEST_F(OvmsConfigDeathTest, metricEnableMissing) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model", "--model_name", "some_name", "--metrics_list", "metric1,metric2"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "metrics_enable setting is missing, required when metrics_list is provided");
}

TEST_F(OvmsConfigDeathTest, negativeMissingName) {
    char* n_argv[] = {"ovms", "--model_path", "/path/to/model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeMissingPath) {
    char* n_argv[] = {"ovms", "--model_name", "model"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativeSamePorts) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8080", "--port", "8080"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "port and rest_port cannot");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortGrpcPortBothNotSet) {
    char* n_argv[] = {"ovms", "--config_path", "/path1"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "port and rest_port cannot");
}

TEST_F(OvmsConfigDeathTest, restWorkersTooLarge) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8080", "--port", "8081", "--rest_workers", "100001"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_workers count should be from 2 to ");
}

TEST_F(OvmsConfigDeathTest, restWorkersDefinedRestPortUndefined) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "8080", "--rest_workers", "60"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_workers is set but rest_port is not set");
}

TEST_F(OvmsConfigDeathTest, invalidRestBindAddress) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "8081", "--port", "8080", "--rest_bind_address", "192.0.2"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_bind_address has invalid format");
}

TEST_F(OvmsConfigDeathTest, invalidGrpcBindAddress) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "8080", "--grpc_bind_address", "192.0.2"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "grpc_bind_address has invalid format");
}

TEST_F(OvmsConfigDeathTest, negativeMultiParams) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--batch_size", "10"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Model parameters in CLI are exclusive");
}

TEST_F(OvmsConfigDeathTest, missingParams) {
    char* n_argv[] = {"ovms", "--batch_size", "10"};
    int arg_count = 3;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "Use config_path or model_path");
}

TEST_F(OvmsConfigDeathTest, negativePortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "-1"};
    int arg_count = 5;
#ifdef __linux__
    std::string error = "‘-1’";
#elif _WIN32
    std::string error = "";
#endif
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "error parsing options: Argument " + error);
}

TEST_F(OvmsConfigDeathTest, negativeRestPortMin) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "-1"};
    int arg_count = 5;
#ifdef __linux__
    std::string error = "‘-1’";
#elif _WIN32
    std::string error = "";
#endif
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "error parsing options: Argument " + error);
}

TEST_F(OvmsConfigDeathTest, negativePortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "port number out of range from 0");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortRange) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "65536"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "port number out of range from 0");
}

TEST_F(OvmsConfigDeathTest, negativePortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "port number out of range");
}

TEST_F(OvmsConfigDeathTest, negativeRestPortMax) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "72817"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_port number out of range");
}

TEST_F(OvmsConfigDeathTest, negativeGrpcWorkersMax) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--grpc_workers", "10000"};
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "grpc_workers count should be from 1");
}

TEST_F(OvmsConfigDeathTest, cpuExtensionMissingPath) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--cpu_extension", "/wrong/dir", "--port", "9178"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "File path provided as an --cpu_extension parameter does not exist in the filesystem");
}

TEST_F(OvmsConfigDeathTest, nonExistingLogLevel) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--log_level", "WRONG", "--port", "9178"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "log_level should be one of");
}

TEST_F(OvmsConfigDeathTest, lowLatencyUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--low_latency_transformation", "--port", "9178"};
    int arg_count = 8;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, maxSequenceNumberUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--max_sequence_number", "325", "--port", "9178"};
    int arg_count = 9;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, idleSequenceCleanupUsedForNonStateful) {
    char* n_argv[] = {"ovms", "--model_path", "/path1", "--model_name", "model", "--idle_sequence_cleanup", "--port", "9178"};
    int arg_count = 8;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "require setting stateful flag for the model");
}

TEST_F(OvmsConfigDeathTest, RestPortNegativeUint64Max) {
    char* n_argv[] = {"ovms", "--config_path", "/path1", "--rest_port", "0xffffffffffffffff"};
    int arg_count = 5;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "rest_port number out of range from 0 to 65535");
}

TEST_F(OvmsConfigDeathTest, negativeMissingDashes) {
    char* n_argv[] = {
        "ovms",
        "--config_path",
        "/config.json",
        "--port",
        "44",
        "grpc_workers",
        "2",
    };
    int arg_count = 7;
    EXPECT_EXIT(ovms::Config::instance().parse(arg_count, n_argv), ::testing::ExitedWithCode(OVMS_EX_USAGE), "error parsing options - unmatched arguments: grpc_workers, 2, ");
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
    // Uncompressed IPv6 address
    EXPECT_EQ(ovms::Config::check_hostname_or_ip(
                "fe80:0000:0000:0000:0202:b3ff:fe1e:8329", true);
    // Zero compressed IPv6 address
    EXPECT_EQ(ovms::Config::check_hostname_or_ip(
                "2001:db8:85a3::8a2e:370:7334", true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("::1", true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("::", true);
    // Link-local IPv6 with zone index (RFC 4007 § 11) - unsupported
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("fe80::1234%eth0", false);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("fe80::1234%1", false);
    // IPv4-Embedded IPv6 addresses
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("64:ff9b::192.0.2.33", true);
    EXPECT_EQ(ovms::Config::check_hostname_or_ip(
                "2001:db8:122:344::192.0.2.33", true);
    // IPv4-mapped IPv6 addresses
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("::ffff:192.0.2.128", true);
    //  IPv4-translated IPv6 addresses
    EXPECT_EQ(ovms::Config::check_hostname_or_ip("::ffff:0:192.0.2.128", true);
}

TEST(OvmsConfigTest, positiveMulti) {
#ifdef _WIN32
    const std::string cpu_extension_lib_path = "tmp_cpu_extension_library_dir";
    std::filesystem::create_directory(cpu_extension_lib_path);
#endif

    char* n_argv[] = {"ovms",
        "--port", "44",
        "--rest_workers", "46",
        "--grpc_bind_address", "1.1.1.1",
        "--rest_port", "45",
        "--rest_bind_address", "2.2.2.2",
        "--grpc_channel_arguments", "grpc_channel_args",
        "--file_system_poll_wait_seconds", "2",
        "--sequence_cleaner_poll_wait_minutes", "7",
        "--custom_node_resources_cleaner_interval_seconds", "8",
#ifdef _WIN32
        "--grpc_workers", "1",
        "--cpu_extension", "tmp_cpu_extension_library_dir",
#else
        "--cpu_extension", "/ovms",
        "--grpc_workers", "2",
#endif
        "--cache_dir", "/tmp/model_cache",
        "--log_path", "/tmp/log_path",
        "--log_level", "ERROR",
        "--grpc_max_threads", "100",
        "--grpc_memory_quota", "1000000",
        "--config_path", "/config.json"};

    int arg_count = 35;
    ConstructorEnabledConfig config;
    config.parse(arg_count, n_argv);

    EXPECT_EQ(config.port(), 44);
    EXPECT_EQ(config.restWorkers(), 46);
    EXPECT_EQ(config.grpcBindAddress(), "1.1.1.1");
    EXPECT_EQ(config.restPort(), 45);
    EXPECT_EQ(config.restBindAddress(), "2.2.2.2");
    EXPECT_EQ(config.grpcChannelArguments(), "grpc_channel_args");
    EXPECT_EQ(config.filesystemPollWaitMilliseconds(), 2000);
    EXPECT_EQ(config.sequenceCleanerPollWaitMinutes(), 7);
    EXPECT_EQ(config.resourcesCleanerPollWaitSeconds(), 8);
#ifdef _WIN32
    EXPECT_EQ(config.cpuExtensionLibraryPath(), cpu_extension_lib_path);
    EXPECT_EQ(config.grpcWorkers(), 1);
#else
    EXPECT_EQ(config.cpuExtensionLibraryPath(), "/ovms");

    EXPECT_EQ(config.grpcWorkers(), 2);
#endif
    EXPECT_EQ(config.cacheDir(), "/tmp/model_cache");
    EXPECT_EQ(config.logPath(), "/tmp/log_path");
    EXPECT_EQ(config.logLevel(), "ERROR");
    EXPECT_EQ(config.configPath(), "/config.json");
    EXPECT_EQ(config.grpcMaxThreads(), 100);
    EXPECT_EQ(config.grpcMemoryQuota(), (size_t)1000000);

#ifdef _WIN32
    std::filesystem::remove_all(cpu_extension_lib_path);
#endif
}

TEST(OvmsConfigTest, positiveSingle) {
#ifdef _WIN32
    const std::string cpu_extension_lib_path = "tmp_cpu_extension_library_dir";
    std::filesystem::create_directory(cpu_extension_lib_path);
#endif

    char* n_argv[] = {
        "ovms",
        "--port",
        "44",
        "--rest_workers",
        "46",
        "--grpc_bind_address",
        "1.1.1.1",
        "--rest_port",
        "45",
        "--rest_bind_address",
        "2.2.2.2",
        "--grpc_channel_arguments",
        "grpc_channel_args",
        "--file_system_poll_wait_seconds",
        "2",
        "--sequence_cleaner_poll_wait_minutes",
        "7",
        "--custom_node_resources_cleaner_interval_seconds",
        "8",
#ifdef _WIN32
        "--cpu_extension",
        "tmp_cpu_extension_library_dir",
        "--grpc_workers",
        "1",
#else
        "--cpu_extension",
        "/ovms",
        "--grpc_workers",
        "2",
#endif
        "--cache_dir",
        "/tmp/model_cache",
        "--log_path",
        "/tmp/log_path",
        "--log_level",
        "ERROR",
        "--model_name",
        "model",
        "--model_path",
        "/path",
        "--batch_size",
        "(3:5)",
        "--shape",
        "(3:5,5:6)",
        "--layout",
        "nchw:nhwc",
        "--model_version_policy",
        "setting",
        "--nireq",
        "2",
        "--target_device",
        "GPU",
        "--plugin_config",
        "pluginsetting",
        "--stateful",
        "--metrics_enable",
        "--metrics_list",
        "ovms_streams,ovms_other",
        "--idle_sequence_cleanup=false",
        "--low_latency_transformation",
        "--max_sequence_number",
        "52",
    };
    int arg_count = 55;
    ConstructorEnabledConfig config;
    config.parse(arg_count, n_argv);

    EXPECT_EQ(config.port(), 44);
    EXPECT_EQ(config.restWorkers(), 46);
    EXPECT_EQ(config.grpcBindAddress(), "1.1.1.1");
    EXPECT_EQ(config.restPort(), 45);
    EXPECT_EQ(config.restBindAddress(), "2.2.2.2");
    EXPECT_EQ(config.grpcChannelArguments(), "grpc_channel_args");
    EXPECT_EQ(config.filesystemPollWaitMilliseconds(), 2000);
    EXPECT_EQ(config.sequenceCleanerPollWaitMinutes(), 7);
    EXPECT_EQ(config.resourcesCleanerPollWaitSeconds(), 8);
#ifdef _WIN32
    EXPECT_EQ(config.cpuExtensionLibraryPath(), cpu_extension_lib_path);
    EXPECT_EQ(config.grpcWorkers(), 1);
#else
    EXPECT_EQ(config.cpuExtensionLibraryPath(), "/ovms");
    EXPECT_EQ(config.grpcWorkers(), 2);
#endif
    EXPECT_EQ(config.cacheDir(), "/tmp/model_cache");
    EXPECT_EQ(config.logPath(), "/tmp/log_path");
    EXPECT_EQ(config.logLevel(), "ERROR");
    EXPECT_EQ(config.modelPath(), "/path");
    EXPECT_EQ(config.modelName(), "model");
    EXPECT_EQ(config.batchSize(), "(3:5)");
    EXPECT_EQ(config.shape(), "(3:5,5:6)");
    EXPECT_EQ(config.layout(), "nchw:nhwc");
    EXPECT_EQ(config.modelVersionPolicy(), "setting");
    EXPECT_EQ(config.nireq(), 2);
    EXPECT_EQ(config.targetDevice(), "GPU");
    EXPECT_EQ(config.pluginConfig(), "pluginsetting");
    EXPECT_EQ(config.stateful(), true);
    EXPECT_EQ(config.metricsEnabled(), true);
    EXPECT_EQ(config.metricsList(), "ovms_streams,ovms_other");
    EXPECT_EQ(config.idleSequenceCleanup(), false);
    EXPECT_EQ(config.lowLatencyTransformation(), true);
    EXPECT_EQ(config.maxSequenceNumber(), 52);
    EXPECT_EQ(config.grpcMaxThreads(), ovms::getCoreCount() * 8.0);
    EXPECT_EQ(config.grpcMemoryQuota(), (size_t)2 * 1024 * 1024 * 1024);

#ifdef _WIN32
    std::filesystem::remove_all(cpu_extension_lib_path);
#endif
}

#pragma GCC diagnostic pop
