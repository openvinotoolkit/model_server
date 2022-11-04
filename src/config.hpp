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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cxxopts.hpp>

namespace ovms {
class GeneralOptionsImpl;
class MultiModelOptionsImpl;

/**
     * @brief Provides all the configuration options from command line
     */
class Config {
private:
    /**
         * @brief A default constructor is private
         */
    Config() = default;

    /**
         * @brief Private copying constructor
         */
    Config(const Config&) = delete;

    /**
         * @brief cxxopts options dictionary definition
         */
    std::unique_ptr<cxxopts::Options> options;

    /**
         * @brief cxxopts contains parsed parameters
         */
    std::unique_ptr<cxxopts::ParseResult> result;

    /**
         * @brief 
         */
    const std::string empty;

    // new
    std::string _configPath;
    uint64_t _port = 9178;
    std::string _cpuExtensionLibraryPath;
    std::string _grpcBindAddress = "0.0.0.0";
    uint64_t _restPort = 0;
    std::string _restBindAddress = "0.0.0.0";
    uint _grpcWorkers = 1;
    uint _restWorkers = 8;  // TODO: In OVMS this is nproc * 4
    std::string _modelName;
    std::string _modelPath;
    std::string _batchSize;
    std::string _shape;
    std::string _layout;
    std::string _modelVersionPolicy;
    uint32_t _nireq = 0;
    std::string _targetDevice;
    std::string _pluginConfig;
    bool _stateful = false;
    bool _metricsEnabled = false;
    std::string _metricsList;
    bool _idleSequenceCleanup = true;
    bool _lowLatencyTransformation = false;
    uint32_t _maxSequenceNumber = 500;
    std::string _logLevel = "DEBUG";
    std::string _logPath;
#ifdef MTR_ENABLED
    std::string _tracePath;
#endif
    std::string _grpcChannelArguments;
    uint _filesystemPollWaitSeconds = 1;
    uint32_t _sequenceCleanerPollWaitMinutes = 5;
    uint32_t _resourcesCleanerPollWaitSeconds = 1;
    std::string _cacheDir;

public:
    /**
         * @brief Gets the instance of the config
         */
    static Config& instance() {
        // TODO to remove singleton
        static Config instance;

        return instance;
    }

    /**
         * @brief Parse the commandline parameters
         * 
         * @param argc 
         * @param argv 
         * @return Config& 
         */
    Config& parse(int argc, char** argv);
    Config& parse(GeneralOptionsImpl*, MultiModelOptionsImpl*);

    /**
         * @brief Validate passed arguments
         * 
         * @return void 
         */
    void validate();

    /**
         * @brief checks if input is a proper hostname or IP address value
         *
         * @return bool
         */
    static bool check_hostname_or_ip(const std::string& input);

    /**
         * @brief Get the config path
         * 
         * @return std::string 
         */
    const std::string& __configPath() const;  // TODO: Move to CLI parser when ready
    const std::string& configPath() const;

    /**
         * @brief Gets the grpc port
         * 
         * @return uint64_t
         */
    uint64_t __port() const;
    uint64_t port() const;

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string
         */
    const std::string __cpuExtensionLibraryPath() const;  // TODO: Move to CLI parser when ready
    const std::string cpuExtensionLibraryPath() const;

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string __grpcBindAddress() const;  // TODO: Move to CLI parser when ready
    const std::string grpcBindAddress() const;

    /**
         * @brief Gets the REST port
         * 
         * @return uint64_t
         */
    uint64_t __restPort() const;
    uint64_t restPort() const;

    /**
         * @brief Get the rest network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string __restBindAddress() const;  // TODO: Move to CLI parser when ready
    const std::string restBindAddress() const;

    /**
         * @brief Gets the gRPC workers count
         * 
         * @return uint
         */
    uint __grpcWorkers() const;  // TODO: Move to CLI parser when ready
    uint grpcWorkers() const;

    /**
         * @brief Gets the rest workers count
         * 
         * @return uint
         */
    uint __restWorkers() const;  // TODO: Move to CLI parser when ready
    uint restWorkers() const;

    /**
         * @brief Get the model name
         * 
         * @return const std::string&
         */
    const std::string& __modelName() const;  // TODO: Move to CLI parser when ready
    const std::string& modelName() const;

    /**
         * @brief Gets the model path
         * 
         * @return const std::string&
         */
    const std::string& __modelPath() const;  // TODO: Move to CLI parser when ready
    const std::string& modelPath() const;

    /**
         * @brief Gets the batch size
         * 
         * @return const std::string&
         */
    const std::string& __batchSize() const;  // TODO: Move to CLI parser when ready
    const std::string& batchSize() const;

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& __shape() const;  // TODO: Move to CLI parser when ready
    const std::string& shape() const;

    /**
         * @brief Get the layout
         * 
         * @return const std::string&
         */
    const std::string& __layout() const;  // TODO: Move to CLI parser when ready
    const std::string& layout() const;

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& __modelVersionPolicy() const;  // TODO: Move to CLI parser when ready
    const std::string& modelVersionPolicy() const;

    /**
         * @brief Get the nireq
         *
         * @return uint 
         */
    uint32_t __nireq() const;
    uint32_t nireq() const;

    /**
         * @brief Get the target device
         * 
         * @return const std::string& 
         */
    const std::string& __targetDevice() const;  // TODO: Move to CLI parser when ready
    const std::string& targetDevice() const;

    /**
         * @brief Get the plugin config
         * 
         * @return const std::string& 
         */
    const std::string& __pluginConfig() const;  // TODO: Move to CLI parser when ready
    const std::string& pluginConfig() const;

    /**
         * @brief Get stateful flag
         *
         * @return bool
         */
    bool __stateful() const;  // TODO: Move to CLI parser when ready
    bool stateful() const;

    /**
     * @brief Get metrics enabled flag
     *
     * @return bool
     */
    bool __metricsEnabled() const;  // TODO: Move to CLI parser when ready
    bool metricsEnabled() const;

    /**
        * @brief Get metrics list
        *
        * @return std::string
        */
    std::string __metricsList() const;  // TODO: Move to CLI parser when ready
    std::string metricsList() const;

    /**
     * @brief Get idle sequence cleanup flag
     *
     * @return uint
     */
    bool __idleSequenceCleanup() const;  // TODO: Move to CLI parser when ready
    bool idleSequenceCleanup() const;

    /**
         * @brief Get low latency transformation flag
         *
         * @return bool
         */
    bool __lowLatencyTransformation() const;  // TODO: Move to CLI parser when ready
    bool lowLatencyTransformation() const;

    /**
     * @brief Get max number of sequences that can be processed concurrently 
     *
     * @return uint
     */
    uint32_t __maxSequenceNumber() const;  // TODO: Move to CLI parser when ready
    uint32_t maxSequenceNumber() const;

    /**
        * @brief Get the log level
        *
        * @return const std::string&
        */
    const std::string& __logLevel() const;  // TODO: Move to CLI parser when ready
    const std::string& logLevel() const;

    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& __logPath() const;  // TODO: Move to CLI parser when ready
    const std::string& logPath() const;

#ifdef MTR_ENABLED
    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& __tracePath() const;  // TODO: Move to CLI parser when ready
    const std::string& tracePath() const;
#endif

    /**
        * @brief Get the plugin config
        *
        * @return const std::string&
        */
    const std::string& __grpcChannelArguments() const;  // TODO: Move to CLI parser when ready
    const std::string& grpcChannelArguments() const;

    /**
     * @brief Get the filesystem poll wait time in seconds
     * 
     * @return uint 
     */
    uint __filesystemPollWaitSeconds() const;  // TODO: Move to CLI parser when ready
    uint filesystemPollWaitSeconds() const;

    /**
     * @brief Get the sequence cleanup poll wait time in minutes
     * 
     * @return uint32_t
     */
    uint32_t __sequenceCleanerPollWaitMinutes() const;  // TODO: Move to CLI parser when ready
    uint32_t sequenceCleanerPollWaitMinutes() const;

    /**
     * @brief Get the resources cleanup poll wait time in seconds
     * 
     * @return uint32_t
     */
    uint32_t __resourcesCleanerPollWaitSeconds() const;  // TODO: Move to CLI parser when ready
    uint32_t resourcesCleanerPollWaitSeconds() const;

    /**
         * @brief Model cache directory
         * 
         * @return const std::string& 
         */
    const std::string __cacheDir() const;  // TODO: Move to CLI parser when ready
    const std::string cacheDir() const;
};
}  // namespace ovms
