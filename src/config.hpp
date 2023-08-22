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

#include <optional>
#include <string>

#include "capi_frontend/server_settings.hpp"

namespace ovms {

/**
     * @brief Provides all the configuration options from command line
     */
class Config {
protected:
    /**
         * @brief A default constructor is private
         */
    Config() = default;

private:
    /**
         * @brief Private copying constructor
         */
    Config(const Config&) = delete;

    /**
         * @brief 
         */
    const std::string empty;

    ServerSettingsImpl serverSettings;
    ModelsSettingsImpl modelsSettings;

public:
    /**
         * @brief Gets the instance of the config
         */
    static Config& instance() {
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
    bool parse(ServerSettingsImpl*, ModelsSettingsImpl*);

    /**
         * @brief Validate passed arguments
         * 
         * @return void 
         */
    bool validate();

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
    const std::string& configPath() const;

    /**
         * @brief Gets the grpc port
         * 
         * @return uint32_t
         */
    uint32_t port() const;

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string
         */
    const std::string cpuExtensionLibraryPath() const;

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string grpcBindAddress() const;

    /**
         * @brief Gets the REST port
         * 
         * @return uint32_t
         */
    uint32_t restPort() const;

    /**
         * @brief Get the rest network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string restBindAddress() const;

    /**
         * @brief Gets the gRPC workers count
         * 
         * @return uint
         */
    uint32_t grpcWorkers() const;

    /**
         * @brief Gets the rest workers count
         * 
         * @return uint
         */
    uint32_t restWorkers() const;

    /**
         * @brief Get the model name
         * 
         * @return const std::string&
         */
    const std::string& modelName() const;

    /**
         * @brief Gets the model path
         * 
         * @return const std::string&
         */
    const std::string& modelPath() const;

    /**
         * @brief Gets the batch size
         * 
         * @return const std::string&
         */
    const std::string& batchSize() const;

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& shape() const;

    /**
         * @brief Get the layout
         * 
         * @return const std::string&
         */
    const std::string& layout() const;

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& modelVersionPolicy() const;

    /**
         * @brief Get the nireq
         *
         * @return uint 
         */
    uint32_t nireq() const;

    /**
         * @brief Get the target device
         * 
         * @return const std::string& 
         */
    const std::string& targetDevice() const;

    /**
         * @brief Get the plugin config
         * 
         * @return const std::string& 
         */
    const std::string& pluginConfig() const;

    /**
         * @brief Get stateful flag
         *
         * @return bool
         */
    bool stateful() const;

    /**
     * @brief Get metrics enabled flag
     *
     * @return bool
     */
    bool metricsEnabled() const;

    /**
        * @brief Get metrics list
        *
        * @return std::string
        */
    std::string metricsList() const;

    /**
     * @brief Get idle sequence cleanup flag
     *
     * @return uint
     */
    bool idleSequenceCleanup() const;

    /**
         * @brief Get low latency transformation flag
         *
         * @return bool
         */
    bool lowLatencyTransformation() const;

    /**
     * @brief Get max number of sequences that can be processed concurrently 
     *
     * @return uint
     */
    uint32_t maxSequenceNumber() const;

    /**
        * @brief Get the log level
        *
        * @return const std::string&
        */
    const std::string& logLevel() const;

    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& logPath() const;

#ifdef MTR_ENABLED
    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& tracePath() const;
#endif

    /**
        * @brief Get the plugin config
        *
        * @return const std::string&
        */
    const std::string& grpcChannelArguments() const;

    /**
     * @brief Get the filesystem poll wait time in seconds
     * 
     * @return uint 
     */
    uint32_t filesystemPollWaitSeconds() const;

    /**
     * @brief Get the sequence cleanup poll wait time in minutes
     * 
     * @return uint32_t
     */
    uint32_t sequenceCleanerPollWaitMinutes() const;

    /**
     * @brief Get the resources cleanup poll wait time in seconds
     * 
     * @return uint32_t
     */
    uint32_t resourcesCleanerPollWaitSeconds() const;

    /**
         * @brief Model cache directory
         * 
         * @return const std::string& 
         */
    const std::string cacheDir() const;
};
}  // namespace ovms
