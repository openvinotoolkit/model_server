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
    const std::string& __configPath() const {
        if (result->count("config_path"))
            return result->operator[]("config_path").as<std::string>();
        return empty;
    }
    const std::string& configPath() const { return _configPath; }

    /**
         * @brief Gets the grpc port
         * 
         * @return uint64_t
         */
    uint64_t __port() const {
        return result->operator[]("port").as<uint64_t>();
    }
    uint64_t port() const { return _port; }

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string
         */
    const std::string __cpuExtensionLibraryPath() const {
        if (result != nullptr && result->count("cpu_extension")) {
            return result->operator[]("cpu_extension").as<std::string>();
        }
        return "";
    }
    const std::string cpuExtensionLibraryPath() const { return _cpuExtensionLibraryPath; }

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string __grpcBindAddress() const {
        if (result->count("grpc_bind_address"))
            return result->operator[]("grpc_bind_address").as<std::string>();
        return "0.0.0.0";
    }
    const std::string grpcBindAddress() const { return _grpcBindAddress; }

    /**
         * @brief Gets the REST port
         * 
         * @return uint64_t
         */
    uint64_t __restPort() const {
        return result->operator[]("rest_port").as<uint64_t>();
    }
    uint64_t restPort() const { return _restPort; }

    /**
         * @brief Get the rest network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string __restBindAddress() const {
        if (result->count("rest_bind_address"))
            return result->operator[]("rest_bind_address").as<std::string>();
        return "0.0.0.0";
    }
    const std::string restBindAddress() const { return _restBindAddress; }

    /**
         * @brief Gets the gRPC workers count
         * 
         * @return uint
         */
    uint __grpcWorkers() const {
        return result->operator[]("grpc_workers").as<uint>();
    }
    uint grpcWorkers() const { return _grpcWorkers; }

    /**
         * @brief Gets the rest workers count
         * 
         * @return uint
         */
    uint __restWorkers() const {
        return result->operator[]("rest_workers").as<uint>();
    }
    uint restWorkers() const { return _restWorkers; }

    /**
         * @brief Get the model name
         * 
         * @return const std::string&
         */
    const std::string& __modelName() const {
        if (result->count("model_name"))
            return result->operator[]("model_name").as<std::string>();
        return empty;
    }
    const std::string& modelName() const { return _modelName; }

    /**
         * @brief Gets the model path
         * 
         * @return const std::string&
         */
    const std::string& __modelPath() const {
        if (result->count("model_path"))
            return result->operator[]("model_path").as<std::string>();
        return empty;
    }
    const std::string& modelPath() const { return _modelPath; }

    /**
         * @brief Gets the batch size
         * 
         * @return const std::string&
         */
    const std::string& __batchSize() const {
        if (!result->count("batch_size")) {
            static const std::string d = "0";
            return d;
        }
        return result->operator[]("batch_size").as<std::string>();
    }
    const std::string& batchSize() const { return _batchSize; }

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& __shape() const {
        if (result->count("shape"))
            return result->operator[]("shape").as<std::string>();
        return empty;
    }
    const std::string& shape() const { return _shape; }

    /**
         * @brief Get the layout
         * 
         * @return const std::string&
         */
    const std::string& __layout() const {
        if (result->count("layout"))
            return result->operator[]("layout").as<std::string>();
        return empty;
    }
    const std::string& layout() const { return _layout; }

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& __modelVersionPolicy() const {
        if (result->count("model_version_policy"))
            return result->operator[]("model_version_policy").as<std::string>();
        return empty;
    }
    const std::string& modelVersionPolicy() const { return _modelVersionPolicy; }

    /**
         * @brief Get the nireq
         *
         * @return uint 
         */
    uint32_t __nireq() const {
        if (!result->count("nireq")) {
            return 0;
        }
        return result->operator[]("nireq").as<uint32_t>();
    }
    uint32_t nireq() const { return _nireq; }

    /**
         * @brief Get the target device
         * 
         * @return const std::string& 
         */
    const std::string& __targetDevice() const {
        return result->operator[]("target_device").as<std::string>();
    }
    const std::string& targetDevice() const { return _targetDevice; }

    /**
         * @brief Get the plugin config
         * 
         * @return const std::string& 
         */
    const std::string& __pluginConfig() const {
        if (result->count("plugin_config"))
            return result->operator[]("plugin_config").as<std::string>();
        return empty;
    }
    const std::string& pluginConfig() const { return _pluginConfig; }

    /**
         * @brief Get stateful flag
         *
         * @return bool
         */
    bool __stateful() const {
        return result->operator[]("stateful").as<bool>();
    }
    bool stateful() const { return _stateful; }

    /**
     * @brief Get metrics enabled flag
     *
     * @return bool
     */
    bool __metricsEnabled() const {
        if (!result->count("metrics_enable")) {
            return false;
        }
        return result->operator[]("metrics_enable").as<bool>();
    }
    bool metricsEnabled() const { return _metricsEnabled; }

    /**
        * @brief Get metrics list
        *
        * @return std::string
        */
    std::string __metricsList() const {
        if (!result->count("metrics_list")) {
            return std::string("");
        }
        return result->operator[]("metrics_list").as<std::string>();
    }
    std::string metricsList() const { return _metricsList; }

    /**
     * @brief Get idle sequence cleanup flag
     *
     * @return uint
     */
    bool __idleSequenceCleanup() const {
        return result->operator[]("idle_sequence_cleanup").as<bool>();
    }
    bool idleSequenceCleanup() const { return _idleSequenceCleanup; }

    /**
         * @brief Get low latency transformation flag
         *
         * @return bool
         */
    bool __lowLatencyTransformation() const {
        return result->operator[]("low_latency_transformation").as<bool>();
    }
    bool lowLatencyTransformation() const { return _lowLatencyTransformation; }

    /**
     * @brief Get max number of sequences that can be processed concurrently 
     *
     * @return uint
     */
    uint32_t __maxSequenceNumber() const;
    uint32_t maxSequenceNumber() const;

    /**
        * @brief Get the log level
        *
        * @return const std::string&
        */
    const std::string& __logLevel() const {
        if (result->count("log_level"))
            return result->operator[]("log_level").as<std::string>();
        return empty;
    }
    const std::string& logLevel() const { return _logLevel; }

    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& __logPath() const {
        if (result->count("log_path"))
            return result->operator[]("log_path").as<std::string>();
        return empty;
    }
    const std::string& logPath() const { return _logPath; }

#ifdef MTR_ENABLED
    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& __tracePath() const {
        if (result->count("trace_path"))
            return result->operator[]("trace_path").as<std::string>();
        return empty;
    }
    const std::string& tracePath() const { return _tracePath; }
#endif

    /**
        * @brief Get the plugin config
        *
        * @return const std::string&
        */
    const std::string& __grpcChannelArguments() const {
        if (result->count("grpc_channel_arguments"))
            return result->operator[]("grpc_channel_arguments").as<std::string>();
        return empty;
    }
    const std::string& grpcChannelArguments() const { return _grpcChannelArguments; }

    /**
     * @brief Get the filesystem poll wait time in seconds
     * 
     * @return uint 
     */
    uint __filesystemPollWaitSeconds() const {
        return result->operator[]("file_system_poll_wait_seconds").as<uint>();
    }
    uint filesystemPollWaitSeconds() const { return _filesystemPollWaitSeconds; }

    /**
     * @brief Get the sequence cleanup poll wait time in minutes
     * 
     * @return uint32_t
     */
    uint32_t __sequenceCleanerPollWaitMinutes() const {
        return result->operator[]("sequence_cleaner_poll_wait_minutes").as<uint32_t>();
    }
    uint32_t sequenceCleanerPollWaitMinutes() const { return _sequenceCleanerPollWaitMinutes; }

    /**
     * @brief Get the resources cleanup poll wait time in seconds
     * 
     * @return uint32_t
     */
    uint32_t __resourcesCleanerPollWaitSeconds() const {
        return result->operator[]("custom_node_resources_cleaner_interval").as<uint32_t>();
    }
    uint32_t resourcesCleanerPollWaitSeconds() const { return _resourcesCleanerPollWaitSeconds; }

    /**
         * @brief Model cache directory
         * 
         * @return const std::string& 
         */
    const std::string __cacheDir() const {
        if (result != nullptr && result->count("cache_dir")) {
            return result->operator[]("cache_dir").as<std::string>();
        }
        return "";
    }
    const std::string cacheDir() const { return _cacheDir; }
};
}  // namespace ovms
