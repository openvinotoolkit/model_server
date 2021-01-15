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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cxxopts.hpp>

namespace ovms {
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
    bool check_hostname_or_ip(const std::string& input);

    /**
         * @brief Get the config path
         * 
         * @return std::string 
         */
    const std::string& configPath() {
        if (result->count("config_path"))
            return result->operator[]("config_path").as<std::string>();
        return empty;
    }

    /**
         * @brief Gets the grpc port
         * 
         * @return uint64_t
         */
    uint64_t port() {
        return result->operator[]("port").as<uint64_t>();
    }

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string
         */
    const std::string cpuExtensionLibraryPath() {
        if (result != nullptr && result->count("cpu_extension")) {
            return result->operator[]("cpu_extension").as<std::string>();
        }
        return "";
    }

    /**
         * @brief Get the gRPC network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string grpcBindAddress() {
        if (result->count("grpc_bind_address"))
            return result->operator[]("grpc_bind_address").as<std::string>();
        return "0.0.0.0";
    }
    /**
         * @brief Gets the REST port
         * 
         * @return uint64_t
         */
    uint64_t restPort() {
        return result->operator[]("rest_port").as<uint64_t>();
    }

    /**
         * @brief Get the rest network interface address to bind to
         * 
         * @return const std::string&
         */
    const std::string restBindAddress() {
        if (result->count("rest_bind_address"))
            return result->operator[]("rest_bind_address").as<std::string>();
        return "0.0.0.0";
    }

    /**
         * @brief Gets the gRPC workers count
         * 
         * @return uint
         */
    uint grpcWorkers() {
        return result->operator[]("grpc_workers").as<uint>();
    }

    /**
         * @brief Gets the rest workers count
         * 
         * @return uint
         */
    uint restWorkers() {
        return result->operator[]("rest_workers").as<uint>();
    }

    /**
         * @brief Get the model name
         * 
         * @return const std::string&
         */
    const std::string& modelName() {
        if (result->count("model_name"))
            return result->operator[]("model_name").as<std::string>();
        return empty;
    }

    /**
         * @brief Gets the model path
         * 
         * @return const std::string&
         */
    const std::string& modelPath() {
        if (result->count("model_path"))
            return result->operator[]("model_path").as<std::string>();
        return empty;
    }

    /**
         * @brief Gets the batch size
         * 
         * @return const std::string&
         */
    const std::string& batchSize() {
        if (!result->count("batch_size")) {
            static const std::string d = "0";
            return d;
        }
        return result->operator[]("batch_size").as<std::string>();
    }

    /**
         * @brief Get the shape
         * 
         * @return const std::vector<size_t>&
         */
    const std::string& shape() {
        if (result->count("shape"))
            return result->operator[]("shape").as<std::string>();
        return empty;
    }

    /**
         * @brief Get the shape
         * 
         * @return const std::string&
         */
    const std::string& modelVersionPolicy() {
        if (result->count("model_version_policy"))
            return result->operator[]("model_version_policy").as<std::string>();
        return empty;
    }

    /**
         * @brief Get the nireq
         *
         * @return uint 
         */
    uint32_t nireq() {
        if (!result->count("nireq")) {
            return 0;
        }
        return result->operator[]("nireq").as<uint32_t>();
    }

    /**
         * @brief Get the target device
         * 
         * @return const std::string& 
         */
    const std::string& targetDevice() {
        return result->operator[]("target_device").as<std::string>();
    }

    /**
         * @brief Get the plugin config
         * 
         * @return const std::string& 
         */
    const std::string& pluginConfig() {
        if (result->count("plugin_config"))
            return result->operator[]("plugin_config").as<std::string>();
        return empty;
    }

    /**
         * @brief Get stateful flag
         *
         * @return bool
         */
    bool stateful() {
        return result->operator[]("stateful").as<bool>();
    }

    /**
     * @brief Get stateful sequence timeout
     *
     * @return uint
     */
    uint32_t sequenceTimeoutSeconds() {
        if (!result->count("sequence_timeout_seconds")) {
            return 0;
        }
        return result->operator[]("sequence_timeout_seconds").as<uint32_t>();
    }

    /**
         * @brief Get low latency transformation flag
         *
         * @return bool
         */
    bool lowLatencyTransformation() {
        return result->operator[]("low_latency_transformation").as<bool>();
    }

    /**
     * @brief Get max number of sequences that can be processed concurrently 
     *
     * @return uint
     */
    uint32_t maxSequenceNumber() {
        if (!result->count("max_sequence_number")) {
            return 0;
        }
        return result->operator[]("max_sequence_number").as<uint32_t>();
    }

    /**
        * @brief Get the log level
        *
        * @return const std::string&
        */
    const std::string& logLevel() {
        if (result->count("log_level"))
            return result->operator[]("log_level").as<std::string>();
        return empty;
    }

    /**
        * @brief Get the log path
         *
        * @return const std::string&
        */
    const std::string& logPath() {
        if (result->count("log_path"))
            return result->operator[]("log_path").as<std::string>();
        return empty;
    }

    /**
        * @brief Get the plugin config
        *
        * @return const std::string&
        */
    const std::string& grpcChannelArguments() {
        if (result->count("grpc_channel_arguments"))
            return result->operator[]("grpc_channel_arguments").as<std::string>();
        return empty;
    }

    /**
     * @brief Get the filesystem pool wait time in seconds
     * 
     * @return uint 
     */
    uint filesystemPollWaitSeconds() {
        return result->operator[]("file_system_poll_wait_seconds").as<uint>();
    }
};
}  // namespace ovms
