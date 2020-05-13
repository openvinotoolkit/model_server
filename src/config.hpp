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
        Config(const Config&);

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

        /**
         * @brief 
         */
        const std::vector<size_t> emptyShape;

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
         * @return true 
         * @return false 
         */
        bool validate();

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
         * @return uint16_t
         */
        uint16_t port() {
            return result->operator[]("port").as<uint16_t>();
        }

        /**
         * @brief Gets the REST port
         * 
         * @return uint16_t
         */
        uint16_t restPort() {
            return result->operator[]("rest_port").as<uint16_t>();
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
         * @return size_t
         */
        size_t batchSize() {
            return result->operator[]("batch_size").as<size_t>();
        }

        /**
         * @brief Get the shape
         * 
         * @return const std::vector<size_t>&
         */
        const std::vector<size_t> shape() {
            if (result->count("shape"))
                return result->operator[]("shape").as<std::vector<size_t>>();
            return emptyShape;
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
        uint nireq() {
            return result->operator[]("nireq").as<uint>();
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
        * @return const std::string&f
        */
        const std::string& grpcChannelArguments() {
            if (result->count("grpc_channel_arguments"))
                return result->operator[]("grpc_channel_arguments").as<std::string>();
            return empty;
        }
    };
}  // namespace ovms
