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
#include "logging.hpp"

#include <vector>

namespace ovms {

std::shared_ptr<spdlog::logger> gcs_logger = std::make_shared<spdlog::logger>("gcs");
std::shared_ptr<spdlog::logger> azurestorage_logger = std::make_shared<spdlog::logger>("azurestorage");
std::shared_ptr<spdlog::logger> s3_logger = std::make_shared<spdlog::logger>("s3");
std::shared_ptr<spdlog::logger> modelmanager_logger = std::make_shared<spdlog::logger>("modelmanager");
std::shared_ptr<spdlog::logger> dag_executor_logger = std::make_shared<spdlog::logger>("dag_executor");

const std::string default_pattern = "[%Y-%m-%d %T.%e][%t][%n][%l][%s:%#] %v";

void set_log_level(const std::string log_level, std::shared_ptr<spdlog::logger> logger) {
    logger->set_level(spdlog::level::info);
    if (!log_level.empty()) {
        if (log_level == "DEBUG") {
            logger->set_level(spdlog::level::debug);
            logger->flush_on(spdlog::level::trace);
        } else if (log_level == "ERROR") {
            logger->set_level(spdlog::level::err);
            logger->flush_on(spdlog::level::err);
        }
    }
}

void register_loggers(const std::string log_level, std::vector<spdlog::sink_ptr> sinks) {
    auto serving_logger = std::make_shared<spdlog::logger>("serving", begin(sinks), end(sinks));
    serving_logger->set_pattern(default_pattern);
    gcs_logger->set_pattern(default_pattern);
    azurestorage_logger->set_pattern(default_pattern);
    s3_logger->set_pattern(default_pattern);
    modelmanager_logger->set_pattern(default_pattern);
    dag_executor_logger->set_pattern(default_pattern);
    for (auto sink : sinks) {
        gcs_logger->sinks().push_back(sink);
        azurestorage_logger->sinks().push_back(sink);
        s3_logger->sinks().push_back(sink);
        modelmanager_logger->sinks().push_back(sink);
        dag_executor_logger->sinks().push_back(sink);
    }
    set_log_level(log_level, serving_logger);
    set_log_level(log_level, gcs_logger);
    set_log_level(log_level, azurestorage_logger);
    set_log_level(log_level, s3_logger);
    set_log_level(log_level, modelmanager_logger);
    set_log_level(log_level, dag_executor_logger);
    spdlog::set_default_logger(serving_logger);
}

void configure_logger(const std::string log_level, const std::string log_path) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    if (!log_path.empty()) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
    }
    register_loggers(log_level, sinks);
}

}  // namespace ovms
