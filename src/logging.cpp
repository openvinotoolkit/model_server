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

#if (MEDIAPIPE_DISABLE == 0)
#include <glog/logging.h>
#endif
#include <vector>

namespace ovms {

std::shared_ptr<spdlog::logger> gcs_logger = std::make_shared<spdlog::logger>("gcs");
std::shared_ptr<spdlog::logger> azurestorage_logger = std::make_shared<spdlog::logger>("azurestorage");
std::shared_ptr<spdlog::logger> s3_logger = std::make_shared<spdlog::logger>("s3");
std::shared_ptr<spdlog::logger> modelmanager_logger = std::make_shared<spdlog::logger>("modelmanager");
std::shared_ptr<spdlog::logger> dag_executor_logger = std::make_shared<spdlog::logger>("dag_executor");
std::shared_ptr<spdlog::logger> sequence_manager_logger = std::make_shared<spdlog::logger>("sequence_manager");
std::shared_ptr<spdlog::logger> capi_logger = std::make_shared<spdlog::logger>("C-API");
#if (MEDIAPIPE_DISABLE == 0)
std::shared_ptr<spdlog::logger> mediapipe_logger = std::make_shared<spdlog::logger>("mediapipe");
std::shared_ptr<spdlog::logger> llm_executor_logger = std::make_shared<spdlog::logger>("llm_executor");
std::shared_ptr<spdlog::logger> llm_calculator_logger = std::make_shared<spdlog::logger>("llm_calculator");
std::shared_ptr<spdlog::logger> embeddings_calculator_logger = std::make_shared<spdlog::logger>("embeddings_calculator");
std::shared_ptr<spdlog::logger> rerank_calculator_logger = std::make_shared<spdlog::logger>("rerank_calculator");
#endif
#if (OV_TRACE == 1)
std::shared_ptr<spdlog::logger> ov_logger = std::make_shared<spdlog::logger>("openvino");
#endif
const std::string default_pattern = "[%i] [%Y-%m-%d %T.%f][%t][%n][%l][%s:%#] %v";

static void set_log_level(const std::string log_level, std::shared_ptr<spdlog::logger> logger) {
    logger->set_level(spdlog::level::info);
    if (!log_level.empty()) {
        if (log_level == "DEBUG") {
            logger->set_level(spdlog::level::debug);
            logger->flush_on(spdlog::level::debug);
        } else if (log_level == "ERROR") {
            logger->set_level(spdlog::level::err);
            logger->flush_on(spdlog::level::err);
        } else if (log_level == "WARNING") {
            logger->set_level(spdlog::level::warn);
            logger->flush_on(spdlog::level::warn);
        } else if (log_level == "TRACE") {
            logger->set_level(spdlog::level::trace);
            logger->flush_on(spdlog::level::trace);
        }
    }
}

static void register_loggers(const std::string& log_level, std::vector<spdlog::sink_ptr> sinks) {
    auto serving_logger = std::make_shared<spdlog::logger>("serving", begin(sinks), end(sinks));
    serving_logger->set_pattern(default_pattern);
    gcs_logger->set_pattern(default_pattern);
    azurestorage_logger->set_pattern(default_pattern);
    s3_logger->set_pattern(default_pattern);
    modelmanager_logger->set_pattern(default_pattern);
    dag_executor_logger->set_pattern(default_pattern);
    sequence_manager_logger->set_pattern(default_pattern);
    capi_logger->set_pattern(default_pattern);
#if (MEDIAPIPE_DISABLE == 0)
    mediapipe_logger->set_pattern(default_pattern);
    llm_executor_logger->set_pattern(default_pattern);
    llm_calculator_logger->set_pattern(default_pattern);
    rerank_calculator_logger->set_pattern(default_pattern);
    embeddings_calculator_logger->set_pattern(default_pattern);
#endif
#if (OV_TRACE == 1)
    ov_logger->set_pattern(default_pattern);
#endif
    for (auto& sink : sinks) {
        gcs_logger->sinks().push_back(sink);
        azurestorage_logger->sinks().push_back(sink);
        s3_logger->sinks().push_back(sink);
        modelmanager_logger->sinks().push_back(sink);
        dag_executor_logger->sinks().push_back(sink);
        sequence_manager_logger->sinks().push_back(sink);
        capi_logger->sinks().push_back(sink);
#if (MEDIAPIPE_DISABLE == 0)
        mediapipe_logger->sinks().push_back(sink);
        llm_executor_logger->sinks().push_back(sink);
        llm_calculator_logger->sinks().push_back(sink);
        rerank_calculator_logger->sinks().push_back(sink);
        embeddings_calculator_logger->sinks().push_back(sink);
#endif
#if (OV_TRACE == 1)
        ov_logger->sinks().push_back(sink);
#endif
    }
    set_log_level(log_level, serving_logger);
    set_log_level(log_level, gcs_logger);
    set_log_level(log_level, azurestorage_logger);
    set_log_level(log_level, s3_logger);
    set_log_level(log_level, modelmanager_logger);
    set_log_level(log_level, dag_executor_logger);
    set_log_level(log_level, sequence_manager_logger);
    set_log_level(log_level, capi_logger);
#if (MEDIAPIPE_DISABLE == 0)
    set_log_level(log_level, mediapipe_logger);
    set_log_level(log_level, llm_executor_logger);
    set_log_level(log_level, llm_calculator_logger);
    set_log_level(log_level, rerank_calculator_logger);
    set_log_level(log_level, embeddings_calculator_logger);
#endif
#if (OV_TRACE == 1)
    set_log_level(log_level, ov_logger);
#endif
    spdlog::set_default_logger(serving_logger);
}

void configure_logger(const std::string& log_level, const std::string& log_path) {
    static bool wasRun = false;
    if (wasRun) {
        SPDLOG_WARN("Tried to configure loggers twice. Keeping previous settings.");
        return;
    }
    wasRun = true;
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    if (!log_path.empty()) {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path));
    }
    register_loggers(log_level, sinks);
#if (MEDIAPIPE_DISABLE == 0)
#ifdef __linux__
    if (log_level == "DEBUG" || log_level == "TRACE")
        FLAGS_minloglevel = google::INFO;
    else if (log_level == "WARNING")
        FLAGS_minloglevel = google::WARNING;
    else  // ERROR, FATAL
        FLAGS_minloglevel = google::ERROR;
#elif _WIN32
    if (log_level == "DEBUG" || log_level == "TRACE")
        FLAGS_minloglevel = google::GLOG_INFO;
    else if (log_level == "WARNING")
        FLAGS_minloglevel = google::GLOG_WARNING;
    else  // ERROR, FATAL
        FLAGS_minloglevel = google::GLOG_ERROR;
#endif
#endif
}

}  // namespace ovms
