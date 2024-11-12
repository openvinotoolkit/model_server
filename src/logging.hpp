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

#include <fmt/ranges.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace ovms {

extern std::shared_ptr<spdlog::logger> gcs_logger;
extern std::shared_ptr<spdlog::logger> azurestorage_logger;
extern std::shared_ptr<spdlog::logger> s3_logger;
extern std::shared_ptr<spdlog::logger> modelmanager_logger;
extern std::shared_ptr<spdlog::logger> dag_executor_logger;
extern std::shared_ptr<spdlog::logger> sequence_manager_logger;
extern std::shared_ptr<spdlog::logger> capi_logger;
#if (MEDIAPIPE_DISABLE == 0)
extern std::shared_ptr<spdlog::logger> mediapipe_logger;
extern std::shared_ptr<spdlog::logger> llm_executor_logger;
extern std::shared_ptr<spdlog::logger> llm_calculator_logger;
extern std::shared_ptr<spdlog::logger> embeddings_calculator_logger;
extern std::shared_ptr<spdlog::logger> rerank_calculator_logger;
#endif
#if (OV_TRACE == 1)
extern std::shared_ptr<spdlog::logger> ov_logger;
#define OV_LOGGER(...) SPDLOG_LOGGER_TRACE(ov_logger, __VA_ARGS__)
#else
#define OV_LOGGER(...)
#endif

void configure_logger(const std::string& log_level, const std::string& log_path);

}  // namespace ovms
