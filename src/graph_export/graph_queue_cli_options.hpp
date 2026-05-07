//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <cxxopts.hpp>

#include "../capi_frontend/server_settings.hpp"

namespace ovms {

// Common graph queue CLI options shared across all mediapipe graph task parsers.
// Call addGraphQueueOptions() in createOptions() and extractGraphQueueOptions() in prepare().

inline void addGraphQueueOptions(cxxopts::Options& options, const std::string& group = "graph pool") {
    options.add_options(group)("graph_initial_queue_size",
        "Initial graph pool size at startup. Positive integer or AUTO. Default: 1.",
        cxxopts::value<std::string>(),
        "GRAPH_INITIAL_QUEUE_SIZE")("graph_queue_max_size",
        "Maximum graph pool size (expansion ceiling). Positive integer or AUTO. Default: same as initial (no expansion).",
        cxxopts::value<std::string>(),
        "GRAPH_QUEUE_MAX_SIZE");
}

inline void extractGraphQueueOptions(const cxxopts::ParseResult& result, HFSettingsImpl& hfSettings) {
    if (result.count("graph_initial_queue_size")) {
        hfSettings.exportSettings.graphInitialQueueSize = result["graph_initial_queue_size"].as<std::string>();
    }
    if (result.count("graph_queue_max_size")) {
        hfSettings.exportSettings.graphQueueMaxSize = result["graph_queue_max_size"].as<std::string>();
    }
}

}  // namespace ovms
