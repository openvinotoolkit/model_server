//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <future>
#include <optional>
#include <string>

#include "src/modelconfig.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "src/mediapipe_internal/mediapipegraphconfig.hpp"
#endif

namespace ovms {

enum class ServableLoadingTaskType {
    LoadModel,
    RetireModel,
    LoadMediapipe,
    UnloadMediapipe
};

struct ServableLoadingTask {
    ServableLoadingTaskType type;
    std::string name;
    std::optional<ModelConfig> modelConfig;
#if (MEDIAPIPE_DISABLE == 0)
    std::optional<MediapipeGraphConfig> graphConfig;
#endif
    std::promise<Status> completion;

    ServableLoadingTask(ServableLoadingTaskType type, const std::string& name, const ModelConfig& config) :
        type(type),
        name(name),
        modelConfig(config) {}

#if (MEDIAPIPE_DISABLE == 0)
    ServableLoadingTask(ServableLoadingTaskType type, const std::string& name, const MediapipeGraphConfig& config) :
        type(type),
        name(name),
        graphConfig(config) {}
#endif

    ServableLoadingTask(ServableLoadingTaskType type, const std::string& name) :
        type(type),
        name(name) {}

    ServableLoadingTask(ServableLoadingTask&&) = default;
    ServableLoadingTask& operator=(ServableLoadingTask&&) = default;
    ServableLoadingTask(const ServableLoadingTask&) = delete;
    ServableLoadingTask& operator=(const ServableLoadingTask&) = delete;
};

}  // namespace ovms
