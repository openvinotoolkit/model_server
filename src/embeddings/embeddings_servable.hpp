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

#include "../sidepacket_servable.hpp"
#include "embeddings_api.hpp"
#include "../filesystem.hpp"
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace ovms {

struct EmbeddingsServable : SidepacketServable {
public:
    EmbeddingsServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath) :
        SidepacketServable(modelDir, targetDevice, pluginConfig, graphPath) {}
};

using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<EmbeddingsServable>>;
}  // namespace ovms
