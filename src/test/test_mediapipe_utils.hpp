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

#if (MEDIAPIPE_DISABLE == 0)

#include <string>

#include "src/mediapipe_internal/mediapipegraphdefinition.hpp"
#include "src/mediapipe_internal/mediapipegraphexecutor.hpp"
#include "src/status.hpp"

#if (PYTHON_DISABLE == 0)
#include "src/python/pythonnoderesources.hpp"
#endif

class DummyMediapipeGraphDefinition : public ovms::MediapipeGraphDefinition {
public:
    std::string inputConfig;
#if (PYTHON_DISABLE == 0)
    ovms::PythonNodeResources* getPythonNodeResources(const std::string& nodeName) {
        auto it = this->sidePacketMaps.pythonNodeResourcesMap.find(nodeName);
        if (it == std::end(this->sidePacketMaps.pythonNodeResourcesMap)) {
            return nullptr;
        } else {
            return it->second.get();
        }
    }
#endif

    ovms::GenAiServable* getGenAiServable(const std::string& nodeName) {
        auto it = this->sidePacketMaps.genAiServableMap.find(nodeName);
        if (it == std::end(this->sidePacketMaps.genAiServableMap)) {
            return nullptr;
        } else {
            return it->second.get();
        }
    }

    ovms::Status validateForConfigLoadablenessPublic() {
        return this->validateForConfigLoadableness();
    }

    ovms::GenAiServableMap& getGenAiServableMap() { return this->sidePacketMaps.genAiServableMap; }

    DummyMediapipeGraphDefinition(const std::string name,
        const ovms::MediapipeGraphConfig& config,
        std::string inputConfig,
        ovms::PythonBackend* pythonBackend = nullptr) :
        ovms::MediapipeGraphDefinition(name, config, nullptr, nullptr, pythonBackend) { this->inputConfig = inputConfig; }

    // Do not read from path - use predefined config contents
    ovms::Status validateForConfigFileExistence() override {
        this->chosenConfig = this->inputConfig;
        return ovms::StatusCode::OK;
    }
};

#endif  // MEDIAPIPE_DISABLE == 0
