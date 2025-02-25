//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include <string>

#include "modelversion.hpp"
#include "tensorinfo.hpp"

namespace ovms {
class Servable {
    const std::string name;
    const model_version_t version = -1;
    // ModelVersionStatus status; // TODO PipelineDefinitionStatus @atobisze
private:
    tensor_map_t inputsInfo;
protected:
    tensor_map_t outputsInfo;
private:
public:
    Servable(const std::string& name, model_version_t version) : name(name), version(version) {};
    virtual ~Servable() = default;

    virtual const std::string& getName() const { // TODO virtual @atobisze
        return name;
    }

    virtual model_version_t getVersion() const { // TODO virtual @atobisze
        return version;
    }

    virtual const tensor_map_t& getInputsInfo() const {
        return inputsInfo;
    }
    virtual const tensor_map_t& getOutputsInfo() const {
        return outputsInfo;
    }
};
}  // namespace ovms
