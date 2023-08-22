#pragma once
//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <string>
#include <utility>
#include <vector>

#include "../modelversion.hpp"
#include "../ovms.h"  // NOLINT
#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"

namespace ovms {

class Status;

class InferenceResponse {
    const std::string& servableName;
    const model_version_t servableVersion;
    std::vector<InferenceParameter> parameters;
    std::vector<std::pair<std::string, InferenceTensor>> outputs;

public:
    // this constructor can be removed with prediction tests overhaul
    InferenceResponse();
    InferenceResponse(const std::string& servableName, model_version_t servableVersion);
    Status addOutput(const std::string& name, OVMS_DataType datatype, const int64_t* shape, size_t dimCount);
    Status getOutput(uint32_t id, const std::string** name, const InferenceTensor** tensor) const;
    Status getOutput(uint32_t id, const std::string** name, InferenceTensor** tensor);

    Status addParameter(const char* parameterName, OVMS_DataType datatype, const void* data);
    const InferenceParameter* getParameter(uint32_t id) const;

    const std::string& getServableName() const;
    model_version_t getServableVersion() const;
    uint32_t getOutputCount() const;
    uint32_t getParameterCount() const;

    Status setId();
    Status getId();
    InferenceParameter* getInferenceParameter(const char* name);

    void Clear();
};
}  // namespace ovms
