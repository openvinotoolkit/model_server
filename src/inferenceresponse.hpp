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

#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"
#include "modelversion.hpp"
#include "pocapi.hpp"

namespace ovms {

class Status;

class InferenceResponse {
    const std::string& servableName;
    const model_version_t servableVersion;
    std::vector<InferenceParameter> parameters;                    // TODO after benchmark app verify if additional map<int, name> wouldn't be better
    std::vector<std::pair<std::string, InferenceTensor>> outputs;  // TODO after benchmark app verify if additional map<int, name> wouldn't be better

public:
    InferenceResponse();
    InferenceResponse(const std::string& servableName, model_version_t servableVersion);
    Status addOutput(const std::string& name, OVMS_DataType datatype, const size_t* shape, size_t dimCount);
    Status getOutput(uint32_t id, const std::string** name, InferenceTensor** tensor);  // TODO consider in the future if we need getOutput by name

    Status addParameter(const char* parameterName, OVMS_DataType datatype, const void* data);
    const InferenceParameter* getParameter(uint32_t id) const;

    const std::string& getServableName() const;
    model_version_t getServableVersion() const;
    uint32_t getOutputCount() const;
    uint32_t getParameterCount() const;

    Status setId();
    Status getId();
    InferenceParameter* getInferenceParameter(const char* name);
    void Clear();  // TODO remove
};
}  // namespace ovms
