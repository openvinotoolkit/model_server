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
#include <unordered_map>

#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"
#include "modelversion.hpp"
#include "pocapi.hpp"

namespace ovms {

class Status;

class InferenceResponse {
    const std::string& servableName;
    const model_version_t servableVersion;
    std::unordered_map<std::string, InferenceParameter> parameters;
    std::unordered_map<std::string, InferenceTensor> outputs;

public:
    InferenceResponse(const std::string& servableName, model_version_t servableVersion);
    Status addOutput(const std::string& name, DataType datatype, const size_t* shape, size_t dimCount);
    Status getOutput(const char* name, InferenceTensor** tensor);
    Status addParameter(const char* parameterName, DataType datatype, const void* data);
    const InferenceParameter* getParameter(const char* name) const;

    const std::string& getServableName() const;
    model_version_t getServableVersion() const;

    Status setId();
    Status getId();
    InferenceParameter* getInferenceParameter(const char* name);
};
}  // namespace ovms
