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
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

#include "../modelversion.hpp"
#include "../ovms.h"  // NOLINT
#include "inferenceparameter.hpp"
#include "inferencetensor.hpp"

namespace ovms {
class Buffer;
class Status;

class InferenceRequest {
    const std::string servableName;
    const model_version_t servableVersion;
    std::unordered_map<std::string, InferenceParameter> parameters;
    std::unordered_map<std::string, InferenceTensor> inputs;
    std::unordered_map<std::string, InferenceTensor> outputs;
    OVMS_InferenceRequestCompletionCallback_t responseCompleteCallback{nullptr};
    void* responseCompleteCallbackData = nullptr;

public:
    // this constructor can be removed with prediction tests overhaul
    InferenceRequest();
    InferenceRequest(const char* modelName, model_version_t modelVersion);
    Status addInput(const char* name, OVMS_DataType datatype, const int64_t* shape, size_t dimCount);
    Status addOutput(const char* name, OVMS_DataType datatype, const int64_t* shape, size_t dimCount);
    Status getInput(const char* name, const InferenceTensor** tensor) const;
    Status getOutput(const char* name, const InferenceTensor** tensor) const;
    uint64_t getInputsSize() const;
    uint64_t getOutputsSize() const;
    Status removeInput(const char* name);
    Status removeOutput(const char* name);
    Status removeAllInputs();

    Status setInputBuffer(const char* name, const void* addr, size_t byteSize, OVMS_BufferType, std::optional<uint32_t> deviceId);
    Status setOutputBuffer(const char* name, const void* addr, size_t byteSize, OVMS_BufferType, std::optional<uint32_t> deviceId);
    Status removeInputBuffer(const char* name);
    Status removeOutputBuffer(const char* name);

    Status addParameter(const char* parameterName, OVMS_DataType datatype, const void* data);
    Status removeParameter(const char* parameterName);
    const InferenceParameter* getParameter(const char* name) const;

    void setCompletionCallback(OVMS_InferenceRequestCompletionCallback_t callback, void* callbackData);
    OVMS_InferenceRequestCompletionCallback_t getResponseCompleteCallback() const {
        return this->responseCompleteCallback;
    }
    void* getResponseCompleteCallbackData() const {
        return this->responseCompleteCallbackData;
    }

    const std::string& getServableName() const;
    model_version_t getServableVersion() const;

    Status setId();
    Status getId();

    Status setPriority();
    Status getPriority();

    Status setTimeoutMicorseconds(uint64_t microseconds);
    InferenceParameter* getInferenceParameter(const char* name);
    InferenceTensor* getTensor(const char* name);

    Status getBatchSize(size_t& batchSize, size_t batchSizeIndex) const;
    std::map<std::string, shape_t> getRequestShapes() const;
};
}  // namespace ovms
