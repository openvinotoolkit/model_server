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

#include <openvino/openvino.hpp>

#include "modelversion.hpp"
#include "profiler.hpp"
#include "tensorinfo.hpp"

namespace ovms {
class Status;
typedef const std::string& (*outputNameChooser_t)(const std::string&, const TensorInfo&);

template <typename T>
class OutputGetter {
public:
    OutputGetter(T t) :
        outputSource(t) {}
    Status get(const std::string& name, ov::Tensor& tensor);

private:
    T outputSource;
};

template <typename T, typename RequestType, typename ResponseType>
Status serializePredictResponse(
    OutputGetter<T>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const RequestType* request,
    ResponseType* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent = true);  // does not apply for TFS frontend

template <typename ProtoStorage, typename ProtoType>
class ProtoGetter {
    ProtoStorage protoStorage;

public:
    ProtoGetter(ProtoStorage protoStorage) :
        protoStorage(protoStorage) {}
    ProtoType createOutput(const std::string& name);
    std::string* createContent(const std::string& name);
};

const std::string& getTensorInfoName(const std::string& first, const TensorInfo& tensorInfo);
const std::string& getOutputMapKeyName(const std::string& first, const TensorInfo& tensorInfo);

void serializeContent(std::string* content, ov::Tensor& tensor);
void serializeStringContent(std::string* content, ov::Tensor& tensor);
void serializeStringContentFrom2DU8(std::string* content, ov::Tensor& tensor);

// used only for KFS
template <typename RequestType>
bool useSharedOutputContentFn(const RequestType* request) {
    return false;
}
}  // namespace ovms
