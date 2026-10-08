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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "python_node_config.hpp"

namespace ovms {
class PythonBackend;
class PythonNodeResources;

inline constexpr uint32_t PYTHON_CALCULATORS_PLUGIN_ABI_VERSION = 2;

enum class PythonPluginResult : int {
    OK = 0,
    PYTHON_EXCEPTION = 1,
    OTHER_FAILURE = 2,
};

// Owned Python object reference; the receiver releases it with releaseObject.
struct PythonNodeOutput {
    std::string tag;
    void* object;
};

// Entry points exported by libpython_calculators, the only module linking libpython. The host owns MediaPipe:
// calculator registration, packets and protos. Python objects cross this boundary only as opaque handles.
struct PythonCalculatorsPluginApi {
    uint32_t abiVersion;

    // Returns ovms::StatusCode.
    int (*createNodeResources)(const PythonNodeConfig& config, PythonBackend* pythonBackend, std::shared_ptr<PythonNodeResources>& nodeResources);
    bool (*hasPythonBackend)(const PythonNodeResources& nodeResources);

    void (*releaseObject)(void* object);
    void (*releaseGenerator)(void* generator);

    // Calls OvmsPythonModel.execute with borrowed input handles. Sets generator instead of outputs when execute yielded.
    PythonPluginResult (*execute)(PythonNodeResources& nodeResources, const std::vector<const void*>& inputs,
        std::vector<PythonNodeOutput>& outputs, void*& generator, std::string& message);
    // produced is false when the generator is exhausted. Outputs may be produced even if advancing the generator failed.
    PythonPluginResult (*generatorNext)(PythonNodeResources& nodeResources, void* generator,
        std::vector<PythonNodeOutput>& outputs, bool& produced, std::string& message);

    // Creates pyovms.Tensor owning a copy of data.
    PythonPluginResult (*createTensor)(const std::string& name, const void* data, const std::vector<int64_t>& shape,
        const std::string& datatype, size_t size, void*& tensor, std::string& message);
    // Returned data stays valid as long as the tensor handle.
    PythonPluginResult (*getTensorInfo)(const void* tensor, std::string& datatype, std::vector<int64_t>& shape,
        const void*& data, size_t& size, std::string& message);
};

}  // namespace ovms
