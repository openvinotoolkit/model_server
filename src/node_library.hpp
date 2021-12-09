//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <inference_engine.hpp>

#include "custom_node_interface.h"  // NOLINT

namespace ovms {

typedef int (*initialize_fn)(void**, const struct CustomNodeParam*, int);
typedef int (*deinitialize_fn)(void*);
typedef int (*execute_fn)(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*metadata_fn)(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*release_fn)(void*, void*);

struct NodeLibrary {
    std::string basePath = "";
    initialize_fn initialize = nullptr;
    deinitialize_fn deinitialize = nullptr;
    execute_fn execute = nullptr;
    metadata_fn getInputsInfo = nullptr;
    metadata_fn getOutputsInfo = nullptr;
    release_fn release = nullptr;

    NodeLibrary(const std::string& basePath,
        initialize_fn initialize,
        deinitialize_fn deinitialize,
        execute_fn execute,
        metadata_fn getInputsInfo,
        metadata_fn getOutputsInfo,
        release_fn release) :
        basePath(basePath),
        initialize(initialize),
        deinitialize(deinitialize),
        execute(execute),
        getInputsInfo(getInputsInfo),
        getOutputsInfo(getOutputsInfo),
        release(release) {}

    bool isValid() const;
    virtual ~NodeLibrary() = default;
};

class NodeLibraryExecutor {
public:
    virtual int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
    virtual int deinitialize(void* customNodeLibraryInternalManager);
    virtual int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    virtual int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    virtual int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
    virtual int release(void* ptr, void* customNodeLibraryInternalManager);
    
    bool isValid() const {
        return nodeLibrary->isValid();
    }
    std::string getBasePath() const {
        return nodeLibrary->basePath;
    }
    NodeLibraryExecutor(NodeLibraryExecutor&& rhs) :
        nodeLibrary(std::move(rhs.nodeLibrary)) {}
    NodeLibraryExecutor(const ovms::NodeLibraryExecutor&) = delete;
    NodeLibraryExecutor& operator=(NodeLibraryExecutor&& rhs) {
        if (this == &rhs) {
            return *this;
        }
        this->nodeLibrary = std::move(rhs.nodeLibrary);
    }
    NodeLibraryExecutor(std::unique_ptr<NodeLibrary>&& ptr);
    NodeLibraryExecutor() {}
    //private: // TODO
    std::unique_ptr<NodeLibrary> nodeLibrary;
    ~NodeLibraryExecutor() = default;
};

}  // namespace ovms
