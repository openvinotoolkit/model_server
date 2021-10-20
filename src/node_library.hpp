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

// TODO require more typedefs :/

namespace ovms {

typedef int (*execute_fn)(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int);
typedef int (*metadata_fn)(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int);
typedef int (*release_fn)(void*);

typedef int (*initialize_ver2_fn)(void**, const struct CustomNodeParam*, int);
typedef int (*deinitialize_ver2_fn)(void*);
typedef int (*execute_ver2_fn)(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*metadata_ver2_fn)(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*release_ver2_fn)(void*, void*);

struct NodeLibraryBase {
    NodeLibraryBase(const std::string& basePath) :
        basePath(basePath) {}
    std::string basePath = "";
    virtual bool isValid() const = 0;
    virtual ~NodeLibraryBase() = default;
};

struct NodeLibrary : NodeLibraryBase {
    execute_fn execute = nullptr;
    metadata_fn getInputsInfo = nullptr;
    metadata_fn getOutputsInfo = nullptr;
    release_fn release = nullptr;
    bool isValid() const override;
    NodeLibrary(const std::string& basePath,
        execute_fn execute,
        metadata_fn getInputsInfo,
        metadata_fn getOutputsInfo,
        release_fn release) :
        NodeLibraryBase(basePath),
        execute(execute),
        getInputsInfo(getInputsInfo),
        getOutputsInfo(getOutputsInfo),
        release(release) {}
};

struct NodeLibraryV2 : NodeLibraryBase {
    initialize_ver2_fn initialize = nullptr;
    deinitialize_ver2_fn deinitialize = nullptr;
    execute_ver2_fn execute = nullptr;
    metadata_ver2_fn getInputsInfo = nullptr;
    metadata_ver2_fn getOutputsInfo = nullptr;
    release_ver2_fn release = nullptr;
    bool isValid() const override;
    NodeLibraryV2(const std::string& basePath,
        initialize_ver2_fn initialize,
        deinitialize_ver2_fn deinitialize,
        execute_ver2_fn execute,
        metadata_ver2_fn getInputsInfo,
        metadata_ver2_fn getOutputsInfo,
        release_ver2_fn release) :
        NodeLibraryBase(basePath),
        initialize(initialize),
        deinitialize(deinitialize),
        execute(execute),
        getInputsInfo(getInputsInfo),
        getOutputsInfo(getOutputsInfo),
        release(release) {}
};
class NodeLibraryExecutor {
public:
    initialize_ver2_fn initialize;
    deinitialize_ver2_fn deinitialize;
    execute_ver2_fn execute;
    metadata_ver2_fn getInputsInfo;
    metadata_ver2_fn getOutputsInfo;
    release_ver2_fn release;

    bool isValid() const {
        return nodeLibrary->isValid();
    }
    std::string getBasePath() const {
        return nodeLibrary->basePath;
    }
    NodeLibraryExecutor(NodeLibraryExecutor&& rhs) :
        nodeLibrary(std::move(rhs.nodeLibrary)) {}
    NodeLibraryExecutor(std::unique_ptr<NodeLibraryBase>&& ptr) :
        nodeLibrary(std::move(ptr)) {}

private:
    std::unique_ptr<NodeLibraryBase> nodeLibrary;
};

}  // namespace ovms
