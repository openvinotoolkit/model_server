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
    initialize_fn initialize = nullptr;
    deinitialize_fn deinitialize = nullptr;
    execute_fn execute = nullptr;
    metadata_fn getInputsInfo = nullptr;
    metadata_fn getOutputsInfo = nullptr;
    release_fn release = nullptr;

    std::string basePath = "";

    bool isValid() const;
};

struct NodeLibraryBase {
    NodeLibraryBase(const std::string& basePath) :
        basePath(basePath) {}
    std::string basePath = "";
    virtual bool isValid() const;
    virtual ~NodeLibraryBase() = default;
};

class NodeLibraryExecutor {
public:
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
    NodeLibraryExecutor(std::unique_ptr<NodeLibraryBase>&& ptr);
    NodeLibraryExecutor() {}
    //private: // TODO
    std::unique_ptr<NodeLibraryBase> nodeLibrary;
    ~NodeLibraryExecutor() = default;
};

}  // namespace ovms
