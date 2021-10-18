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

typedef int (*initialize_ver2_fn)(void**, int*, const struct CustomNodeParam*, int);
typedef int (*deinitialize_ver2_fn)(void*);
typedef int (*execute_ver2_fn)(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*metadata_ver2_fn)(struct CustomNodeTensorInfo**, int*, const struct CustomNodeParam*, int, void*);
typedef int (*release_ver2_fn)(void*, void*);


struct NodeLibraryBase {
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
};

struct NodeLibraryV2 :NodeLibraryBase {
    initialize_ver2_fn initialize = nullptr;
    deinitialize_ver2_fn deinitialize = nullptr;
    execute_ver2_fn execute = nullptr;
    metadata_ver2_fn getInputsInfo = nullptr;
    metadata_ver2_fn getOutputsInfo = nullptr;
    release_ver2_fn release = nullptr;
    bool isValid() const override;
};
class NodeLibraryExecutor {
        public:
int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
private;
    std::unique_ptr<NodeLibraryBase> nodeLibrary;
}

}  // namespace ovms
