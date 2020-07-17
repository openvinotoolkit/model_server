//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <unordered_map>
#include <vector>

#include <inference_engine.hpp>

#include "status.hpp"

namespace ovms {
using BlobMap = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;
using BlobNames = std::vector<std::string>;

class Node {
protected:
    std::string node_name;

    std::vector<Node*> previous;
    std::vector<Node*> next;

    size_t finished_dependencies_count = 0;

    // Blobs waiting for inference to happen
    std::unordered_map<std::string, BlobMap> input_blobs;

    // List of required previous node outputs
    std::unordered_map<std::string, BlobNames> required_blob_names;

public:
    Node(const std::string node_name) :
        node_name(node_name) {
    }

    const std::string& getName() const { return this->node_name; }

    virtual Status execute() = 0;
    virtual Status fetchResults(BlobMap& map) = 0;

    void setInputs(const Node& dependency, BlobMap& inputs);

    virtual void addDependency(Node& node, const BlobNames& required_blob_names) {
        this->previous.emplace_back(&node);
        this->required_blob_names[node.getName()] = required_blob_names;
    }

    virtual void addDependant(Node& node) { this->next.emplace_back(&node); }

    void increaseFinishedDependencyCount() {
        finished_dependencies_count++;
    }

    bool isReady() const {
        return finished_dependencies_count == previous.size();
    }
};

}  // namespace ovms
