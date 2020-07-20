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
#include <utility>
#include <vector>

#include <inference_engine.hpp>

#include "status.hpp"

namespace ovms {

using BlobMap = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;
using BlobNames = std::vector<std::string>;
using InputPairs = std::vector<std::pair<std::string, std::string>>;

class Node {
protected:
    std::string node_name;

    std::vector<std::reference_wrapper<Node>> previous;
    std::vector<std::reference_wrapper<Node>> next;

    size_t finished_dependencies_count = 0;

    // Blobs ready and waiting for execution
    std::unordered_map<std::string, InferenceEngine::Blob::Ptr> input_blobs;

    // Input/Output name mapping and list of required inputs from previous nodes
    std::unordered_map<std::string, InputPairs> blob_names_mapping;

public:
    Node(const std::string& node_name) :
        node_name(node_name) {
    }

    const std::string& getName() const { return this->node_name; }

    virtual Status execute() = 0;
    virtual Status fetchResults(BlobMap& outputs) = 0;

    Status setInputs(const Node& dependency, BlobMap& inputs);

    virtual void addDependency(Node& node, const InputPairs& blob_names_mapping) {
        this->previous.emplace_back(node);
        this->blob_names_mapping[node.getName()] = blob_names_mapping;
    }

    virtual void addDependant(Node& node) { this->next.emplace_back(node); }

    void increaseFinishedDependencyCount() {
        finished_dependencies_count++;
    }

    bool isReady() const {
        return finished_dependencies_count == previous.size();
    }
};

}  // namespace ovms
