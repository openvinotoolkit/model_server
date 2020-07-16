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

#include <vector>

#include "status.hpp"

namespace ovms {

class Node {
protected:
    std::vector<Node*> previous;
    std::vector<Node*> next;

    size_t finished_dependencies_count = 0;

public:
    virtual Status execute() = 0;

    virtual void addDependency(Node& node) { this->previous.emplace_back(&node); }
    virtual void addDependant(Node& node) { this->next.emplace_back(&node); }

    void increaseFinishedDependencyCount() {
        finished_dependencies_count++;
    }

    bool isReady() const {
        return finished_dependencies_count == previous.size();
    }
};

}  // namespace ovms
