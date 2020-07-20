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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dl_node.hpp"
#include "entry_node.hpp"
#include "exit_node.hpp"
#include "pipelinemessage.hpp"
#include "status.hpp"
#include "threadsafequeue.hpp"

namespace ovms {

class Pipeline {
    std::vector<std::unique_ptr<Node>> nodes;

    EntryNode& entry;
    ExitNode& exit;

public:
    Pipeline(EntryNode& entry, ExitNode& exit) :
        entry(entry),
        exit(exit) {}

    void push(std::unique_ptr<Node> node) {
        nodes.emplace_back(std::move(node));
    }

    EntryNode& getEntry() const { return this->entry; }
    ExitNode& getExit() const { return this->exit; }

    static void connect(Node& from, Node& to, const InputPairs& blob_names_mapping) {
        from.addDependant(to);
        to.addDependency(from, blob_names_mapping);
    }
    Status execute();

private:
    ThreadSafeQueue<PipelineMessage> messageQueue;
};

}  // namespace ovms
