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
#include "node.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace ovms {

void Node::setInputs(const Node& dependency, BlobMap& inputs) {
    auto& map = this->input_blobs[dependency.getName()];

    // This node had no dependency
    if (this->required_blob_names.count(dependency.getName()) == 0) {
        // Possibly some kind of error?
        return;
    }

    const auto& names = this->required_blob_names[dependency.getName()];

    // Set only inputs that are required by this node
    for (const auto& kv : inputs) {
        if (std::find(names.cbegin(), names.cend(), kv.first) != names.cend()) {
            SPDLOG_INFO("Node::setInputs: setting required input for {} from {}, input name: {}", getName(), dependency.getName(), kv.first);
            map[kv.first] = kv.second;
        }
    }
}

}  // namespace ovms
