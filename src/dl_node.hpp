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

#include <optional>
#include <string>

#include "model_version_policy.hpp"  // for model_version_t typename
#include "modelinstance.hpp"
#include "node.hpp"

namespace ovms {

class DLNode : public Node {
    std::string model_name;
    std::optional<model_version_t> model_version;

public:
    DLNode(const std::string& node_name, const std::string& model_name, std::optional<model_version_t> model_version) :
        Node(node_name),
        model_name(model_name),
        model_version(model_version) {
    }

    Status execute() override;

    Status fetchResults(BlobMap& outputs) override;
};

}  // namespace ovms
