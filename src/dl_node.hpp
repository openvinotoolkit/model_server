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

#include "modelinstance.hpp"
#include "node.hpp"

namespace ovms {

class DLNode : public Node {
    ModelInstance* model;

public:
    // Instead of passing instance, consider pasing name & version.
    // This would not block model instances during pipeline executing up till instance is really needed.
    DLNode(ModelInstance* model) :
        model(model) {
    }

    Status execute() override {
        // Get required multiple Blob::Ptr from previous nodes
        // Do inference
        // Retrieve multiple output Blob::Ptr from InferRequest result
        return StatusCode::OK;
    }
};

}  // namespace ovms
