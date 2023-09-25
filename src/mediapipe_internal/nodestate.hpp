//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <filesystem>
#include <map>
#include <string>

#include <pybind11/embed.h>  // everything needed for embedding

#include "src/mediapipe_calculators/python_backend_calculator.pb.h"

namespace py = pybind11;

namespace ovms {
class Status;

class NodeState {
    py::object pythonNodeState;

public:
    NodeState();
    NodeState(const NodeState& other);
    Status Create(const std::string handler_path);

    ~NodeState();
};

}  // namespace ovms
