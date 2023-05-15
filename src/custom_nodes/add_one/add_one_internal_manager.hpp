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

#include "../common/custom_node_library_internal_manager.hpp"

namespace ovms {
namespace custom_nodes_common {

class AddOneInternalManager : public CustomNodeLibraryInternalManager {
    int outputSize;
    int inputSize;
    int currentOutputQueueSize;
    int currentInfoQueueSize;

public:
    AddOneInternalManager(int outputSize = 10, int inputSize = 10,
        int currentOutputQueueSize = 10, int currentInfoQueueSize = 10) :
        outputSize(outputSize),
        inputSize(inputSize),
        currentOutputQueueSize(currentOutputQueueSize),
        currentInfoQueueSize(currentInfoQueueSize) {
    }
    ~AddOneInternalManager() {}
    int getOutputSize() {
        return outputSize;
    }
    int getInputSize() {
        return inputSize;
    }
    void setOutputSize(int outputSize) {
        this->outputSize = outputSize;
    }
    void setInputSize(int inputSize) {
        this->inputSize = inputSize;
    }
    int getCurrentOutputQueueSize() {
        return currentOutputQueueSize;
    }
    int getCurrentInfoQueueSize() {
        return currentInfoQueueSize;
    }
    void setCurrentOutputQueueSize(int currentOutputQueueSize) {
        this->currentOutputQueueSize = currentOutputQueueSize;
    }
    void setCurrentInfoQueueSize(int currentInfoQueueSize) {
        this->currentInfoQueueSize = currentInfoQueueSize;
    }
};
}  // namespace custom_nodes_common
}  // namespace ovms
