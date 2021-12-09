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
#include "node_library.hpp"

namespace ovms {

bool NodeLibrary::isValid() const {
    return execute != nullptr &&
           getInputsInfo != nullptr &&
           getOutputsInfo != nullptr &&
           release != nullptr &&
           initialize != nullptr &&
           deinitialize != nullptr;
}

NodeLibraryExecutor::NodeLibraryExecutor(std::unique_ptr<NodeLibrary>&& ptr) :
    nodeLibrary(std::move(ptr)) {
    //    auto libPtr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
}

int NodeLibraryExecutor::initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->initialize(customNodeLibraryInternalManager, params, paramsCount);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::deinitialize(void* customNodeLibraryInternalManager) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->deinitialize(customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->execute(inputs, inputsCount, outputs, outputsCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->getInputsInfo(info, infoCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->getOutputsInfo(info, infoCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::release(void* ptr, void* customNodeLibraryInternalManager) {
    auto libPtr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (libPtr != nullptr) {
        return libPtr->release(ptr, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}

}  // namespace ovms
