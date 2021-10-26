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

#include <functional>

#include <spdlog/spdlog.h>
namespace ovms {

NodeLibraryExecutor::NodeLibraryExecutor(std::unique_ptr<NodeLibraryBase>&& ptr) :
    nodeLibrary(std::move(ptr)) {
    //    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
}
int NodeLibraryExecutor::initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->initialize(customNodeLibraryInternalManager, params, paramsCount);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::deinitialize(void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->deinitialize(customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->execute(inputs, inputsCount, outputs, outputsCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->getInputsInfo(info, infoCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->getOutputsInfo(info, infoCount, params, paramsCount, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor::release(void* ptr, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->release(ptr, customNodeLibraryInternalManager);
    } else {
        return 999;
    }
}

int NodeLibraryExecutor1::initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int NodeLibraryExecutor1::deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int NodeLibraryExecutor1::execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->execute(inputs, inputsCount, outputs, outputsCount, params, paramsCount);
    } else {
        return 999;
    }
}

int NodeLibraryExecutor1::getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->getInputsInfo(info, infoCount, params, paramsCount);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor1::getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->getOutputsInfo(info, infoCount, params, paramsCount);
    } else {
        return 999;
    }
}
int NodeLibraryExecutor1::release(void* ptr, void* customNodeLibraryInternalManager) {
    auto v2Ptr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
    if (v2Ptr != nullptr) {
        return v2Ptr->release(ptr);
    } else {
        return 999;
    }
}

bool NodeLibrary::isValid() const {
    return execute != nullptr &&
           getInputsInfo != nullptr &&
           getOutputsInfo != nullptr &&
           release != nullptr;
}

bool NodeLibraryV2::isValid() const {
    return initialize != nullptr &&
           deinitialize != nullptr &&
           execute != nullptr &&
           getInputsInfo != nullptr &&
           getOutputsInfo != nullptr &&
           release != nullptr;
}

}  // namespace ovms
