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
namespace ovms {

NodeLibraryExecutor::NodeLibraryExecutor(std::unique_ptr<NodeLibraryBase>&& ptr) :
        nodeLibrary(std::move(ptr)) {
     auto v2Ptr = dynamic_cast<NodeLibraryV2*>(nodeLibrary.get());
     if (v2Ptr != nullptr) {
         this->initialize = v2Ptr->initialize;
         this->deinitialize = v2Ptr->deinitialize;
         this->execute = v2Ptr->execute;
         this->getInputsInfo = v2Ptr->getInputsInfo;
         this->getOutputsInfo = v2Ptr->getOutputsInfo;
         this->release = v2Ptr->release;
     } else {
        auto v1Ptr = dynamic_cast<NodeLibrary*>(nodeLibrary.get());
         this->translator = V1ToV2ApiTranslator(*v1Ptr);
         using namespace std::placeholders;
         auto fInit = std::bind(&V1ToV2ApiTranslator::initialize, &translator, _1, _2, _3);
         this->initialize = fInit;
         this->initialize = translator.initialize;
         this->deinitialize = translator.deinitialize;
         this->execute = translator.execute;
         this->getInputsInfo = translator.getInputsInfo;
         this->getOutputsInfo = translator.getOutputsInfo;
         this->release = translator.release;
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
