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
#include "modelinstanceunloadguard.hpp"

#include <atomic>
#include <cstddef>

#include "modelinstance.hpp"
#include "logging.hpp"

namespace ovms {
ModelInstanceUnloadGuard::ModelInstanceUnloadGuard(ModelInstance& modelInstance) :
    modelInstance(modelInstance) {
    modelInstance.increasePredictRequestsHandlesCount();
    //SPDLOG_DEBUG("predictRequestsHandlesCount {}", modelInstance.predictRequestsHandlesCount);
}

uint64_t ModelInstanceUnloadGuard::GetHandlesCount() {
    SPDLOG_DEBUG("modelInstance ADDRESS: {}", (void*)&modelInstance);
    if (modelInstance.predictRequestsHandlesCount)
        return modelInstance.predictRequestsHandlesCount.load();
    else
        SPDLOG_DEBUG("NULL modelInstance");

    return 999;
}

ModelInstanceUnloadGuard::~ModelInstanceUnloadGuard() {
    modelInstance.decreasePredictRequestsHandlesCount();
    //SPDLOG_DEBUG("DESTROYED predictRequestsHandlesCount {}", modelInstance.predictRequestsHandlesCount);
    //SPDLOG_DEBUG("GUARD DESTROYED");
}
}  // namespace ovms
