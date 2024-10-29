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

#include <iostream>
#include <memory>

#include "node_library.hpp"

namespace ovms {

struct CNLIMWrapper {
    void* ptr;
    deinitialize_fn deinitialize = nullptr;

    CNLIMWrapper(void* CNLIM, deinitialize_fn deinitialize) :
        ptr(CNLIM),
        deinitialize(deinitialize) {}

    ~CNLIMWrapper() {
        deinitialize(ptr);
    }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-function"
// TODO: move to class ?
static void* getCNLIMWrapperPtr(const std::shared_ptr<CNLIMWrapper>& wrapper) {
    if (wrapper == nullptr) {
        return nullptr;
    }
    return wrapper->ptr;
}
#pragma GCC diagnostic pop
}  // namespace ovms
