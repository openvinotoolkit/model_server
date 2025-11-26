//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <memory>
#include <utility>

template <typename T>
class UniqueObjectHolder {
    std::unique_ptr<T> object = nullptr;

public:
    std::unique_ptr<T>& get() {
        return object;
    }

    void reset() {
        object.reset();
    }

    bool valid() const {
        return object != nullptr;
    }
};

template <typename T>
class CopyableObjectWrapper {
    std::shared_ptr<UniqueObjectHolder<T>> objectHolder = nullptr;

public:
    explicit CopyableObjectWrapper() :
        objectHolder(std::make_shared<UniqueObjectHolder<T>>()) {}

    explicit CopyableObjectWrapper(std::shared_ptr<UniqueObjectHolder<T>> objectHolder) :
        objectHolder(std::move(objectHolder)) {}

    std::shared_ptr<UniqueObjectHolder<T>>& getObjectHolder() {
        return objectHolder;
    }
};
