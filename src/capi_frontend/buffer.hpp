#pragma once
//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <memory>
#include <optional>
#include <typeinfo>
#include "../logging.hpp"
#include "../ovms.h"  // NOLINT
namespace ovms {
struct BaseHolder {
    const void* ptr;
    BaseHolder(const void* ptr) :
        ptr(ptr) {
       SPDLOG_ERROR("ER:{}", ptr);
        }
    virtual ~BaseHolder() = default;
};
/*
 * Class intended for deep copy storage of more complex object in case we require memory
 * ownership to remain in Buffer. It allows us to not pollute Buffer class with dependencies on actual
 * types, and performs deep copy of underlying object as long as it has proper copy-constructor.
 */
template <typename T>
class DeepCopyHolder : public BaseHolder {
    T storage;

public:
    //    DeepCopyHolder(T* val) : storage(std::make_unique<T>(*val)) {} // here happens implicit copy
    DeepCopyHolder(const T* val) :
        BaseHolder(reinterpret_cast<const void*>(&storage)),
        storage(3, "a") {
       SPDLOG_ERROR("ER:{}", (void*)val);
       SPDLOG_ERROR("ER:{}", typeid(val).name());
       storage = *val;
    }
};
class Buffer {
    size_t byteSize{0};
    OVMS_BufferType bufferType;
    std::optional<uint32_t> bufferDeviceId;
    std::unique_ptr<char[]> ownedCopy = nullptr;
    std::unique_ptr<BaseHolder> holder;
    const void* ptr{nullptr};

public:
    template <typename T>
    Buffer(const T* val, bool createCopy) :
        bufferType(OVMS_BUFFERTYPE_CPU),
        holder(createCopy ? std::unique_ptr<BaseHolder>(new DeepCopyHolder<T>(val)) : nullptr),
        ptr(createCopy ? holder->ptr : val) {
       SPDLOG_ERROR("ER");
        //ptr = createCopy ? holder->ptr : val; // TODO how to pass this as string
    }
    Buffer(const void* ptr, size_t byteSize, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> bufferDeviceId = std::nullopt, bool createCopy = false);
    Buffer(size_t byteSize, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> bufferDeviceId = std::nullopt);
    //    template<typename T> Buffer(T val, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU) : bufferType(bufferType), hold(std::make_unique<Holder<T>>(std::move(val))) {}
    ~Buffer();
    const void* data() const;
    void* data();
    OVMS_BufferType getBufferType() const;
    const std::optional<uint32_t>& getDeviceId() const;
    size_t getByteSize() const;
};

}  // namespace ovms
