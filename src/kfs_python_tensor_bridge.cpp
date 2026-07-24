//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "kfs_python_tensor_bridge.hpp"

#include <atomic>

#include <spdlog/spdlog.h>

namespace ovms {

#if defined(_WIN32)
#define KFS_BRIDGE_EXPORT __declspec(dllexport)
#else
#define KFS_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

namespace {
// Non-owning pointer. The vtable object is owned by the caller of
// setKfsPyTensorBridgeVTable() (typically a statically-initialized instance
// in the Python runtime plugin). The caller must ensure the vtable outlives
// every getKfsPyTensorBridgeVTable() consumer, and must reset it to nullptr
// before the underlying storage is destroyed.
std::atomic<const KfsPyTensorBridgeVTable*> g_vtable{nullptr};
}  // namespace

bool setKfsPyTensorBridgeVTable(const KfsPyTensorBridgeVTable* vtable) {
    if (vtable != nullptr && vtable->abiVersion != KFS_PY_TENSOR_BRIDGE_ABI_VERSION) {
        SPDLOG_ERROR(
            "KFS Python tensor bridge ABI version mismatch: expected {}, got {}. "
            "Refusing to install vtable; OVMS_PY_TENSOR KFS paths will remain unavailable. "
            "Rebuild libpython_calculators against the current ovms tree.",
            KFS_PY_TENSOR_BRIDGE_ABI_VERSION, vtable->abiVersion);
        g_vtable.store(nullptr, std::memory_order_release);
        return false;
    }
    g_vtable.store(vtable, std::memory_order_release);
    return true;
}

const KfsPyTensorBridgeVTable* getKfsPyTensorBridgeVTable() {
    return g_vtable.load(std::memory_order_acquire);
}

extern "C" KFS_BRIDGE_EXPORT int OVMS_setKfsPyTensorBridgeVTable(const KfsPyTensorBridgeVTable* vtable) {
    return setKfsPyTensorBridgeVTable(vtable) ? 1 : 0;
}

}  // namespace ovms
