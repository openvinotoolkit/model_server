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

namespace ovms {

#if defined(_WIN32)
#define KFS_BRIDGE_EXPORT __declspec(dllexport)
#else
#define KFS_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

namespace {
std::atomic<const KfsPyTensorBridgeVTable*> g_vtable{nullptr};
}  // namespace

void setKfsPyTensorBridgeVTable(const KfsPyTensorBridgeVTable* vtable) {
    g_vtable.store(vtable, std::memory_order_release);
}

const KfsPyTensorBridgeVTable* getKfsPyTensorBridgeVTable() {
    return g_vtable.load(std::memory_order_acquire);
}

extern "C" KFS_BRIDGE_EXPORT void OVMS_setKfsPyTensorBridgeVTable(const KfsPyTensorBridgeVTable* vtable) {
    setKfsPyTensorBridgeVTable(vtable);
}

}  // namespace ovms
