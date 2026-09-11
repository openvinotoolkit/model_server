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
#pragma once
#include <cstddef>
#include <cstdint>

// Plain C-compatible vtable for the KFS OVMS_PY_TENSOR runtime bridge.
// This header intentionally has no pybind11, no KFS proto, and no mediapipe
// includes so that it can be included from any build target without pulling
// in heavy Python-runtime dependencies.

namespace ovms {

// Bump this whenever KfsPyTensorBridgeVTable layout or the semantics of any
// existing entry change. The vtable is only accepted by
// setKfsPyTensorBridgeVTable()/OVMS_setKfsPyTensorBridgeVTable() when the
// caller's abiVersion matches this value exactly. This protects against
// silently loading a libpython_calculators built against a different
// generation of the bridge contract.
inline constexpr uint32_t KFS_PY_TENSOR_BRIDGE_ABI_VERSION = 1;

struct KfsPyTensorBridgeVTable {
    // Must be initialized to KFS_PY_TENSOR_BRIDGE_ABI_VERSION by the producer
    // (libpython_calculators). Kept as the first field so that the setter can
    // safely read it before touching any other member even if the rest of the
    // layout drifts.
    uint32_t abiVersion;

    // Deserialize one OVMS_PY_TENSOR input from a KFS request and push the
    // resulting mediapipe packet into the graph.
    //
    // rawData/rawSize  – pointer and byte count from raw_input_contents
    // shape/shapeLen   – tensor shape dimensions
    // datatype         – null-terminated KFS datatype string (e.g. "FP32")
    // graph            – opaque mediapipe::CalculatorGraph*
    // timestampMicros  – mediapipe::Timestamp value
    //
    // Returns 0 on success, or -static_cast<int>(StatusCode) on failure.
    int (*deserializeAndPush)(
        const char* streamName,
        const void* rawData,
        size_t rawSize,
        const int64_t* shape,
        size_t shapeLen,
        const char* datatype,
        void* graph,
        int64_t timestampMicros);

    // Extract data from an OVMS_PY_TENSOR output mediapipe packet and copy
    // the metadata into caller-provided buffers.
    //
    // packet                   – opaque const mediapipe::Packet*
    // datatypeBuf/datatypeMax  – caller buffer for null-terminated datatype
    // shapeBuf/shapeMax        – caller buffer for shape; *shapeLenOut is set
    // dataPtrOut               – set to raw data pointer (valid until packet
    //                            is destroyed – copy immediately)
    // dataSizeOut              – set to byte count
    //
    // Returns 0 on success, or -static_cast<int>(StatusCode) on failure.
    int (*extractPacketData)(
        const void* packet,
        char* datatypeBuf,
        size_t datatypeMax,
        int64_t* shapeBuf,
        size_t shapeMax,
        size_t* shapeLenOut,
        const void** dataPtrOut,
        size_t* dataSizeOut);
};

// Store/retrieve the active vtable.  setKfsPyTensorBridgeVTable is called
// by python_calculators_plugin_loader after loading libpython_calculators.so.
// getKfsPyTensorBridgeVTable is called by the KFS graph executor at request
// time to decide whether the bridge is available.
//
// setKfsPyTensorBridgeVTable rejects any non-null vtable whose abiVersion
// does not equal KFS_PY_TENSOR_BRIDGE_ABI_VERSION; in that case the stored
// pointer is left as nullptr (so KFS OVMS_PY_TENSOR paths fall back to
// NOT_IMPLEMENTED instead of dispatching through a mismatched contract) and
// the function returns false. Passing nullptr always succeeds and returns
// true (used to clear the vtable on shutdown/test teardown).
bool setKfsPyTensorBridgeVTable(const KfsPyTensorBridgeVTable* vtable);
const KfsPyTensorBridgeVTable* getKfsPyTensorBridgeVTable();

}  // namespace ovms
