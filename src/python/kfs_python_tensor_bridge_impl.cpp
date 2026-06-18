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
//
// KFS OVMS_PY_TENSOR bridge — compiled into libpython_calculators.so.
//
// Why this file lives in libpython_calculators.so (not libovmspython.so):
//   Both PythonExecutorCalculator and this bridge must use the same RTTI for
//   PyObjectWrapper<py::object> so that mediapipe's packet.Get<T>() succeeds
//   across both the input (KFS→packet) and output (packet→KFS) paths.  Both
//   DSOs are built without -fvisibility=hidden, so the dynamic linker
//   deduplicates the typeinfo and typeid comparisons work correctly.

#include "python_backend.hpp"
#include "utils.hpp"
#include "../kfs_python_tensor_bridge.hpp"
#include "../logging.hpp"
#include "../status.hpp"

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma warning(pop)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Input path: KFS raw bytes → PyObjectWrapper<py::object> packet → graph
// ---------------------------------------------------------------------------
static int kfsBridgeDeserializeAndPush(
    const char* streamName,
    const void* rawData,
    size_t rawSize,
    const int64_t* shape,
    size_t shapeLen,
    const char* datatype,
    void* graphPtr,
    int64_t timestampMicros) {
    try {
        py::gil_scoped_acquire acquire;
        py::module_ pyovms = py::module_::import("pyovms");
        py::object TensorClass = pyovms.attr("Tensor");

        std::vector<py::ssize_t> pyShape(shape, shape + shapeLen);
        // copy=true so the packet owns its data independently of the request buffer
        py::object tensor = TensorClass.attr("_create_from_data")(
            streamName,
            const_cast<void*>(rawData),
            pyShape,
            datatype,
            py::ssize_t(rawSize),
            true /* copy */);

        auto* wrapper = new ovms::PyObjectWrapper<py::object>(tensor);
        auto packet = ::mediapipe::packet_internal::Create(
                          new ::mediapipe::packet_internal::Holder<
                              ovms::PyObjectWrapper<py::object>>(wrapper))
                          .At(mediapipe::Timestamp(timestampMicros));

        auto* graph = static_cast<mediapipe::CalculatorGraph*>(graphPtr);
        auto absStatus = graph->AddPacketToInputStream(streamName, packet);
        if (!absStatus.ok()) {
            SPDLOG_DEBUG("KFS Python tensor bridge: AddPacketToInputStream failed: {}",
                absStatus.ToString());
            return -static_cast<int>(ovms::StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
        }
        return 0;
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_DEBUG("KFS Python tensor bridge deserialize: Python error: {}", e.what());
        return -static_cast<int>(ovms::StatusCode::UNKNOWN_ERROR);
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("KFS Python tensor bridge deserialize: error: {}", e.what());
        return -static_cast<int>(ovms::StatusCode::UNKNOWN_ERROR);
    }
}

// ---------------------------------------------------------------------------
// Output path: mediapipe packet → metadata + data pointer for KFS response
// ---------------------------------------------------------------------------
static int kfsBridgeExtractPacketData(
    const void* packetPtr,
    char* datatypeBuf,
    size_t datatypeMax,
    int64_t* shapeBuf,
    size_t shapeMax,
    size_t* shapeLenOut,
    const void** dataPtrOut,
    size_t* dataSizeOut) {
    try {
        const auto* packet = static_cast<const mediapipe::Packet*>(packetPtr);
        const auto& wrapper = packet->Get<ovms::PyObjectWrapper<py::object>>();

        std::string datatype = wrapper.getProperty<std::string>("datatype");
        std::vector<py::ssize_t> userShape =
            wrapper.getProperty<std::vector<py::ssize_t>>("userShape");
        size_t size = wrapper.getProperty<size_t>("size");
        void* ptr = wrapper.getProperty<void*>("ptr");

        std::strncpy(datatypeBuf, datatype.c_str(), datatypeMax - 1);
        datatypeBuf[datatypeMax - 1] = '\0';

        *shapeLenOut = std::min(userShape.size(), shapeMax);
        for (size_t i = 0; i < *shapeLenOut; i++) {
            shapeBuf[i] = static_cast<int64_t>(userShape[i]);
        }

        *dataPtrOut = ptr;
        *dataSizeOut = size;
        return 0;
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_DEBUG("KFS Python tensor bridge serialize: Python error: {}", e.what());
        return -static_cast<int>(ovms::StatusCode::UNKNOWN_ERROR);
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("KFS Python tensor bridge serialize: error: {}", e.what());
        return -static_cast<int>(ovms::StatusCode::UNKNOWN_ERROR);
    }
}

// Static vtable — always valid for the lifetime of libpython_calculators.so
static const ovms::KfsPyTensorBridgeVTable g_kfsBridgeVTable{
    kfsBridgeDeserializeAndPush,
    kfsBridgeExtractPacketData,
};

}  // namespace

#if defined(_WIN32)
#define KFS_BRIDGE_EXPORT __declspec(dllexport)
#else
#define KFS_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

// Exported symbol loaded by python_calculators_plugin_loader via dlsym.
// Returns the vtable pointer if libpython_calculators.so is loaded and Python
// calculators are registered; nullptr is never returned (the vtable is always
// available once the DSO is loaded).
extern "C" KFS_BRIDGE_EXPORT const ovms::KfsPyTensorBridgeVTable*
OVMS_getKfsPyTensorBridgeVTable() {
    return &g_kfsBridgeVTable;
}
