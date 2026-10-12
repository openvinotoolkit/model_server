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
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 6001 6385 6386 6326 6011 6246 4456)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/kfs_python_tensor_bridge.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"
#include "py_object_handle.hpp"
#include "python_calculators_plugin_api.hpp"
#include "python_calculators_plugin_loader.hpp"

namespace ovms {
namespace {

int kfsBridgeDeserializeAndPush(
    const char* streamName,
    const void* rawData,
    size_t rawSize,
    const int64_t* shape,
    size_t shapeLen,
    const char* datatype,
    void* graphPtr,
    int64_t timestampMicros) {
    const auto* api = getPythonCalculatorsPluginApi();
    if (api == nullptr) {
        return -static_cast<int>(StatusCode::NOT_IMPLEMENTED);
    }
    void* tensor = nullptr;
    std::string message;
    if (api->createTensor(streamName, rawData, std::vector<int64_t>(shape, shape + shapeLen), datatype, rawSize, tensor, message) != PythonPluginResult::OK) {
        SPDLOG_DEBUG("KFS Python tensor bridge deserialize error: {}", message);
        return -static_cast<int>(StatusCode::UNKNOWN_ERROR);
    }
    auto packet = ::mediapipe::Adopt(new PyObjectHandle(tensor, api->releaseObject)).At(::mediapipe::Timestamp(timestampMicros));
    auto absStatus = static_cast<::mediapipe::CalculatorGraph*>(graphPtr)->AddPacketToInputStream(streamName, std::move(packet));
    if (!absStatus.ok()) {
        SPDLOG_ERROR(
            "KFS Python tensor bridge: AddPacketToInputStream failed for stream: {} datatype: {} raw_size: {} timestamp_us: {} status: {}",
            streamName, datatype, rawSize, timestampMicros, absStatus.ToString());
        return -static_cast<int>(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    }
    return 0;
}

int kfsBridgeExtractPacketData(
    const void* packetPtr,
    char* datatypeBuf,
    size_t datatypeMax,
    int64_t* shapeBuf,
    size_t shapeMax,
    size_t* shapeLenOut,
    const void** dataPtrOut,
    size_t* dataSizeOut) {
    const auto* api = getPythonCalculatorsPluginApi();
    if (api == nullptr) {
        return -static_cast<int>(StatusCode::NOT_IMPLEMENTED);
    }
    if (datatypeBuf == nullptr || datatypeMax == 0 || shapeBuf == nullptr || shapeMax == 0 || shapeLenOut == nullptr || dataPtrOut == nullptr || dataSizeOut == nullptr) {
        return -static_cast<int>(StatusCode::INTERNAL_ERROR);
    }
    const auto* packet = static_cast<const ::mediapipe::Packet*>(packetPtr);
    if (!packet->ValidateAsType<PyObjectHandle>().ok()) {
        return -static_cast<int>(StatusCode::INTERNAL_ERROR);
    }
    std::string datatype;
    std::vector<int64_t> shape;
    std::string message;
    if (api->getTensorInfo(packet->Get<PyObjectHandle>().get(), datatype, shape, *dataPtrOut, *dataSizeOut, message) != PythonPluginResult::OK) {
        SPDLOG_DEBUG("KFS Python tensor bridge serialize error: {}", message);
        return -static_cast<int>(StatusCode::UNKNOWN_ERROR);
    }
    std::strncpy(datatypeBuf, datatype.c_str(), datatypeMax - 1);
    datatypeBuf[datatypeMax - 1] = '\0';
    *shapeLenOut = std::min(shape.size(), shapeMax);
    std::copy_n(shape.begin(), *shapeLenOut, shapeBuf);
    return 0;
}

const KfsPyTensorBridgeVTable kfsPyTensorBridgeVTable{
    KFS_PY_TENSOR_BRIDGE_ABI_VERSION,
    kfsBridgeDeserializeAndPush,
    kfsBridgeExtractPacketData,
};

bool kfsPyTensorBridgeInstalled = []() {
    return setKfsPyTensorBridgeVTable(&kfsPyTensorBridgeVTable);
}();

}  // namespace
}  // namespace ovms
