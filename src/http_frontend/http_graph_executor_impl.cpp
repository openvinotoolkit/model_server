//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "http_graph_executor_impl.hpp"

#include <string>
#include <utility>
#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6294 6201 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../mediapipe_internal/mediapipe_utils.hpp"

namespace ovms {

static const std::string UNUSED_REQUEST_ID = "";

Status deserializeInputSidePacketsFromFirstRequestImpl(
    std::map<std::string, mediapipe::Packet>& inputSidePackets,  // out
    const HttpPayload& request) {                                // in
    return StatusCode::OK;
}

const std::string& getRequestId(
    const HttpPayload& request) {
    return UNUSED_REQUEST_ID;
}

Status onPacketReadySerializeAndSendImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    HttpReaderWriter& serverReaderWriter) {
    std::string out;
    OVMS_RETURN_ON_FAIL(
        onPacketReadySerializeImpl(
            requestId,
            endpointName,
            endpointVersion,
            packetName,
            packetType,
            packet,
            out));
    serverReaderWriter.PartialReply(std::move(out));
    return StatusCode::OK;
}

Status onPacketReadySerializeImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    std::string& response) {
    response = packet.Get<std::string>();
    return StatusCode::OK;
}

Status createAndPushPacketsImpl(
    std::shared_ptr<const HttpPayload> request,
    stream_types_mapping_t& inputTypes,
    PythonBackend* pythonBackend,
    ::mediapipe::CalculatorGraph& graph,
    ::mediapipe::Timestamp& currentTimestamp,
    size_t& numberOfPacketsCreated) {
    MP_RETURN_ON_FAIL(
        graph.AddPacketToInputStream(
            "input", ::mediapipe::MakePacket<HttpPayload>(*request.get()).At(currentTimestamp)),  // TODO: Possibly avoid making copy
        "failed to deserialize",
        StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    numberOfPacketsCreated = 1;
    // TODO FIXME @atobisze properly implement on all exit paths
        auto now = std::chrono::system_clock::now();
        currentTimestamp = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
    return StatusCode::OK;
}

Status validateSubsequentRequestImpl(
    const HttpPayload& request,
    const std::string& endpointName,
    const std::string& endpointVersion,
    stream_types_mapping_t& inputTypes) {
    return StatusCode::OK;
}

Status sendErrorImpl(
    const std::string& message,
    HttpReaderWriter& serverReaderWriter) {
    serverReaderWriter.PartialReply("{\"error\": \"" + message + "\"}");
    return StatusCode::OK;
}

bool waitForNewRequest(
    HttpReaderWriter& serverReaderWriter,
    HttpPayload& newRequest) {
    return false;
}

}  // namespace ovms
