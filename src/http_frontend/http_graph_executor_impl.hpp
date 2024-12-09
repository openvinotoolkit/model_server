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
#pragma once

#include <map>
#include <memory>
#include <string>

#include <rapidjson/document.h>

#include "../http_payload.hpp"
#include "../mediapipe_internal/packettypes.hpp"
#include "../status.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/packet.h"
#pragma GCC diagnostic pop

#if (PYTHON_DISABLE == 0)
#include "../python/python_backend.hpp"
#endif

#include "../http_async_writer_interface.hpp"

namespace ovms {

class PythonBackend;

using HttpReaderWriter = HttpAsyncWriter;

// Deserialization of parameters inside KServe gRPC request
// into mediapipe Packets.
// To be used by both - infer & inferStream.
Status deserializeInputSidePacketsFromFirstRequestImpl(
    std::map<std::string, mediapipe::Packet>& inputSidePackets,  // out
    const HttpPayload& request);                                 // in

// For unary graph execution request ID is forwarded to serialization function.
const std::string& getRequestId(
    const HttpPayload& request);

// Used by inferStream only.
// Whenever MediaPipe graph produces some packet, this function is triggered.
// Implementation should transform packet into KServe gRPC response and send it.
// Data race safety:
// MediaPipe packet available callbacks can be triggered simultaneously, from different threads.
// However, the graph executor synchronizes it with locking mechanism.
Status onPacketReadySerializeAndSendImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    HttpReaderWriter& serverReaderWriter);

// Used by infer only.
// infer produces single response and lets the caller send the response back on its own.
// This function is triggered when output poller has packet ready for serialization.
// Implementation should transform packet into KServe gRPC response and send it.
// Data race safety:
// This is always triggered on the same thread.
Status onPacketReadySerializeImpl(
    const std::string& requestId,
    const std::string& endpointName,
    const std::string& endpointVersion,
    const std::string& packetName,
    const mediapipe_packet_type_enum packetType,
    const ::mediapipe::Packet& packet,
    std::string& response);

// This is called whenever new request is received.
// It is responsible for creating packet(s) out of the request.
// It is also responsive of pushing the packet inside the graph.
// To be used by both - infer & inferStream.
Status createAndPushPacketsImpl(
    // The request wrapped in shared pointer.
    std::shared_ptr<const HttpPayload> request,
    // Graph input name => type mapping.
    // Request can contain multiple packets.
    // Implementation should validate for existence of such packet type.
    stream_types_mapping_t& inputTypes,
    // Context for creating Python bufferprotocol packets.
    PythonBackend* pythonBackend,
    // The graph instance.
    // Implementation is required to push the packet down the graph.
    ::mediapipe::CalculatorGraph& graph,
    // Timestamp to be used if request specified no manual timestamp.
    // Implementation is also expected to leave the timestamp in next
    // available state for usage in subsequent requests.
    ::mediapipe::Timestamp& currentTimestamp,
    // Unary (non-streaming) execution requires information about number of packets
    // in order to validate if all inputs were fed into the graph.
    size_t& numberOfPacketsCreated);

// This is called before subsequent createAndPushPacketsImpl in inferStream scenario.
// At this point we may reject requests with invalid data.
Status validateSubsequentRequestImpl(
    const HttpPayload& request,
    const std::string& endpointName,
    const std::string& endpointVersion,
    stream_types_mapping_t& inputTypes);

// Data race safety:
// This may be called from different threads.
// However, the caller implements synchronization mechanism
// which prevents writes at the same time.
Status sendErrorImpl(
    const std::string& message,
    HttpReaderWriter& serverReaderWriter);

// Imitation of stream.Read(...) in gRPC stream API
// Required for inferStream only.
bool waitForNewRequest(
    HttpReaderWriter& serverReaderWriter,
    HttpPayload& newRequest);

}  // namespace ovms
