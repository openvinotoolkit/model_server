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
#include "mediapipe_utils.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"

namespace ovms {
const std::string KFS_REQUEST_PREFIX{"REQUEST"};
const std::string KFS_RESPONSE_PREFIX{"RESPONSE"};
const std::string MP_TENSOR_PREFIX{"TENSOR"};
const std::string TF_TENSOR_PREFIX{"TFTENSOR"};
const std::string TFLITE_TENSOR_PREFIX{"TFLITE_TENSOR"};
const std::string OV_TENSOR_PREFIX{"OVTENSOR"};
const std::string OVMS_PY_TENSOR_PREFIX{"OVMS_PY_TENSOR"};
const std::string MP_IMAGE_PREFIX{"IMAGE"};

const std::string EMPTY_STREAM_NAME{""};

static std::string streamTypeToString(MediaPipeStreamType streamType) {
    std::string streamTypeStr;
    switch (streamType) {
    case MediaPipeStreamType::INPUT:
        streamTypeStr = "input";
        break;
    case MediaPipeStreamType::OUTPUT:
        streamTypeStr = "output";
        break;
    }
    return streamTypeStr;
}

std::pair<std::string, mediapipe_packet_type_enum> getStreamNamePair(const std::string& streamFullName, MediaPipeStreamType streamType) {
    std::string streamTypeStr = streamTypeToString(streamType);

    static std::unordered_map<std::string, mediapipe_packet_type_enum> prefix2enum{
        {KFS_REQUEST_PREFIX, mediapipe_packet_type_enum::KFS_REQUEST},
        {KFS_RESPONSE_PREFIX, mediapipe_packet_type_enum::KFS_RESPONSE},
        {TF_TENSOR_PREFIX, mediapipe_packet_type_enum::TFTENSOR},
        {TFLITE_TENSOR_PREFIX, mediapipe_packet_type_enum::TFLITETENSOR},
        {OV_TENSOR_PREFIX, mediapipe_packet_type_enum::OVTENSOR},
        {OVMS_PY_TENSOR_PREFIX, mediapipe_packet_type_enum::OVMS_PY_TENSOR},
        {MP_TENSOR_PREFIX, mediapipe_packet_type_enum::MPTENSOR},
        {MP_IMAGE_PREFIX, mediapipe_packet_type_enum::MEDIAPIPE_IMAGE}};
    std::vector<std::string> tokens = tokenize(streamFullName, ':');
    // MP convention
    // input_stream: "lowercase_input_stream_name"
    // input_stream: "PACKET_TAG:lowercase_input_stream_name"
    // input_stream: "PACKET_TAG:[0-9]:lowercase_input_stream_name"
    if (tokens.size() == 2 || tokens.size() == 3) {
        auto it = std::find_if(prefix2enum.begin(), prefix2enum.end(), [tokens](const auto& p) {
            const auto& [k, v] = p;
            bool b = startsWith(tokens[0], k);
            return b;
        });
        size_t inputStreamIndex = tokens.size() - 1;
        if (it != prefix2enum.end()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting {} stream: {} packet type: {} from: {}", streamTypeStr, tokens[inputStreamIndex], it->first, streamFullName);
            return {tokens[inputStreamIndex], it->second};
        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting {} stream: {} packet type: {} from: {}", streamTypeStr, tokens[inputStreamIndex], "UNKNOWN", streamFullName);
            return {tokens[inputStreamIndex], mediapipe_packet_type_enum::UNKNOWN};
        }
    } else if (tokens.size() == 1) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting {} stream: {} packet type: {} from: {}", streamTypeStr, tokens[0], "UNKNOWN", streamFullName);
        return {tokens[0], mediapipe_packet_type_enum::UNKNOWN};
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting {} stream: {} packet type: {} from: {}", streamTypeStr, "", "UNKNOWN", streamFullName);
    return {"", mediapipe_packet_type_enum::UNKNOWN};
}

std::string getStreamName(const std::string& streamFullName) {
    std::vector<std::string> tokens = tokenize(streamFullName, ':');
    // Stream name is the last part of the full name
    if (tokens.size() > 0 || tokens.size() <= 3)
        return tokens[tokens.size() - 1];
    return EMPTY_STREAM_NAME;
}

}  // namespace ovms
