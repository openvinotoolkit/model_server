//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unordered_set>

#include "test_utils.hpp"
#include "../status.hpp"

namespace ovms {
StatusCode& operator++(StatusCode& statusCode) {
    if (statusCode == StatusCode::STATUS_CODE_END) {
        throw std::out_of_range("for E& operator ++ (E&)");
    }
    statusCode = StatusCode(static_cast<std::underlying_type<StatusCode>::type>(statusCode) +1);
    return statusCode;
}

const std::unordered_set<StatusCode> standardWhiteList = {
    StatusCode::FILESYSTEM_ERROR,
    StatusCode::JSON_SERIALIZATION_ERROR,
    StatusCode::GRPC_CHANNEL_ARG_WRONG_FORMAT,
    StatusCode::CONFIG_FILE_TIMESTAMP_READING_FAILED,
    StatusCode::RESHAPE_REQUIRED,
    StatusCode::BATCHSIZE_CHANGE_REQUIRED,
    StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER,
    StatusCode::REQUESTED_MODEL_TYPE_CHANGE,
    StatusCode::INVALID_MISSING_OUTPUT,
    StatusCode::OV_CLONE_BLOB_ERROR,
    StatusCode::UNKNOWN_ERROR,
    StatusCode::REST_NOT_FOUND,
    StatusCode::REST_COULD_NOT_PARSE_VERSION,
    StatusCode::REST_MALFORMED_REQUEST,
    StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE,
    StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY,
    StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_BLOB_SHARD_COUNT,
    StatusCode::PIPELINE_MANUAL_GATHERING_FROM_MULTIPLE_NODES_NOT_SUPPORTED,
    StatusCode::PIPELINE_NOT_ENOUGH_SHAPE_DIMENSIONS_TO_DEMULTIPLY,
    StatusCode::IMAGE_PARSING_FAILED,
    StatusCode::OK_NOT_RELOADED,
    StatusCode::OK_RELOADED
};

TEST(StatusCodeTest, AllStatusCodesMapped) {
    for(auto statusCode = StatusCode::OK; statusCode != StatusCode::STATUS_CODE_END; ++statusCode) {
        if (standardWhiteList.find(statusCode) == standardWhiteList.end()) {
            spdlog::info("Checking statusCode: {}\n", statusCode);
            Status status = Status(statusCode);
            ASSERT_NE(status.string(), "Undefined error");
            spdlog::info("StatusCode: {} succeded\n", statusCode);
        }
    }
}
}   // namespace ovms
