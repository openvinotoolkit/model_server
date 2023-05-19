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

#include "grpc_utils.hpp"

#include <string>
#include <unordered_map>

#include "status.hpp"

namespace ovms {
const grpc::Status grpc(const Status& status) {
    static const std::unordered_map<const StatusCode, grpc::StatusCode> grpcStatusMap = {
        {StatusCode::OK, grpc::StatusCode::OK},

        {StatusCode::PATH_INVALID, grpc::StatusCode::INTERNAL},
        {StatusCode::FILE_INVALID, grpc::StatusCode::INTERNAL},
        {StatusCode::NO_MODEL_VERSION_AVAILABLE, grpc::StatusCode::INTERNAL},
        {StatusCode::MODEL_NOT_LOADED, grpc::StatusCode::INTERNAL},
        {StatusCode::JSON_INVALID, grpc::StatusCode::INTERNAL},
        {StatusCode::MODELINSTANCE_NOT_FOUND, grpc::StatusCode::INTERNAL},
        {StatusCode::SHAPE_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
        {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
        {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
        {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, grpc::StatusCode::INTERNAL},
        {StatusCode::RESHAPE_ERROR, grpc::StatusCode::FAILED_PRECONDITION},
        {StatusCode::MODEL_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MODEL_NAME_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_NOT_LOADED_YET, grpc::StatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, grpc::StatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, grpc::StatusCode::NOT_FOUND},
        {StatusCode::MODEL_SPEC_MISSING, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::MODEL_VERSION_INVALID_FORMAT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_SIGNATURE_DEF, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS, grpc::StatusCode::ABORTED},
        {StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE, grpc::StatusCode::FAILED_PRECONDITION},

        // Sequence management
        {StatusCode::SEQUENCE_MISSING, grpc::StatusCode::NOT_FOUND},
        {StatusCode::SEQUENCE_ALREADY_EXISTS, grpc::StatusCode::ALREADY_EXISTS},
        {StatusCode::SEQUENCE_ID_NOT_PROVIDED, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::SEQUENCE_ID_BAD_TYPE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::SEQUENCE_TERMINATED, grpc::StatusCode::FAILED_PRECONDITION},
        {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::MAX_SEQUENCE_NUMBER_REACHED, grpc::StatusCode::UNAVAILABLE},

        // Predict request validation
        {StatusCode::INVALID_NO_OF_INPUTS, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_MISSING_INPUT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_BATCH_SIZE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_SHAPE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_BUFFER_TYPE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_DEVICE_ID, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_STRING_INPUT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_INPUT_FORMAT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_PRECISION, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_VALUE_COUNT, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_CONTENT_SIZE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::INVALID_MESSAGE_STRUCTURE, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::UNSUPPORTED_LAYOUT, grpc::StatusCode::INVALID_ARGUMENT},

        // Deserialization

        // Should never occur - ModelInstance::validate takes care of that
        {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, grpc::StatusCode::INTERNAL},
        {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, grpc::StatusCode::INTERNAL},

        // Inference
        {StatusCode::OV_INTERNAL_INFERENCE_ERROR, grpc::StatusCode::INTERNAL},

        // Serialization

        // Should never occur - it should be validated during model loading
        {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, grpc::StatusCode::INTERNAL},
        {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, grpc::StatusCode::INTERNAL},

        // GetModelStatus
        {StatusCode::INTERNAL_ERROR, grpc::StatusCode::INTERNAL},

        // Binary input
        {StatusCode::INVALID_NO_OF_CHANNELS, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::STRING_VAL_EMPTY, grpc::StatusCode::INVALID_ARGUMENT},
        {StatusCode::BYTES_CONTENTS_EMPTY, grpc::StatusCode::INVALID_ARGUMENT},
    };
    auto it = grpcStatusMap.find(status.getCode());
    if (it != grpcStatusMap.end()) {
        return grpc::Status(it->second, status.string());
    } else {
        return grpc::Status(grpc::StatusCode::UNKNOWN, "Unknown error");
    }
}
}  // namespace ovms
