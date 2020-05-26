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
#pragma once

#include <map>
#include <string>
#include <utility>

#include <grpcpp/server_context.h>

namespace ovms {

enum class StatusCode {
    OK,                                 /*!< Success */

    PATH_INVALID,                       /*!< The provided path is invalid or doesn't exists */
    FILE_INVALID,                       /*!< File not found or cannot open */
    NETWORK_NOT_LOADED,
    JSON_INVALID,                       /*!< The file/content is not valid json */
    MODELINSTANCE_NOT_FOUND,
    SHAPE_WRONG_FORMAT,                 /*!< The provided shape param is in wrong format */
    PLUGIN_CONFIG_WRONG_FORMAT,         /*!< Plugin config is in wrong format */
    MODEL_VERSION_POLICY_WRONG_FORMAT,  /*!< Model version policy is in wrong format */
    NO_MODEL_VERSION_AVAILABLE,         /*!< No model version found in path */

    // Model management
    MODEL_MISSING,                      /*!< Model with such name and/or version does not exist */
    MODEL_NAME_MISSING,                 /*!< Model with requested name is not found */
    MODEL_VERSION_MISSING,              /*!< Model with requested version is not found */
    MODEL_VERSION_NOT_LOADED_ANYMORE,   /*!< Model with requested version is retired */
    MODEL_VERSION_NOT_LOADED_YET,       /*!< Model with requested version is not loaded yet */


    // Predict request validation
    INVALID_NO_OF_INPUTS,               /*!< Invalid number of inputs */
    INVALID_MISSING_INPUT,              /*!< Missing one or more of inputs */
    INVALID_NO_OF_SHAPE_DIMENSIONS,     /*!< Invalid number of shape dimensions */
    INVALID_BATCH_SIZE,                 /*!< Input batch size other than required */
    INVALID_SHAPE,                      /*!< Invalid shape dimension number or dimension value */
    INVALID_PRECISION,                  /*!< Invalid precision */
    INVALID_VALUE_COUNT,                /*!< Invalid value count error status for uint16 and half float data types */
    INVALID_CONTENT_SIZE,               /*!< Invalid content size error status for types using tensor_content() */

    // Deserialization
    OV_UNSUPPORTED_DESERIALIZATION_PRECISION,   /*!< Unsupported deserialization precision, theoretically should never be returned since ModelInstance::validation checks against network precision */
    OV_INTERNAL_DESERIALIZATION_ERROR,          /*!< Error occured during deserialization */

    // Inference
    OV_INTERNAL_INFERENCE_ERROR,                /*!< Error occured during inference */

    // Serialization
    OV_UNSUPPORTED_SERIALIZATION_PRECISION,     /*!< Unsupported serializaton precision */
    OV_INTERNAL_SERIALIZATION_ERROR,            /*!< Error occurred during serialization */

    // GetModelStatus
    INVALID_SIGNATURE_DEF,  /*!< Requested signature is not supported */

    // Common request validation errors
    MODEL_SPEC_MISSING,     /*!< Request lacks model_spec */

    INTERNAL_ERROR,

    UNKNOWN_ERROR,
};

class Status {
    StatusCode code;

    static const std::map<const StatusCode, const std::pair<grpc::StatusCode, const std::string>> grpcMessages;

public:
    Status(StatusCode code = StatusCode::OK) :
        code(code) {}

    bool ok() const {
        return code == StatusCode::OK;
    }

    bool operator==(const Status& status) const {
        return this->code == status.code;
    }

    bool operator!=(const Status& status) const {
        return this->code != status.code;
    }

    const grpc::Status grpc() const {
        static const grpc::Status defaultStatus = grpc::Status(
            grpc::StatusCode::UNKNOWN,
            "Unknown error");

        auto it = grpcMessages.find(code);
        if (it != grpcMessages.end()) {
            return grpc::Status(it->second.first, it->second.second);
        } else {
            return defaultStatus;
        }
    }

    operator grpc::Status() const {
        return this->grpc();
    }

    const std::string& string() const {
        static const std::string defaultString = "Unknown error";

        auto it = grpcMessages.find(code);
        if (it != grpcMessages.end()) {
            return it->second.second;
        } else {
            return defaultString;
        }
    }

    operator const std::string&() const {
        return this->string();
    }
};

}  // namespace ovms
