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

#include "status.hpp"

namespace ovms {

const std::map<const StatusCode, const std::pair<grpc::StatusCode, const std::string>> Status::grpcMessages = {
    {StatusCode::OK,
        {grpc::StatusCode::OK, ""}},

    {StatusCode::PATH_INVALID,
        {grpc::StatusCode::INTERNAL, "The provided base path is invalid or doesn't exists"}},
    {StatusCode::FILE_INVALID,
        {grpc::StatusCode::INTERNAL, "File not found or cannot open"}},
    {StatusCode::NETWORK_NOT_LOADED,
        {grpc::StatusCode::INTERNAL, "Error while loading a network"}},
    {StatusCode::JSON_INVALID,
        {grpc::StatusCode::INTERNAL, "The file is not valid json"}},
    {StatusCode::MODELINSTANCE_NOT_FOUND,
        {grpc::StatusCode::INTERNAL, "ModelInstance not found"}},
    {StatusCode::SHAPE_WRONG_FORMAT,
        {grpc::StatusCode::INTERNAL, "The provided shape is in wrong format"}},
    {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT,
        {grpc::StatusCode::INTERNAL, "Plugin config is in wrong format"}},
    {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT,
        {grpc::StatusCode::INTERNAL, "Model version policy is in wrong format"}},
    {StatusCode::RESHAPE_ERROR,
        {grpc::StatusCode::INTERNAL, "Model reshape failed"}},
    {StatusCode::AMBIGUOUS_SHAPE_PARAM,
        {grpc::StatusCode::INTERNAL, "Anonymous fixed shape is invalid for models with multiple inputs"}},
    {StatusCode::MODEL_MISSING,
        {grpc::StatusCode::NOT_FOUND, "Model with requested name and/or version is not found"}},
    {StatusCode::MODEL_NAME_MISSING,
        {grpc::StatusCode::NOT_FOUND, "Model with requested name is not found"}},
    {StatusCode::MODEL_VERSION_MISSING,
        {grpc::StatusCode::NOT_FOUND, "Model with requested version is not found"}},
    {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE,
        {grpc::StatusCode::NOT_FOUND, "Model with requested version is retired"}},
    {StatusCode::MODEL_VERSION_NOT_LOADED_YET,
        {grpc::StatusCode::NOT_FOUND, "Model with requested version is not loaded yet"}},
    {StatusCode::MODEL_SPEC_MISSING,
        {grpc::StatusCode::INVALID_ARGUMENT, "model_spec missing in request"}},
    {StatusCode::INVALID_SIGNATURE_DEF,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid signature name"}},

    // Predict request validation
    {StatusCode::INVALID_NO_OF_INPUTS,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid number of inputs"}},
    {StatusCode::INVALID_MISSING_INPUT,
        {grpc::StatusCode::INVALID_ARGUMENT, "Missing input with specific name"}},
    {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid number of shape dimensions"}},
    {StatusCode::INVALID_BATCH_SIZE,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid input batch size"}},
    {StatusCode::INVALID_SHAPE,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid input shape"}},
    {StatusCode::INVALID_PRECISION,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid input precision"}},
    {StatusCode::INVALID_VALUE_COUNT,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid number of values in tensor proto container"}},
    {StatusCode::INVALID_CONTENT_SIZE,
        {grpc::StatusCode::INVALID_ARGUMENT, "Invalid content size of tensor proto"}},

    // Deserialization
    {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION,  // Should never occur - ModelInstance::validate takes care of that
        {grpc::StatusCode::INTERNAL, "Unsupported deserialization precision"}},
    {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR,
        {grpc::StatusCode::INTERNAL, "Internal deserialization error"}},

    // Inference
    {StatusCode::OV_INTERNAL_INFERENCE_ERROR,
        {grpc::StatusCode::INTERNAL, "Internal inference error"}},

    // Serialization
    {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION,  // Should never occur - it should be validated during model loading
        {grpc::StatusCode::INTERNAL, "Unsupported serialization precision"}},
    {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR,
        {grpc::StatusCode::INTERNAL, "Internal serialization error"}},

    // GetModelStatus
    {StatusCode::INTERNAL_ERROR,
        {grpc::StatusCode::INTERNAL, "Internal server error"}},
};

const net_http::HTTPStatusCode Status::http() const {
    using net_http::HTTPStatusCode;
    switch (code) {
    case StatusCode::OK:
        return HTTPStatusCode::OK;
    case StatusCode::MODEL_NAME_MISSING:
        return HTTPStatusCode::NOT_FOUND;
    case StatusCode::MODEL_VERSION_MISSING:
        return HTTPStatusCode::NOT_FOUND;
    case StatusCode::REST_NOT_FOUND:
        return HTTPStatusCode::NOT_FOUND;
    case StatusCode::REST_COULD_NOT_PARSE_VERSION:
    case StatusCode::REST_MALFORMED_REQUEST:
        return HTTPStatusCode::BAD_REQUEST;
    case StatusCode::REST_BODY_IS_NOT_AN_OBJECT:
    case StatusCode::REST_PREDICT_UNKNOWN_ORDER:
    case StatusCode::REST_INSTANCES_NOT_AN_ARRAY:
    case StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT:
    case StatusCode::REST_INPUT_NOT_PREALLOCATED:
    case StatusCode::REST_NO_INSTANCES_FOUND:
    case StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED:
    case StatusCode::REST_COULD_NOT_PARSE_INSTANCE:
    case StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER:
    case StatusCode::REST_INPUTS_NOT_AN_OBJECT:
    case StatusCode::REST_NO_INPUTS_FOUND:
    case StatusCode::REST_COULD_NOT_PARSE_INPUT:
        return HTTPStatusCode::BAD_REQUEST;
    default:
        return HTTPStatusCode::ERROR;
    }
}

}  // namespace ovms
