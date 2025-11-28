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
#include "http_server.hpp"

#ifdef _WIN32
#include <map>
#endif
#include <chrono>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "http_rest_api_handler.hpp"
#include "http_status_code.hpp"
#include "logging.hpp"
#include "status.hpp"

#include <drogon/drogon.h>

#include "drogon_http_async_writer_impl.hpp"
#include "http_frontend/multi_part_parser_drogon_impl.hpp"  // At this point there is no going back to net_http

namespace ovms {

static const ovms::HTTPStatusCode http(const ovms::Status& status) {
#ifdef __linux__
    const std::unordered_map<const StatusCode, ovms::HTTPStatusCode> httpStatusMap = {
#elif _WIN32
    const std::map<const StatusCode, ovms::HTTPStatusCode> httpStatusMap = {
#endif
        {StatusCode::OK, ovms::HTTPStatusCode::OK},
        {StatusCode::OK_RELOADED, ovms::HTTPStatusCode::CREATED},
        {StatusCode::OK_NOT_RELOADED, ovms::HTTPStatusCode::OK},

        // REST handler failure
        {StatusCode::REST_INVALID_URL, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_UNSUPPORTED_METHOD, ovms::HTTPStatusCode::NONE_ACC},
        {StatusCode::REST_NOT_FOUND, ovms::HTTPStatusCode::NOT_FOUND},

        // REST parser failure
        {StatusCode::REST_BODY_IS_NOT_AN_OBJECT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_PREDICT_UNKNOWN_ORDER, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_NOT_AN_ARRAY, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INPUT_NOT_PREALLOCATED, ovms::HTTPStatusCode::ERROR},
        {StatusCode::REST_NO_INSTANCES_FOUND, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_INSTANCE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INPUTS_NOT_AN_OBJECT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_NO_INPUTS_FOUND, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_INPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_OUTPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_PARAMETERS, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_PROTO_TO_STRING_ERROR, ovms::HTTPStatusCode::ERROR},
        {StatusCode::REST_UNSUPPORTED_PRECISION, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, ovms::HTTPStatusCode::ERROR},
        {StatusCode::REST_BINARY_BUFFER_EXCEEDED, ovms::HTTPStatusCode::BAD_REQUEST},

        {StatusCode::PATH_INVALID, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::FILE_INVALID, ovms::HTTPStatusCode::ERROR},
        {StatusCode::NO_MODEL_VERSION_AVAILABLE, ovms::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_NOT_LOADED, ovms::HTTPStatusCode::ERROR},
        {StatusCode::JSON_INVALID, ovms::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MODELINSTANCE_NOT_FOUND, ovms::HTTPStatusCode::ERROR},
        {StatusCode::SHAPE_WRONG_FORMAT, ovms::HTTPStatusCode::ERROR},
        {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, ovms::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, ovms::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, ovms::HTTPStatusCode::ERROR},
        {StatusCode::RESHAPE_ERROR, ovms::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MODEL_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_NAME_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MEDIAPIPE_EXECUTION_ERROR, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::MEDIAPIPE_PRECONDITION_FAILED, ovms::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, ovms::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_NOT_LOADED_YET, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_SPEC_MISSING, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SIGNATURE_DEF, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS, ovms::HTTPStatusCode::NO_CONTENT},
        {StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE, ovms::HTTPStatusCode::PRECOND_FAILED},

        // Sequence management
        {StatusCode::SEQUENCE_MISSING, ovms::HTTPStatusCode::NOT_FOUND},
        {StatusCode::SEQUENCE_ALREADY_EXISTS, ovms::HTTPStatusCode::CONFLICT},
        {StatusCode::SEQUENCE_ID_NOT_PROVIDED, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_ID_BAD_TYPE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_TERMINATED, ovms::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::MAX_SEQUENCE_NUMBER_REACHED, ovms::HTTPStatusCode::SERVICE_UNAV},

        // Predict request validation
        {StatusCode::INVALID_NO_OF_INPUTS, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_MISSING_INPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_UNEXPECTED_INPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_BATCH_SIZE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SHAPE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_BUFFER_TYPE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_DEVICE_ID, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_STRING_INPUT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_INPUT_FORMAT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_PRECISION, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_VALUE_COUNT, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_CONTENT_SIZE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_MESSAGE_STRUCTURE, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::UNSUPPORTED_LAYOUT, ovms::HTTPStatusCode::BAD_REQUEST},

        // Deserialization

        // Should never occur - ModelInstance::validate takes care of that
        {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, ovms::HTTPStatusCode::ERROR},
        {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, ovms::HTTPStatusCode::ERROR},

        // Inference
        {StatusCode::OV_INTERNAL_INFERENCE_ERROR, ovms::HTTPStatusCode::ERROR},

        // Serialization

        // Should never occur - it should be validated during model loading
        {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, ovms::HTTPStatusCode::ERROR},
        {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, ovms::HTTPStatusCode::ERROR},

        // GetModelStatus
        {StatusCode::INTERNAL_ERROR, ovms::HTTPStatusCode::ERROR},

        // Binary input
        {StatusCode::INVALID_NO_OF_CHANNELS, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH, ovms::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::STRING_VAL_EMPTY, ovms::HTTPStatusCode::BAD_REQUEST},
    };
    auto it = httpStatusMap.find(status.getCode());
    if (it != httpStatusMap.end()) {
        return it->second;
    } else {
        return ovms::HTTPStatusCode::ERROR;
    }
}

std::unique_ptr<DrogonHttpServer> createAndStartDrogonHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, const ovms::Config& config, int timeout_in_ms) {
    auto server = std::make_unique<DrogonHttpServer>(num_threads, num_threads, port, address);
    auto handler = std::make_shared<HttpRestApiHandler>(ovmsServer, timeout_in_ms, config.apiKey());
    auto& pool = server->getPool();
    server->registerRequestDispatcher([handler, &pool](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)> drogonResponseInitializeCallback) {
        SPDLOG_DEBUG("REST request {}", req->getOriginalPath());

        std::unordered_map<std::string, std::string> headers;

        for (const auto& header : req->headers()) {
            headers[header.first] = header.second;
        }

        SPDLOG_DEBUG("Processing HTTP request: {} {} body: {} bytes",
            req->getMethodString(),
            req->getOriginalPath(),
            req->getBody().size());

        auto body = std::string(req->getBody());
        std::string output;
        HttpResponseComponents responseComponents;
        std::shared_ptr<HttpAsyncWriter> writer = std::make_shared<DrogonHttpAsyncWriterImpl>(drogonResponseInitializeCallback, pool, req);
        std::shared_ptr<MultiPartParser> multiPartParser = std::make_shared<DrogonMultiPartParser>(req);

        const auto status = handler->processRequest(
            drogon::to_string_view(req->getMethod()),
            req->getOriginalPath(),
            body,
            &headers,
            &output,
            responseComponents,
            std::move(writer),
            std::move(multiPartParser));
        if (status == StatusCode::PARTIAL_END) {
            // No further messaging is required.
            // Partial responses were delivered via "req" object.
            return;
        }
        if (!status.ok() && output.empty()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> errorWriter(buffer);
            errorWriter.StartObject();
            errorWriter.String("error");
            errorWriter.String(status.string().c_str());
            errorWriter.EndObject();
            output = buffer.GetString();
        }
        auto resp = drogon::HttpResponse::newHttpResponse();

        if (responseComponents.contentType == ContentType::PLAIN_TEXT) {
            resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
        } else {
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        }

        if (responseComponents.inferenceHeaderContentLength.has_value()) {
            resp->addHeader("inference-header-content-length", std::to_string(responseComponents.inferenceHeaderContentLength.value()));
        }

        resp->setBody(output);

        const auto http_status = http(status);

        if (http_status != ovms::HTTPStatusCode::OK && http_status != ovms::HTTPStatusCode::CREATED) {
            SPDLOG_DEBUG("Processing HTTP/REST request failed: {} {}. Reason: {}",
                req->getMethodString(),
                req->getOriginalPath(),
                status.string());
        }

        if (!status.ok()) {
            resp->setStatusCode(drogon::HttpStatusCode(http_status));
        }
        drogonResponseInitializeCallback(resp);
    });
    if (!server->startAcceptingRequests().ok()) {
        SPDLOG_ERROR("Failed to start Drogon server");
        return nullptr;
    }
    if (config.apiKey().empty()) {
        SPDLOG_INFO("API key not provided via --api_key_file or API_KEY environment variable. Authentication will be disabled.");
    }
    return server;
}

}  // namespace ovms
