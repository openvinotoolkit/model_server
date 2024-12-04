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

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "http_rest_api_handler.hpp"
#include "http_status_code.hpp"
#include "status.hpp"

#if (USE_DROGON == 0)
#include "cpphttplib_http_async_writer_impl.hpp"
#include "httplib.h"  // NOLINT
#else
#include <drogon/drogon.h>

#include "drogon_http_async_writer_impl.hpp"
#endif

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

#if (USE_DROGON == 1)
std::unique_ptr<DrogonHttpServer> createAndStartDrogonHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, int timeout_in_ms) {
    auto server = std::make_unique<DrogonHttpServer>(num_threads, port, address);
    auto handler = std::make_shared<HttpRestApiHandler>(ovmsServer, timeout_in_ms);
    auto& pool = server->getPool();
    server->registerRequestDispatcher([handler, &pool](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        SPDLOG_DEBUG("REST request {}", req->getPath());

        std::vector<std::pair<std::string, std::string>> headers;

        for (const std::pair<const std::string, const std::string>& header : req->headers()) {
            headers.emplace_back(header.first, header.second);
        }

        SPDLOG_DEBUG("Processing HTTP request: {} {} body: {} bytes",
            req->getMethodString(),
            req->getPath(),
            req->getBody().size());

        auto body = std::string(req->getBody());
        std::string output;
        HttpResponseComponents responseComponents;
        std::shared_ptr<HttpAsyncWriter> writer = std::make_shared<DrogonHttpAsyncWriterImpl>(callback, pool);

        const auto status = handler->processRequest(
            drogon::to_string_view(req->getMethod()),
            req->getPath(),
            body,
            &headers,
            &output,
            responseComponents,
            writer);
        if (status == StatusCode::PARTIAL_END) {
            // No further messaging is required.
            // Partial responses were delivered via "req" object.
            return;
        }
        if (!status.ok() && output.empty()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.StartObject();
            writer.String("error");
            writer.String(status.string().c_str());
            writer.EndObject();
            output = buffer.GetString();
        }
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);

        if (responseComponents.inferenceHeaderContentLength.has_value()) {
            std::pair<std::string, std::string> header{"Inference-Header-Content-Length", std::to_string(responseComponents.inferenceHeaderContentLength.value())};
            headers.emplace_back(header);
        }
        for (const auto& [key, value] : headers) {
            resp->addHeader(key, value);
        }
        resp->setBody(output);

        const auto http_status = http(status);

        if (http_status != ovms::HTTPStatusCode::OK && http_status != ovms::HTTPStatusCode::CREATED) {
            SPDLOG_DEBUG("Processing HTTP/REST request failed: {} {}. Reason: {}",
                req->getMethodString(),
                req->getPath(),
                status.string());
        }

        if (!status.ok()) {
            resp->setStatusCode(drogon::HttpStatusCode(http_status));
        }
        callback(resp);
    });
    if (!server->startAcceptingRequests().ok()) {
        SPDLOG_ERROR("Failed to start Drogon server");
        return nullptr;
    }
    return server;
}

#else

std::unique_ptr<CppHttpLibHttpServer> createAndStartCppHttpLibHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, int timeout_in_ms) {
    auto server = std::make_unique<CppHttpLibHttpServer>(num_threads, port, address);
    auto& pool = server->getPool();
    auto handler = std::make_shared<HttpRestApiHandler>(ovmsServer, timeout_in_ms);
    server->registerRequestDispatcher([handler, &pool](const httplib::Request& req, httplib::Response& resp) {
        SPDLOG_DEBUG("REST request {}", req.path);

        std::vector<std::pair<std::string, std::string>> headers;

        for (const std::pair<const std::string, const std::string>& header : req.headers) {
            headers.emplace_back(header.first, header.second);
        }

        SPDLOG_DEBUG("Processing HTTP request: {} {} body: {} bytes",
            req.method,
            req.path,
            req.body.size());

        std::string output;
        HttpResponseComponents responseComponents;
        std::shared_ptr<HttpAsyncWriter> writer = std::make_shared<CppHttpLibHttpAsyncWriterImpl>(resp, pool);

        const auto status = handler->processRequest(
            req.method,
            req.path,
            req.body,
            &headers,
            &output,
            responseComponents,
            writer);
        if (status == StatusCode::PARTIAL_END) {
            // No further messaging is required.
            // Partial responses were delivered via "req" object.
            return;
        }
        if (!status.ok() && output.empty()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.StartObject();
            writer.String("error");
            writer.String(status.string().c_str());
            writer.EndObject();
            output = buffer.GetString();
        }

        if (responseComponents.inferenceHeaderContentLength.has_value()) {
            std::pair<std::string, std::string> header{"Inference-Header-Content-Length", std::to_string(responseComponents.inferenceHeaderContentLength.value())};
            headers.emplace_back(header);
        }
        for (const auto& kv : headers) {
            resp.set_header(kv.first, kv.second);
        }
        resp.set_content(output, "application/json");

        const auto http_status = http(status);

        if (http_status != ovms::HTTPStatusCode::OK && http_status != ovms::HTTPStatusCode::CREATED) {
            SPDLOG_DEBUG("Processing HTTP/REST request failed: {} {}. Reason: {}",
                req.method,
                req.path,
                status.string());
        }

        if (!status.ok()) {
            resp.status = int(http_status);
        }
    });

    server->startAcceptingRequests();
    return server;
}

#endif

}  // namespace ovms
