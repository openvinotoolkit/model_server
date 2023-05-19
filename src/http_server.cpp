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

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"
#pragma GCC diagnostic pop

#include "http_rest_api_handler.hpp"
#include "status.hpp"

namespace ovms {

namespace net_http = tensorflow::serving::net_http;

static const net_http::HTTPStatusCode http(const ovms::Status& status) {
    const std::unordered_map<const StatusCode, net_http::HTTPStatusCode> httpStatusMap = {
        {StatusCode::OK, net_http::HTTPStatusCode::OK},
        {StatusCode::OK_RELOADED, net_http::HTTPStatusCode::CREATED},
        {StatusCode::OK_NOT_RELOADED, net_http::HTTPStatusCode::OK},

        // REST handler failure
        {StatusCode::REST_INVALID_URL, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_UNSUPPORTED_METHOD, net_http::HTTPStatusCode::NONE_ACC},
        {StatusCode::REST_NOT_FOUND, net_http::HTTPStatusCode::NOT_FOUND},

        // REST parser failure
        {StatusCode::REST_BODY_IS_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_PREDICT_UNKNOWN_ORDER, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_NOT_AN_ARRAY, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INPUT_NOT_PREALLOCATED, net_http::HTTPStatusCode::ERROR},
        {StatusCode::REST_NO_INSTANCES_FOUND, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_INSTANCE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_INPUTS_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_NO_INPUTS_FOUND, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_OUTPUT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_COULD_NOT_PARSE_PARAMETERS, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_PROTO_TO_STRING_ERROR, net_http::HTTPStatusCode::ERROR},
        {StatusCode::REST_UNSUPPORTED_PRECISION, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, net_http::HTTPStatusCode::ERROR},
        {StatusCode::REST_BINARY_BUFFER_EXCEEDED, net_http::HTTPStatusCode::BAD_REQUEST},

        {StatusCode::PATH_INVALID, net_http::HTTPStatusCode::ERROR},
        {StatusCode::FILE_INVALID, net_http::HTTPStatusCode::ERROR},
        {StatusCode::NO_MODEL_VERSION_AVAILABLE, net_http::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_NOT_LOADED, net_http::HTTPStatusCode::ERROR},
        {StatusCode::JSON_INVALID, net_http::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MODELINSTANCE_NOT_FOUND, net_http::HTTPStatusCode::ERROR},
        {StatusCode::SHAPE_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
        {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
        {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, net_http::HTTPStatusCode::ERROR},
        {StatusCode::RESHAPE_ERROR, net_http::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::MODEL_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_NAME_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_VERSION_NOT_LOADED_YET, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::MODEL_SPEC_MISSING, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SIGNATURE_DEF, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS, net_http::HTTPStatusCode::NO_CONTENT},
        {StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE, net_http::HTTPStatusCode::PRECOND_FAILED},

        // Sequence management
        {StatusCode::SEQUENCE_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
        {StatusCode::SEQUENCE_ALREADY_EXISTS, net_http::HTTPStatusCode::CONFLICT},
        {StatusCode::SEQUENCE_ID_NOT_PROVIDED, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_ID_BAD_TYPE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::SEQUENCE_TERMINATED, net_http::HTTPStatusCode::PRECOND_FAILED},
        {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::MAX_SEQUENCE_NUMBER_REACHED, net_http::HTTPStatusCode::SERVICE_UNAV},

        // Predict request validation
        {StatusCode::INVALID_NO_OF_INPUTS, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_MISSING_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_BATCH_SIZE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_SHAPE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_BUFFER_TYPE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_DEVICE_ID, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_STRING_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_INPUT_FORMAT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_PRECISION, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_VALUE_COUNT, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_CONTENT_SIZE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::INVALID_MESSAGE_STRUCTURE, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::UNSUPPORTED_LAYOUT, net_http::HTTPStatusCode::BAD_REQUEST},

        // Deserialization

        // Should never occur - ModelInstance::validate takes care of that
        {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, net_http::HTTPStatusCode::ERROR},
        {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, net_http::HTTPStatusCode::ERROR},

        // Inference
        {StatusCode::OV_INTERNAL_INFERENCE_ERROR, net_http::HTTPStatusCode::ERROR},

        // Serialization

        // Should never occur - it should be validated during model loading
        {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, net_http::HTTPStatusCode::ERROR},
        {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, net_http::HTTPStatusCode::ERROR},

        // GetModelStatus
        {StatusCode::INTERNAL_ERROR, net_http::HTTPStatusCode::ERROR},

        // Binary input
        {StatusCode::INVALID_NO_OF_CHANNELS, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH, net_http::HTTPStatusCode::BAD_REQUEST},
        {StatusCode::STRING_VAL_EMPTY, net_http::HTTPStatusCode::BAD_REQUEST},
    };
    auto it = httpStatusMap.find(status.getCode());
    if (it != httpStatusMap.end()) {
        return it->second;
    } else {
        return net_http::HTTPStatusCode::ERROR;
    }
}

class RequestExecutor final : public net_http::EventExecutor {
public:
    explicit RequestExecutor(int num_threads) :
        executor_(tensorflow::Env::Default(), "httprestserver", num_threads) {}

    void Schedule(std::function<void()> fn) override { executor_.Schedule(fn); }

private:
    tensorflow::serving::ThreadPoolExecutor executor_;
};

class RestApiRequestDispatcher {
public:
    RestApiRequestDispatcher(ovms::Server& ovmsServer, int timeout_in_ms) {
        handler_ = std::make_unique<HttpRestApiHandler>(ovmsServer, timeout_in_ms);
    }

    net_http::RequestHandler dispatch(net_http::ServerRequestInterface* req) {
        return [this](net_http::ServerRequestInterface* req) {
            try {
                this->processRequest(req);
            } catch (...) {
                SPDLOG_DEBUG("Exception caught in REST request handler");
                req->ReplyWithStatus(net_http::HTTPStatusCode::ERROR);
            }
        };
    }

private:
    void parseHeaders(const net_http::ServerRequestInterface* req, std::vector<std::pair<std::string, std::string>>* headers) {
        if (req->GetRequestHeader("Inference-Header-Content-Length").size() > 0) {
            std::pair<std::string, std::string> header{"Inference-Header-Content-Length", req->GetRequestHeader("Inference-Header-Content-Length")};
            headers->emplace_back(header);
        }
    }
    void processRequest(net_http::ServerRequestInterface* req) {
        SPDLOG_DEBUG("REST request {}", req->uri_path());
        std::string body;
        int64_t num_bytes = 0;
        auto request_chunk = req->ReadRequestBytes(&num_bytes);
        while (request_chunk != nullptr) {
            body.append(std::string_view(request_chunk.get(), num_bytes));
            request_chunk = req->ReadRequestBytes(&num_bytes);
        }

        std::vector<std::pair<std::string, std::string>> headers;
        parseHeaders(req, &headers);
        std::string output;
        SPDLOG_DEBUG("Processing HTTP request: {} {} body: {} bytes",
            req->http_method(),
            req->uri_path(),
            body.size());
        HttpResponseComponents responseComponents;
        const auto status = handler_->processRequest(req->http_method(), req->uri_path(), body, &headers, &output, responseComponents);
        if (!status.ok() && output.empty()) {
            output.append("{\"error\": \"" + status.string() + "\"}");
        }
        const auto http_status = http(status);
        if (responseComponents.inferenceHeaderContentLength.has_value()) {
            std::pair<std::string, std::string> header{"Inference-Header-Content-Length", std::to_string(responseComponents.inferenceHeaderContentLength.value())};
            headers.emplace_back(header);
        }
        for (const auto& kv : headers) {
            req->OverwriteResponseHeader(kv.first, kv.second);
        }
        req->WriteResponseString(output);
        if (http_status != net_http::HTTPStatusCode::OK && http_status != net_http::HTTPStatusCode::CREATED) {
            SPDLOG_DEBUG("Processing HTTP/REST request failed: {} {}. Reason: {}",
                req->http_method(),
                req->uri_path(),
                status.string());
        }
        req->ReplyWithStatus(http_status);
    }

    std::unique_ptr<HttpRestApiHandler> handler_;
};

std::unique_ptr<http_server> createAndStartHttpServer(const std::string& address, int port, int num_threads, ovms::Server& ovmsServer, int timeout_in_ms) {
    auto options = std::make_unique<net_http::ServerOptions>();
    options->AddPort(static_cast<uint32_t>(port));
    options->SetAddress(address);
    options->SetExecutor(std::make_unique<RequestExecutor>(num_threads));

    auto server = net_http::CreateEvHTTPServer(std::move(options));
    if (server == nullptr) {
        SPDLOG_ERROR("Failed to create http server");
        return nullptr;
    }

    std::shared_ptr<RestApiRequestDispatcher> dispatcher =
        std::make_shared<RestApiRequestDispatcher>(ovmsServer, timeout_in_ms);

    net_http::RequestHandlerOptions handler_options;
    server->RegisterRequestDispatcher(
        [dispatcher](net_http::ServerRequestInterface* req) {
            return dispatcher->dispatch(req);
        },
        handler_options);

    if (server->StartAcceptingRequests()) {
        SPDLOG_INFO("REST server listening on port {} with {} threads", port, num_threads);
        return server;
    }

    return nullptr;
}
}  // namespace ovms
