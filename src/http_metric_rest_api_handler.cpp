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
#include "http_metric_rest_api_handler.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#pragma GCC diagnostic pop

#include "config.hpp"
#include "dags/pipeline.hpp"
#include "dags/pipelinedefinition.hpp"
#include "dags/pipelinedefinitionunloadguard.hpp"
#include "execution_context.hpp"
#include "filesystem.hpp"
#include "get_model_metadata_impl.hpp"
#include "grpcservermodule.hpp"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "kfs_frontend/kfs_utils.hpp"
#include "metric_module.hpp"
#include "metric_registry.hpp"
#include "model_metric_reporter.hpp"
#include "model_service.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "prediction_service_utils.hpp"
#include "profiler.hpp"
#include "rest_parser.hpp"
#include "rest_utils.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "status.hpp"
#include "stringutils.hpp"
#include "timer.hpp"


namespace ovms {
const std::string HttpMetricRestApiHandler::metricsRegexExp = R"((.?)\/metrics(\?(.*))?)";

HttpMetricRestApiHandler::HttpMetricRestApiHandler(ovms::Server& ovmsServer, int timeout_in_ms) :
    metricsRegex(metricsRegexExp),
    timeout_in_ms(timeout_in_ms),
    ovmsServer(ovmsServer),
    modelManager(dynamic_cast<const ServableManagerModule*>(this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))
        throw std::logic_error("Tried to create metric rest api handler without grpc server module");
    if (nullptr == this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))
        throw std::logic_error("Tried to create metric rest api handler without servable manager module");
    registerAll();
}

void HttpMetricRestApiHandler::registerAll() {
    handler = [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, tensorflow::serving::net_http::ServerRequestInterface*) -> Status {
        return processMetrics(request_components, response, request_body);
    };
}

Status HttpMetricRestApiHandler::processMetrics(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    auto module = this->ovmsServer.getModule(METRICS_MODULE_NAME);
    if (nullptr == module) {
        SPDLOG_ERROR("Failed to process metrics - metrics module is missing");
        return StatusCode::INTERNAL_ERROR;
    }
    auto& metricConfig = this->modelManager.getMetricConfig();

    if (!metricConfig.metricsPort) {
        return StatusCode::REST_INVALID_URL;
    }

    auto metricModule = dynamic_cast<const MetricModule*>(module);
    response = metricModule->getRegistry().collect();

    return StatusCode::OK;
}

Status HttpMetricRestApiHandler::processRequest(
    const std::string_view http_method,
    const std::string_view request_path,
    const std::string& request_body,
    std::vector<std::pair<std::string, std::string>>* headers,
    std::string* response,
    HttpResponseComponents& responseComponents,
    tensorflow::serving::net_http::ServerRequestInterface* serverReaderWriter) {

    std::smatch sm;
    std::string request_path_str(request_path);
    if (FileSystem::isPathEscaped(request_path_str)) {
        SPDLOG_DEBUG("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    HttpRequestComponents requestComponents;
    auto status = parseRequestComponents(requestComponents, http_method, request_path_str, *headers);

    headers->clear();
    response->clear();
    headers->push_back({"Content-Type", "application/json"});

    if (!status.ok())
        return status;
    return dispatchToProcessor(request_path, request_body, response, requestComponents, responseComponents, serverReaderWriter);
}

Status HttpMetricRestApiHandler::parseRequestComponents(HttpRequestComponents& requestComponents,
    const std::string_view http_method,
    const std::string& request_path,
    const std::vector<std::pair<std::string, std::string>>& headers) {
    std::smatch sm;
    requestComponents.http_method = http_method;
    if (http_method != "POST" && http_method != "GET") {
        return StatusCode::REST_UNSUPPORTED_METHOD;
    }

    if (FileSystem::isPathEscaped(request_path)) {
        SPDLOG_DEBUG("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }
    if (http_method == "GET") {
        if (std::regex_match(request_path, sm, metricsRegex)) {
            std::string params = sm[3];
            if (!params.empty()) {
                SPDLOG_DEBUG("Discarded following url parameters: {}", params);
            }
            requestComponents.type = Metrics;
            return StatusCode::OK;
        }
    }
    return StatusCode::REST_INVALID_URL;
}

Status HttpMetricRestApiHandler::dispatchToProcessor(
    const std::string_view uri,
    const std::string& request_body,
    std::string* response,
    const HttpRequestComponents& request_components,
    HttpResponseComponents& response_components,
    tensorflow::serving::net_http::ServerRequestInterface* serverReaderWriter) {

    if (handler && request_components.type == Metrics)
        return handler(uri, request_components, *response, request_body, response_components, serverReaderWriter);
    return StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE;
}

}  // namespace ovms
