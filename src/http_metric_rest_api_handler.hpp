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

#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "rest_parser.hpp"
#include "status.hpp"
#include "rest_api_handler_common.hpp"

namespace tensorflow::serving::net_http {
class ServerRequestInterface;
}

namespace ovms {
class Server;
class ModelManager;

class HttpMetricRestApiHandler {
public:

    static const std::string metricsRegexExp;

    /**
     * @brief Construct a new HttpRest Api Handler
     *
     * @param timeout_in_ms
     */
    HttpMetricRestApiHandler(ovms::Server& ovmsServer, int timeout_in_ms);

    void registerAll();

    Status processMetrics(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);

    // TODO: This is common for metric and regular server
    Status processRequest(
        const std::string_view http_method,
        const std::string_view request_path,
        const std::string& request_body,
        std::vector<std::pair<std::string, std::string>>* headers,
        std::string* response,
        HttpResponseComponents& responseComponents,
        tensorflow::serving::net_http::ServerRequestInterface* writer);

    // TODO: This is common for metric and regular server
    Status dispatchToProcessor(
        const std::string_view uri,
        const std::string& request_body,
        std::string* response,
        const HttpRequestComponents& request_components,
        HttpResponseComponents& response_components,
        tensorflow::serving::net_http::ServerRequestInterface* writer);

    // TODO: This is common for metric and regular server
    Status parseRequestComponents(HttpRequestComponents& components,
        const std::string_view http_method,
        const std::string& request_path,
        const std::vector<std::pair<std::string, std::string>>& headers = {});

private:

    const std::regex metricsRegex;

    HandlerCallbackFn handler;
    int timeout_in_ms;

    ovms::Server& ovmsServer;
    ovms::ModelManager& modelManager;
};

}  // namespace ovms
