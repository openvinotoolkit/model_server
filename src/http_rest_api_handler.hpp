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

#include <regex>
#include <string>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "modelmanager.hpp"
#include "rest_parser.hpp"
#include "status.hpp"

namespace ovms {
enum RequestType { Predict,
    GetModelStatus,
    GetModelMetadata,
    ConfigReload,
    ConfigStatus };
struct HttpRequestComponents {
    RequestType type;
    std::string_view http_method;
    std::string model_name;
    std::optional<int64_t> model_version;
    std::optional<std::string_view> model_version_label;
    std::string processing_method;
    std::string model_subresource;
};

class HttpRestApiHandler {
public:
    static const std::string predictionRegexExp;
    static const std::string modelstatusRegexExp;
    static const std::string configReloadRegexExp;
    static const std::string configStatusRegexExp;

    /**
     * @brief Construct a new HttpRest Api Handler
     * 
     * @param timeout_in_ms 
     */
    HttpRestApiHandler(int timeout_in_ms) :
        predictionRegex(predictionRegexExp),
        modelstatusRegex(modelstatusRegexExp),
        configReloadRegex(configReloadRegexExp),
        configStatusRegex(configStatusRegexExp),
        timeout_in_ms(timeout_in_ms) {}

    Status parseRequestComponents(HttpRequestComponents& components,
        const std::string_view http_method,
        const std::string& request_path);

    Status parseModelVersion(std::string& model_version_str, std::optional<int64_t>& model_version);

    Status dispatchToProcessor(
        const std::string& request_body,
        std::string* response,
        const HttpRequestComponents& request_components);

    /**
     * @brief Process Request
     * 
     * @param http_method 
     * @param request_path 
     * @param request_body 
     * @param headers 
     * @param resposnse 
     *
     * @return StatusCode 
     */
    Status processRequest(
        const std::string_view http_method,
        const std::string_view request_path,
        const std::string& request_body,
        std::vector<std::pair<std::string, std::string>>* headers,
        std::string* response);

    /**
     * @brief Process predict request
     *
     * @param modelName 
     * @param modelVersion 
     * @param modelVersionLabel 
     * @param request 
     * @param response 
     *
     * @return StatusCode 
     */
    Status processPredictRequest(
        const std::string& modelName,
        const std::optional<int64_t>& modelVersion,
        const std::optional<std::string_view>& modelVersionLabel,
        const std::string& request,
        std::string* response);

    Status processSingleModelRequest(
        const std::string& modelName,
        const std::optional<int64_t>& modelVersion,
        const std::string& request,
        Order& requestOrder,
        tensorflow::serving::PredictResponse& responseProto);

    Status processPipelineRequest(
        const std::string& modelName,
        const std::string& request,
        Order& requestOrder,
        tensorflow::serving::PredictResponse& responseProto);

    /**
     * @brief Process Model Metadata request
     * 
     * @param model_name 
     * @param model_version 
     * @param model_version_label 
     * @param response
     *
     * @return StatusCode 
     */
    Status processModelMetadataRequest(
        const std::string_view model_name,
        const std::optional<int64_t>& model_version,
        const std::optional<std::string_view>& model_version_label,
        std::string* response);

    /**
     * @brief Process Model Status request
     * 
     * @param model_name 
     * @param model_version 
     * @param model_version_label 
     * @param response 
     * @return StatusCode 
     */
    Status processModelStatusRequest(
        const std::string_view model_name,
        const std::optional<int64_t>& model_version,
        const std::optional<std::string_view>& model_version_label,
        std::string* response);

    Status processConfigReloadRequest(std::string& response, ModelManager& manager);

    Status processConfigStatusRequest(std::string& response, ModelManager& manager);

private:
    const std::regex predictionRegex;
    const std::regex modelstatusRegex;
    const std::regex configReloadRegex;
    const std::regex configStatusRegex;

    int timeout_in_ms;
};

}  // namespace ovms
