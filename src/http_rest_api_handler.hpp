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
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "http_async_writer_interface.hpp"
#include "rest_parser.hpp"
#include "status.hpp"

namespace ovms {
class ServableMetricReporter;
class KFSInferenceServiceImpl;
class GetModelMetadataImpl;
class Server;
class ModelManager;

enum RequestType { Predict,
    GetModelStatus,
    GetModelMetadata,
    ConfigReload,
    ConfigStatus,
    KFS_GetModelReady,
    KFS_Infer,
    KFS_GetModelMetadata,
    KFS_GetServerReady,
    KFS_GetServerLive,
    KFS_GetServerMetadata,
    V3,
    Metrics };

struct HttpRequestComponents {
    RequestType type;
    std::string_view http_method;
    std::string model_name;
    std::optional<int64_t> model_version;
    std::optional<std::string_view> model_version_label;
    std::string processing_method;
    std::string model_subresource;
    std::optional<int> inferenceHeaderContentLength;
    std::vector<std::pair<std::string, std::string>> headers;
};

struct HttpResponseComponents {
    std::optional<int> inferenceHeaderContentLength;
};

using HandlerCallbackFn = std::function<Status(const std::string_view, const HttpRequestComponents&, std::string&, const std::string&, HttpResponseComponents&, std::shared_ptr<HttpAsyncWriter>)>;

std::string urlDecode(const std::string& encoded);

class HttpRestApiHandler {
public:
    static const std::string predictionRegexExp;
    static const std::string modelstatusRegexExp;
    static const std::string configReloadRegexExp;
    static const std::string configStatusRegexExp;

    static const std::string kfs_modelreadyRegexExp;
    static const std::string kfs_modelmetadataRegexExp;
    static const std::string kfs_inferRegexExp;

    static const std::string metricsRegexExp;

    static const std::string kfs_serverreadyRegexExp;
    static const std::string kfs_serverliveRegexExp;
    static const std::string kfs_servermetadataRegexExp;

    static const std::string v3_RegexExp;
    /**
     * @brief Construct a new HttpRest Api Handler
     *
     * @param timeout_in_ms
     */
    HttpRestApiHandler(ovms::Server& ovmsServer, int timeout_in_ms);

    Status parseRequestComponents(HttpRequestComponents& components,
        const std::string_view http_method,
        const std::string& request_path,
        const std::vector<std::pair<std::string, std::string>>& headers = {});

    Status parseModelVersion(std::string& model_version_str, std::optional<int64_t>& model_version);
    static Status prepareGrpcRequest(const std::string modelName, const std::optional<int64_t>& modelVersion, const std::string& request_body, ::KFSRequest& grpc_request, const std::optional<int>& inferenceHeaderContentLength = {});

    void registerHandler(RequestType type, HandlerCallbackFn);
    void registerAll();

    Status dispatchToProcessor(
        const std::string_view uri,
        const std::string& request_body,
        std::string* response,
        const HttpRequestComponents& request_components,
        HttpResponseComponents& response_components,
        std::shared_ptr<HttpAsyncWriter> writer);

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
        std::string* response,
        HttpResponseComponents& responseComponents,
        std::shared_ptr<HttpAsyncWriter> writer);

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
        tensorflow::serving::PredictResponse& responseProto,
        ServableMetricReporter*& reporterOut);

    Status processPipelineRequest(
        const std::string& modelName,
        const std::string& request,
        Order& requestOrder,
        tensorflow::serving::PredictResponse& responseProto,
        ServableMetricReporter*& reporterOut);

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

    void convertShapeType(rapidjson::Value& scope, rapidjson::Document& doc);
    void convertRTInfo(rapidjson::Value& scope, rapidjson::Document& doc, ov::AnyMap& rt_info);

    Status processConfigStatusRequest(std::string& response, ModelManager& manager);
    Status processModelMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);
    Status processModelReadyKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);
    Status processInferKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, std::optional<int>& inferenceHeaderContentLength);
    Status processMetrics(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);

    Status processServerReadyKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);
    Status processServerLiveKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);
    Status processServerMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body);

    Status processV3(const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, std::shared_ptr<HttpAsyncWriter>& serverReaderWriter);

private:
    const std::regex predictionRegex;
    const std::regex modelstatusRegex;
    const std::regex configReloadRegex;
    const std::regex configStatusRegex;

    const std::regex kfs_modelreadyRegex;
    const std::regex kfs_modelmetadataRegex;

    const std::regex kfs_inferRegex;
    const std::regex kfs_serverreadyRegex;
    const std::regex kfs_serverliveRegex;
    const std::regex kfs_servermetadataRegex;

    const std::regex v3_Regex;

    const std::regex metricsRegex;

    std::map<RequestType, HandlerCallbackFn> handlers;
    int timeout_in_ms;

    ovms::Server& ovmsServer;
    ovms::KFSInferenceServiceImpl& kfsGrpcImpl;
    const GetModelMetadataImpl& grpcGetModelMetadataImpl;
    ovms::ModelManager& modelManager;

    Status getReporter(const HttpRequestComponents& components, ovms::ServableMetricReporter*& reporter);
    Status getPipelineInputsAndReporter(const std::string& modelName, ovms::tensor_map_t& inputs, ovms::ServableMetricReporter*& reporter);
};

}  // namespace ovms
