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
#include "http_rest_api_handler.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "filesystem.hpp"
#include "get_model_metadata_impl.hpp"
#include "model_service.hpp"
#include "modelinstanceunloadguard.hpp"
#include "prediction_service_utils.hpp"
#include "rest_parser.hpp"
#include "rest_utils.hpp"

#define DEBUG
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

const std::string HttpRestApiHandler::kPathRegexExp = R"((.?)\/v1\/models\/.*)";
const std::string HttpRestApiHandler::predictionRegexExp =
    R"((.?)\/v1\/models\/([^\/:]+)(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?:(classify|regress|predict))";
const std::string HttpRestApiHandler::modelstatusRegexExp =
    R"((.?)\/v1\/models(?:\/([^\/:]+))?(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?(?:\/(metadata))?)";

Status HttpRestApiHandler::validateUrlAndMethod(
    const std::string_view http_method,
    const std::string& request_path,
    std::smatch* sm) {

    if (http_method != "POST" && http_method != "GET") {
        return StatusCode::REST_UNSUPPORTED_METHOD;
    }

    if (FileSystem::isPathEscaped(request_path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    if (!std::regex_match(request_path, *sm, sanityRegex)) {
        return StatusCode::REST_INVALID_URL;
    }

    if (http_method == "POST") {
        if (std::regex_match(request_path, *sm, predictionRegex)) {
            return StatusCode::OK;
        } else if (std::regex_match(request_path, *sm, modelstatusRegex)) {
            return StatusCode::REST_UNSUPPORTED_METHOD;
        }
    } else if (http_method == "GET") {
        if (std::regex_match(request_path, *sm, modelstatusRegex)) {
            return StatusCode::OK;
        } else if (std::regex_match(request_path, *sm, predictionRegex)) {
            return StatusCode::REST_UNSUPPORTED_METHOD;
        }
    }
    return StatusCode::REST_INVALID_URL;
}

Status HttpRestApiHandler::parseModelVersion(std::string& model_version_str, std::optional<int64_t>& model_version) {
    if (!model_version_str.empty()) {
        try {
            model_version = std::stoll(model_version_str.c_str());
        } catch (std::exception& e) {
            SPDLOG_ERROR("Couldn't parse model version {}", model_version_str);
            return StatusCode::REST_COULD_NOT_PARSE_VERSION;
        }
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::dispatchToProcessor(
    const std::string_view request_path,
    const std::string& request_body,
    std::string* response,
    const HttpRequestComponents& request_components) {

    if (FileSystem::isPathEscaped({request_path.begin(), request_path.end()})) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    if (request_components.http_method == "POST") {
        if (request_components.processing_method == "predict") {
            return processPredictRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, request_body, response);
        } else {
            SPDLOG_WARN("Requested REST resource {} not found", std::string(request_path));
            return StatusCode::REST_NOT_FOUND;
        }
    } else if (request_components.http_method == "GET") {
        if (!request_components.model_subresource.empty() && request_components.model_subresource == "metadata") {
            return processModelMetadataRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, response);
        } else {
            return processModelStatusRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, response);
        }
    }
    return StatusCode::UNKNOWN_ERROR;
}

Status HttpRestApiHandler::processRequest(
    const std::string_view http_method,
    const std::string_view request_path,
    const std::string& request_body,
    std::vector<std::pair<std::string, std::string>>* headers,
    std::string* response) {

    std::smatch sm;
    std::string request_path_str(request_path);
    if (FileSystem::isPathEscaped(request_path_str)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    auto status = validateUrlAndMethod(http_method, request_path_str, &sm);
    if (!status.ok()) {
        return status;
    }

    headers->clear();
    response->clear();
    headers->push_back({"Content-Type", "application/json"});

    HttpRequestComponents requestComponents;
    requestComponents.http_method = http_method;

    requestComponents.model_name = sm[2];
    std::string model_version_str = sm[3];
    std::string model_version_label_str = sm[4];
    if (requestComponents.http_method == "POST")
        requestComponents.processing_method = sm[5];
    else
        requestComponents.model_subresource = sm[5];

    status = parseModelVersion(model_version_str, requestComponents.model_version);
    if (!status.ok())
        return status;

    if (!model_version_label_str.empty()) {
        requestComponents.model_version_label = model_version_label_str;
    }
    return dispatchToProcessor(request_path, request_body, response, requestComponents);
}

Status HttpRestApiHandler::processPredictRequest(
    const std::string& modelName,
    const std::optional<int64_t>& modelVersion,
    const std::optional<std::string_view>& modelVersionLabel,
    const std::string& request,
    std::string* response) {
    // model_version_label currently is not in use

    Timer timer;
    timer.start("total");
    using std::chrono::microseconds;

    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}",
        modelName, modelVersion.value_or(0));

    ModelManager& modelManager = ModelManager::getInstance();
    Order requestOrder;
    tensorflow::serving::PredictResponse responseProto;
    Status status;

    if (modelManager.modelExists(modelName)) {
        SPDLOG_DEBUG("Found model with name: {}. Searching for requested version...", modelName);
        status = processSingleModelRequest(modelName, modelVersion, request, requestOrder, responseProto);
    } else if (modelManager.pipelineDefinitionExists(modelName)) {
        SPDLOG_DEBUG("Found pipeline with name: {}", modelName);
        status = processPipelineRequest(modelName, request, requestOrder, responseProto);
    } else {
        SPDLOG_WARN("Model or pipeline matching request parameters not found - name: {}, version: {}", modelName, modelVersion.value_or(0));
        status = StatusCode::MODEL_NAME_MISSING;
    }
    if (!status.ok())
        return status;

    status = makeJsonFromPredictResponse(responseProto, response, requestOrder);
    if (!status.ok())
        return status;

    timer.stop("total");
    SPDLOG_DEBUG("Total REST request processing time: {} ms", timer.elapsed<std::chrono::microseconds>("total") / 1000);
    return StatusCode::OK;
}

Status HttpRestApiHandler::processSingleModelRequest(const std::string& modelName,
    const std::optional<int64_t>& modelVersion,
    const std::string& request,
    Order& requestOrder,
    tensorflow::serving::PredictResponse& responseProto) {

    std::shared_ptr<ModelInstance> modelInstance;
    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(
        ModelManager::getInstance(),
        modelName,
        modelVersion.value_or(0),
        modelInstance,
        modelInstanceUnloadGuard);

    if (!status.ok()) {
        SPDLOG_WARN("Requested model instance - name: {}, version: {} - does not exist.", modelName, modelVersion.value_or(0));
        return status;
    }
    Timer timer;
    timer.start("parse");
    RestParser requestParser(modelInstance->getInputsInfo());
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop("parse");
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>("parse") / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    if (modelVersion.has_value()) {
        requestProto.mutable_model_spec()->mutable_version()->set_value(modelVersion.value());
    }
    status = inference(*modelInstance, &requestProto, &responseProto, modelInstanceUnloadGuard);
    return status;
}

Status HttpRestApiHandler::processPipelineRequest(const std::string& modelName,
    const std::string& request,
    Order& requestOrder,
    tensorflow::serving::PredictResponse& responseProto) {

    std::unique_ptr<Pipeline> pipelinePtr;

    Timer timer;
    timer.start("parse");
    RestParser requestParser;
    auto status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop("parse");
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>("parse") / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    status = getPipeline(ModelManager::getInstance(), pipelinePtr, &requestProto, &responseProto);
    if (!status.ok()) {
        return status;
    }
    status = pipelinePtr->execute();
    return status;
}

Status HttpRestApiHandler::processModelMetadataRequest(
    const std::string_view model_name,
    const std::optional<int64_t>& model_version,
    const std::optional<std::string_view>& model_version_label,
    std::string* response) {
    // model_version_label currently is not in use
    tensorflow::serving::GetModelMetadataRequest grpc_request;
    tensorflow::serving::GetModelMetadataResponse grpc_response;
    Status status;
    std::string modelName(model_name);
    status = GetModelMetadataImpl::createGrpcRequest(modelName, model_version, &grpc_request);
    if (!status.ok()) {
        return status;
    }
    status = GetModelMetadataImpl::getModelStatus(&grpc_request, &grpc_response);
    if (!status.ok()) {
        return status;
    }
    status = GetModelMetadataImpl::serializeResponse2Json(&grpc_response, response);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::processModelStatusRequest(
    const std::string_view model_name,
    const std::optional<int64_t>& model_version,
    const std::optional<std::string_view>& model_version_label,
    std::string* response) {
    // model_version_label currently is not in use
    SPDLOG_DEBUG("Processing model status request");
    tensorflow::serving::GetModelStatusRequest grpc_request;
    tensorflow::serving::GetModelStatusResponse grpc_response;
    Status status;
    std::string modelName(model_name);
    status = GetModelStatusImpl::createGrpcRequest(modelName, model_version, &grpc_request);
    if (!status.ok()) {
        return status;
    }
    status = GetModelStatusImpl::getModelStatus(&grpc_request, &grpc_response, ModelManager::getInstance());
    if (!status.ok()) {
        return status;
    }
    status = GetModelStatusImpl::serializeResponse2Json(&grpc_response, response);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}

}  // namespace ovms
