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
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include "model_service.hpp"
#include "http_rest_api_handler.hpp"

namespace ovms {

const std::string HttpRestApiHandler::kPathRegex = "(.?)\\/v1\\/.*";
const std::string HttpRestApiHandler::predictionRegexExp =
    R"((.?)\/v1\/models\/([^\/:]+)(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?:(classify|regress|predict))";
const std::string HttpRestApiHandler::modelstatusRegexExp =
    R"((.?)\/v1\/models(?:\/([^\/:]+))?(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?(?:\/(metadata))?)";

Status HttpRestApiHandler::processRequest(
    const std::string_view http_method,
    const std::string_view request_path,
    const std::string_view request_body,
    std::vector<std::pair<std::string, std::string>>* headers,
    std::string* response) {

    headers->clear();
    response->clear();
    headers->push_back({"Content-Type", "application/json"});
    std::string model_name;
    std::string model_version_str;
    std::string model_version_label_str;
    std::string method;
    std::string model_subresource;
    Status status = StatusCode::REST_MALFORMED_REQUEST;

    // Parse request parameters
    bool parse_successful = false;
    std::smatch sm;
    std::string path(request_path);
    if (http_method == "POST") {
        if (std::regex_match(path, sm, predictionRegex)) {
            parse_successful = true;
            model_name = sm[2];
            model_version_str = sm[3];
            model_version_label_str = sm[4];
            method = sm[5];
        }
    } else if (http_method == "GET") {
        if (std::regex_match(path, sm, modelstatusRegex)) {
            parse_successful = true;
            model_name = sm[2];
            model_version_str = sm[3];
            model_version_label_str = sm[4];
            model_subresource = sm[5];
        }
    }

    std::optional<int64_t> model_version;
    std::optional<std::string_view> model_version_label;
    if (!model_version_str.empty()) {
        int64_t version;
        try {
            version = std::atol(model_version_str.c_str());
        }
        catch (std::exception& e) {
            spdlog::error("Couldn't parse model version {}", model_version_str);
            return StatusCode::REST_COULD_NOT_PARSE_VERSION;
        }
        model_version = version;
    }
    if (!model_version_label_str.empty()) {
        model_version_label = model_version_label_str;
    }

    // Dispatch request to appropriate processor
    if (http_method == "POST" && parse_successful) {
        if (method == "predict") {
            status = processPredictRequest(model_name, model_version, model_version_label, request_body, response);
        } else {
            spdlog::error("Requested REST resource {} not found", path);
            return StatusCode::REST_NOT_FOUND;
        }
    } else if (http_method == "GET" && parse_successful) {
        if (!model_subresource.empty() && model_subresource == "metadata") {
            status = processModelMetadataRequest(model_name, model_version, model_version_label, response);
        } else {
            status = processModelStatusRequest(model_name, model_version, model_version_label, response);
        }
    }

    return status;
}


Status HttpRestApiHandler::processPredictRequest(
    const std::string_view model_name,
    const std::optional<int64_t>& model_version,
    const std::optional<std::string_view>& model_version_label,
    const std::string_view request,
    std::string* response) {
    // model_version_label currently is not in use
    spdlog::debug("Processing predict request");

    response->append("{}");

    return StatusCode::OK;
}

Status HttpRestApiHandler::processModelMetadataRequest(
    const std::string_view model_name,
    const std::optional<int64_t>& model_version,
    const std::optional<std::string_view>& model_version_label,
    std::string* response) {
    // model_version_label currently is not in use
    spdlog::debug("Processing metadata request");
    response->append("{}");

    return StatusCode::OK;
}

Status HttpRestApiHandler::processModelStatusRequest(
    const std::string_view model_name,
    const std::optional<int64_t>& model_version,
    const std::optional<std::string_view>& model_version_label,
    std::string* response) {
    // model_version_label currently is not in use
    spdlog::debug("Processing model status request");
    tensorflow::serving::GetModelStatusRequest grpc_request;
    tensorflow::serving::GetModelStatusResponse grpc_response;
    Status status;
    std::string modelName(model_name);
    status = GetModelStatusImpl::createGrpcRequest(modelName, model_version, &grpc_request);
    if (!status.ok()) {
        response->append("{\"error\": \"" + status.string() + "\"}");
        return status;
    }
    status = GetModelStatusImpl::getModelStatus(&grpc_request, &grpc_response);
    if (!status.ok()) {
        response->append("{\"error\": \"" + status.string() + "\"}");
        return status;
    }
    status = GetModelStatusImpl::serializeResponse2Json(&grpc_response, response);
    if (!status.ok()) {
        response->append("{\"error\": \"" + status.string() + "\"}");
        return status;
    }
    return StatusCode::OK;
}

}  // namespace ovms
