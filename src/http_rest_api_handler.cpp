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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "config.hpp"
#include "filesystem.hpp"
#include "get_model_metadata_impl.hpp"
#include "grpcservermodule.hpp"
#include "kfs_grpc_inference_service.hpp"
#include "model_service.hpp"
#include "modelinstanceunloadguard.hpp"
#include "pipelinedefinition.hpp"
#include "prediction_service_utils.hpp"
#include "rest_parser.hpp"
#include "rest_utils.hpp"
#include "server.hpp"
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

const std::string HttpRestApiHandler::predictionRegexExp =
    R"((.?)\/v1\/models\/([^\/:]+)(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?:(classify|regress|predict))";
const std::string HttpRestApiHandler::modelstatusRegexExp =
    R"((.?)\/v1\/models(?:\/([^\/:]+))?(?:(?:\/versions\/(\d+))|(?:\/labels\/(\w+)))?(?:\/(metadata))?)";
const std::string HttpRestApiHandler::configReloadRegexExp = R"((.?)\/v1\/config\/reload)";
const std::string HttpRestApiHandler::configStatusRegexExp = R"((.?)\/v1\/config)";

const std::string HttpRestApiHandler::kfs_modelreadyRegexExp =
    R"(/v2/models/([^/]+)(?:/versions/([0-9]+))?(?:/(ready)))";
const std::string HttpRestApiHandler::kfs_modelmetadataRegexExp =
    R"(/v2/models/([^/]+)(?:/versions/([0-9]+))?(?:/)?)";
HttpRestApiHandler::HttpRestApiHandler(ovms::Server& ovmsServer, int timeout_in_ms) :
    predictionRegex(predictionRegexExp),
    modelstatusRegex(modelstatusRegexExp),
    configReloadRegex(configReloadRegexExp),
    configStatusRegex(configStatusRegexExp),
    kfs_modelreadyRegex(kfs_modelreadyRegexExp),
    kfs_modelmetadataRegex(kfs_modelmetadataRegexExp),
    timeout_in_ms(timeout_in_ms),
    ovmsServer(ovmsServer),

    kfsGrpcImpl(dynamic_cast<const GRPCServerModule*>(this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))->getKFSGrpcImpl()),
    grpcGetModelMetadataImpl(dynamic_cast<const GRPCServerModule*>(this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))->getTFSModelMetadataImpl()) {
    registerAll();
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

void HttpRestApiHandler::registerHandler(RequestType type, std::function<Status(const HttpRequestComponents&, std::string&, const std::string&)> f) {
    handlers[type] = f;
}

void HttpRestApiHandler::registerAll() {
    registerHandler(Predict, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        if (request_components.processing_method == "predict") {
            return processPredictRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, request_body, &response);
        } else {
            SPDLOG_WARN("Requested REST resource not found");
            return (Status)StatusCode::REST_NOT_FOUND;
        }
    });

    registerHandler(GetModelMetadata, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        return processModelMetadataRequest(request_components.model_name, request_components.model_version,
            request_components.model_version_label, &response);
    });
    registerHandler(GetModelStatus, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        return processModelStatusRequest(request_components.model_name, request_components.model_version,
            request_components.model_version_label, &response);
    });
    registerHandler(ConfigReload, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        // TODO #KFS_CLEANUP
        auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
        if (nullptr == module) {
            return StatusCode::MODEL_NOT_LOADED;
        }
        auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
        // TODO #KFS_CLEANUP
        auto& manager = servableManagerModule->getServableManager();
        return processConfigReloadRequest(response, manager);
    });
    registerHandler(ConfigStatus, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        // TODO #KFS_CLEANUP
        auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
        if (nullptr == module) {
            return StatusCode::MODEL_NOT_LOADED;
        }
        auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
        // TODO #KFS_CLEANUP
        auto& manager = servableManagerModule->getServableManager();
        return processConfigStatusRequest(response, manager);
    });
    registerHandler(KFS_GetModelReady, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processModelReadyKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetModelMetadata, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processModelReadyKFSRequest(request_components, response, request_body);
    });
}

Status HttpRestApiHandler::dispatchToProcessor(
    const std::string& request_body,
    std::string* response,
    const HttpRequestComponents& request_components) {

    auto handler = handlers.find(request_components.type);
    if (handler != handlers.end()) {
        return handler->second(request_components, *response, request_body);
    } else {
        return StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE;
    }
    return StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE;
}

Status HttpRestApiHandler::processModelReadyKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::inference::ModelReadyRequest grpc_request;
    ::inference::ModelReadyResponse grpc_response;
    Status status;
    std::string modelName(request_components.model_name);
    std::string modelVersion(std::to_string(request_components.model_version.value_or(0)));
    grpc_request.set_name(modelName);
    grpc_request.set_version(modelVersion);
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersion);

    kfsGrpcImpl.ModelReady(nullptr, &grpc_request, &grpc_response);
    std::string output;
    google::protobuf::util::JsonPrintOptions opts;
    google::protobuf::util::MessageToJsonString(grpc_response, &output, opts);
    response = output;
    return StatusCode::OK;
}

Status HttpRestApiHandler::processModelMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::inference::ModelMetadataRequest grpc_request;
    ::inference::ModelMetadataResponse grpc_response;
    Status status;
    std::string modelName(request_components.model_name);
    std::string modelVersion(std::to_string(request_components.model_version.value_or(0)));
    grpc_request.set_name(modelName);
    grpc_request.set_version(modelVersion);
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersion);
    kfsGrpcImpl.ModelMetadata(nullptr, &grpc_request, &grpc_response);
    std::string output;
    google::protobuf::util::JsonPrintOptions opts;
    google::protobuf::util::MessageToJsonString(grpc_response, &output, opts);
    response = output;
    return StatusCode::OK;
}

Status HttpRestApiHandler::parseRequestComponents(HttpRequestComponents& requestComponents,
    const std::string_view http_method,
    const std::string& request_path) {
    std::smatch sm;
    requestComponents.http_method = http_method;

    if (http_method != "POST" && http_method != "GET") {
        return StatusCode::REST_UNSUPPORTED_METHOD;
    }

    if (FileSystem::isPathEscaped(request_path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    if (http_method == "POST") {
        if (std::regex_match(request_path, sm, predictionRegex)) {
            requestComponents.type = Predict;
            requestComponents.model_name = sm[2];

            std::string model_version_str = sm[3];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;

            std::string model_version_label_str = sm[4];
            if (!model_version_label_str.empty()) {
                requestComponents.model_version_label = model_version_label_str;
            }

            requestComponents.processing_method = sm[5];
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, configReloadRegex)) {
            requestComponents.type = ConfigReload;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, modelstatusRegex))
            return StatusCode::REST_UNSUPPORTED_METHOD;
    } else if (http_method == "GET") {
        if (std::regex_match(request_path, sm, modelstatusRegex)) {
            requestComponents.model_name = sm[2];

            std::string model_version_str = sm[3];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;

            std::string model_version_label_str = sm[4];
            if (!model_version_label_str.empty()) {
                requestComponents.model_version_label = model_version_label_str;
            }

            requestComponents.model_subresource = sm[5];
            if (!requestComponents.model_subresource.empty() && requestComponents.model_subresource == "metadata") {
                requestComponents.type = GetModelMetadata;
            } else {
                requestComponents.type = GetModelStatus;
            }
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, configStatusRegex)) {
            requestComponents.type = ConfigStatus;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, kfs_modelmetadataRegex)) {
            requestComponents.model_name = sm[1];
            std::string model_version_str = sm[2];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;
            requestComponents.type = KFS_GetModelMetadata;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, kfs_modelreadyRegex)) {
            requestComponents.model_name = sm[1];
            std::string model_version_str = sm[2];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;
            requestComponents.type = KFS_GetModelReady;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, predictionRegex))
            return StatusCode::REST_UNSUPPORTED_METHOD;
    }
    return StatusCode::REST_INVALID_URL;
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

    headers->clear();
    response->clear();
    headers->push_back({"Content-Type", "application/json"});

    HttpRequestComponents requestComponents;
    auto status = parseRequestComponents(requestComponents, http_method, request_path_str);
    if (!status.ok())
        return status;
    return dispatchToProcessor(request_body, response, requestComponents);
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

    // TODO #KFS_CLEANUP
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO #KFS_CLEANUP
    auto& modelManager = servableManagerModule->getServableManager();
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
    // TODO #KFS_CLEANUP
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO #KFS_CLEANUP
    auto& modelManager = servableManagerModule->getServableManager();
    auto status = modelManager.getModelInstance(
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
    status = modelInstance->infer(&requestProto, &responseProto, modelInstanceUnloadGuard);
    return status;
}

Status HttpRestApiHandler::getPipelineInputs(const std::string& modelName, ovms::tensor_map_t& inputs) {
    // TODO #KFS_CLEANUP
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO #KFS_CLEANUP
    auto& modelManager = servableManagerModule->getServableManager();
    auto pipelineDefinition = modelManager.getPipelineFactory().findDefinitionByName(modelName);
    if (!pipelineDefinition) {
        return StatusCode::MODEL_MISSING;
    }
    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;
    Status status = pipelineDefinition->waitForLoaded(unloadGuard);
    if (!status.ok()) {
        return status;
    }

    inputs = pipelineDefinition->getInputsInfo();
    return StatusCode::OK;
}

Status HttpRestApiHandler::processPipelineRequest(const std::string& modelName,
    const std::string& request,
    Order& requestOrder,
    tensorflow::serving::PredictResponse& responseProto) {

    std::unique_ptr<Pipeline> pipelinePtr;

    Timer timer;
    timer.start("parse");
    ovms::tensor_map_t inputs;
    auto status = getPipelineInputs(modelName, inputs);
    if (!status.ok()) {
        return status;
    }

    RestParser requestParser(inputs);
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop("parse");
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>("parse") / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    // TODO #KFS_CLEANUP
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO #KFS_CLEANUP
    auto& manager = servableManagerModule->getServableManager();
    status = manager.createPipeline(pipelinePtr, modelName, &requestProto, &responseProto);
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
    status = grpcGetModelMetadataImpl.createGrpcRequest(modelName, model_version, &grpc_request);
    if (!status.ok()) {
        return status;
    }
    status = grpcGetModelMetadataImpl.getModelStatus(&grpc_request, &grpc_response);
    if (!status.ok()) {
        return status;
    }
    status = grpcGetModelMetadataImpl.serializeResponse2Json(&grpc_response, response);
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
    // TODO #KFS_CLEANUP
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO #KFS_CLEANUP
    auto& manager = servableManagerModule->getServableManager();
    status = GetModelStatusImpl::getModelStatus(&grpc_request, &grpc_response, manager);
    if (!status.ok()) {
        return status;
    }
    status = GetModelStatusImpl::serializeResponse2Json(&grpc_response, response);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}

std::string createErrorJsonWithMessage(std::string message) {
    return "{\n\t\"error\": \"" + message + "\"\n}";
}

Status HttpRestApiHandler::processConfigReloadRequest(std::string& response, ModelManager& manager) {
    SPDLOG_DEBUG("Processing config reload request started.");
    Status status;
    auto& config = ovms::Config::instance();

    bool reloadNeeded = false;
    if (manager.getConfigFilename() != "") {
        status = manager.configFileReloadNeeded(reloadNeeded);
        if (!reloadNeeded) {
            if (status == StatusCode::CONFIG_FILE_TIMESTAMP_READING_FAILED) {
                response = createErrorJsonWithMessage("Config file not found or cannot open.");
                return status;
            }
        }
    }

    if (reloadNeeded) {
        status = manager.loadConfig(config.configPath());
        if (!status.ok()) {
            response = createErrorJsonWithMessage("Reloading config file failed. Check server logs for more info.");
            return status;
        }
    } else {
        if (!status.ok()) {
            status = manager.loadConfig(config.configPath());
            if (!status.ok()) {
                response = createErrorJsonWithMessage("Reloading config file failed. Check server logs for more info.");
                return status;
            }
            reloadNeeded = true;
        }
    }

    status = manager.updateConfigurationWithoutConfigFile();
    if (!status.ok()) {
        response = createErrorJsonWithMessage("Reloading models versions failed. Check server logs for more info.");
        return status;
    }
    if (status == StatusCode::OK_RELOADED) {
        reloadNeeded = true;
    }

    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    status = GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, manager);
    if (!status.ok()) {
        response = createErrorJsonWithMessage("Retrieving all model statuses failed. Check server logs for more info.");
        return status;
    }

    status = GetModelStatusImpl::serializeModelsStatuses2Json(modelsStatuses, response);
    if (!status.ok()) {
        response = createErrorJsonWithMessage("Serializing model statuses to json failed. Check server logs for more info.");
        return status;
    }

    if (!reloadNeeded) {
        SPDLOG_DEBUG("Config file reload was not needed.");
        return StatusCode::OK_NOT_RELOADED;
    }
    return StatusCode::OK_RELOADED;
}

Status HttpRestApiHandler::processConfigStatusRequest(std::string& response, ModelManager& manager) {
    SPDLOG_DEBUG("Processing config status request started.");
    Status status;

    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    status = GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, manager);
    if (!status.ok()) {
        response = createErrorJsonWithMessage("Retrieving all model statuses failed.");
        return status;
    }

    status = GetModelStatusImpl::serializeModelsStatuses2Json(modelsStatuses, response);
    if (!status.ok()) {
        response = createErrorJsonWithMessage("Serializing model statuses to json failed.");
        return status;
    }

    return StatusCode::OK;
}

}  // namespace ovms
