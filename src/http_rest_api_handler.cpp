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
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "config.hpp"
#include "execution_context.hpp"
#include "filesystem.hpp"
#include "get_model_metadata_impl.hpp"
#include "grpcservermodule.hpp"
#include "kfs_grpc_inference_service.hpp"
#include "metric_module.hpp"
#include "metric_registry.hpp"
#include "model_metric_reporter.hpp"
#include "model_service.hpp"
#include "modelinstanceunloadguard.hpp"
#include "pipelinedefinition.hpp"
#include "prediction_service_utils.hpp"
#include "rest_parser.hpp"
#include "rest_utils.hpp"
#include "server.hpp"
#include "stringutils.hpp"
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using rapidjson::Document;
using rapidjson::SizeType;
using rapidjson::Value;

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
const std::string HttpRestApiHandler::kfs_inferRegexExp =
    R"(/v2/models/([^/]+)(?:/versions/([0-9]+))?(?:/(infer)))";
const std::string HttpRestApiHandler::kfs_serverreadyRegexExp =
    R"(/v2/health/ready)";
const std::string HttpRestApiHandler::kfs_serverliveRegexExp =
    R"(/v2/health/live)";
const std::string HttpRestApiHandler::kfs_servermetadataRegexExp =
    R"(/v2)";

const std::string HttpRestApiHandler::metricsRegexExp = R"((.?)\/metrics)";

HttpRestApiHandler::HttpRestApiHandler(ovms::Server& ovmsServer, int timeout_in_ms) :
    predictionRegex(predictionRegexExp),
    modelstatusRegex(modelstatusRegexExp),
    configReloadRegex(configReloadRegexExp),
    configStatusRegex(configStatusRegexExp),
    kfs_modelreadyRegex(kfs_modelreadyRegexExp),
    kfs_modelmetadataRegex(kfs_modelmetadataRegexExp),
    kfs_inferRegex(kfs_inferRegexExp),
    kfs_serverreadyRegex(kfs_serverreadyRegexExp),
    kfs_serverliveRegex(kfs_serverliveRegexExp),
    kfs_servermetadataRegex(kfs_servermetadataRegexExp),
    metricsRegex(metricsRegexExp),
    timeout_in_ms(timeout_in_ms),
    ovmsServer(ovmsServer),

    kfsGrpcImpl(dynamic_cast<const GRPCServerModule*>(this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))->getKFSGrpcImpl()),
    grpcGetModelMetadataImpl(dynamic_cast<const GRPCServerModule*>(this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))->getTFSModelMetadataImpl()),
    modelManager(dynamic_cast<const ServableManagerModule*>(this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == this->ovmsServer.getModule(GRPC_SERVER_MODULE_NAME))
        throw std::logic_error("Tried to create http rest api handler without grpc server module");
    if (nullptr == this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))
        throw std::logic_error("Tried to create http rest api handler without servable manager module");
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
    registerHandler(Predict, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        if (request_components.processing_method == "predict") {
            return processPredictRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, request_body, &response);
        } else {
            SPDLOG_DEBUG("Requested REST resource not found");
            return StatusCode::REST_NOT_FOUND;
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
        return processConfigReloadRequest(response, this->modelManager);
    });
    registerHandler(ConfigStatus, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processConfigStatusRequest(response, this->modelManager);
    });
    registerHandler(KFS_GetModelReady, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processModelReadyKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetModelMetadata, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processModelMetadataKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_Infer, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processInferKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetServerReady, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processServerReadyKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetServerLive, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processServerLiveKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetServerMetadata, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processServerMetadataKFSRequest(request_components, response, request_body);
    });
    registerHandler(Metrics, [this](const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) -> Status {
        return processMetrics(request_components, response, request_body);
    });
}

Status HttpRestApiHandler::processServerReadyKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    bool isReady = this->ovmsServer.isReady();
    SPDLOG_DEBUG("Requested Server readiness state: {}", isReady);
    if (isReady) {
        return StatusCode::OK;
    }
    return StatusCode::MODEL_NOT_LOADED;
}

Status HttpRestApiHandler::processServerLiveKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    bool isLive = this->ovmsServer.isLive();
    SPDLOG_DEBUG("Requested Server liveness state: {}", isLive);
    if (isLive) {
        return StatusCode::OK;
    }
    return StatusCode::INTERNAL_ERROR;
}

Status HttpRestApiHandler::processServerMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::inference::ServerMetadataRequest grpc_request;
    ::inference::ServerMetadataResponse grpc_response;
    Status gstatus = kfsGrpcImpl.ServerMetadataImpl(nullptr, &grpc_request, &grpc_response);
    if (!gstatus.ok()) {
        return gstatus;
    }
    std::string output;
    google::protobuf::util::JsonPrintOptions opts;
    google::protobuf::util::Status status = google::protobuf::util::MessageToJsonString(grpc_response, &output, opts);
    if (!status.ok()) {
        return StatusCode::INTERNAL_ERROR;
    }
    response = output;
    return StatusCode::OK;
}

void HttpRestApiHandler::parseParams(Value& scope, Document& doc) {
    Value::ConstMemberIterator itr = scope.FindMember("parameters");
    if (itr != scope.MemberEnd()) {
        for (Value::ConstMemberIterator i = scope["parameters"].MemberBegin(); i != scope["parameters"].MemberEnd(); ++i) {
            Value param(rapidjson::kObjectType);
            if (i->value.IsInt64()) {
                Value value(i->value.GetInt64());
                param.AddMember("int64_param", value, doc.GetAllocator());
            }
            if (i->value.IsString()) {
                Value value(i->value.GetString(), doc.GetAllocator());
                param.AddMember("string_param", value, doc.GetAllocator());
            }
            if (i->value.IsBool()) {
                Value value(i->value.GetBool());
                param.AddMember("bool_param", value, doc.GetAllocator());
            }
            scope["parameters"].GetObject()[i->name.GetString()] = param;
        }
    }
}

std::string HttpRestApiHandler::preprocessInferRequest(std::string request_body) {
    static std::unordered_map<std::string, std::string> types = {
        {"BOOL", "bool_contents"},
        {"INT8", "int_contents"},
        {"INT16", "int_contents"},
        {"INT32", "int_contents"},
        {"INT64", "int64_contents"},
        {"UINT8", "uint_contents"},
        {"UINT16", "uint_contents"},
        {"UINT32", "uint_contents"},
        {"UINT64", "uint64_contents"},
        {"FP32", "fp32_contents"},
        {"FP64", "fp64_contents"},
        {"BYTES", "bytes_contents"}};

    Document doc;
    doc.Parse(request_body.c_str());
    Value& inputs = doc["inputs"];
    for (SizeType i = 0; i < inputs.Size(); i++) {
        Value data = inputs[i].GetObject()["data"].GetArray();
        Value contents(rapidjson::kObjectType);
        Value datatype(types[inputs[i].GetObject()["datatype"].GetString()].c_str(), doc.GetAllocator());
        contents.AddMember(datatype, data, doc.GetAllocator());
        inputs[i].AddMember("contents", contents, doc.GetAllocator());
        parseParams(inputs[i], doc);
    }
    Value& outputs = doc["outputs"];
    for (SizeType i = 0; i < outputs.Size(); i++) {
        parseParams(outputs[i], doc);
    }
    parseParams(doc, doc);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return buffer.GetString();
}

Status convertStringToVectorOfSizes(const std::string& comma_separated_numbers, std::vector<int>& sizes) {
    std::stringstream streamData(comma_separated_numbers);
    std::vector<int> sizes_;

    std::string numberString;
    while (std::getline(streamData, numberString, ',')) {
        std::optional<int> binarySize = stoi32(numberString);
        if (!binarySize.has_value()) {
            SPDLOG_DEBUG("Invalid argument in binary size string: {}", numberString);
            return StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID;
        }
        sizes_.push_back(binarySize.value());
    }
    sizes = std::move(sizes_);

    return StatusCode::OK;
}

Status parseBinaryInput(::inference::ModelInferRequest_InferInputTensor* input, size_t binary_input_size, const char* buffer) {
    if (input->datatype() == "FP32") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(float)) {
            auto value = input->mutable_contents()->mutable_fp32_contents()->Add();
            *value = (*(reinterpret_cast<const float*>(buffer + i)));
        }
    } else if (input->datatype() == "INT64") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(int64_t)) {
            auto value = input->mutable_contents()->mutable_int64_contents()->Add();
            *value = (*(reinterpret_cast<const int64_t*>(buffer + i)));
        }
    } else if (input->datatype() == "INT32") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(int32_t)) {
            auto value = input->mutable_contents()->mutable_int_contents()->Add();
            *value = (*(reinterpret_cast<const int32_t*>(buffer + i)));
        }
    } else if (input->datatype() == "INT16") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(int16_t)) {
            auto value = input->mutable_contents()->mutable_int_contents()->Add();
            *value = (*(reinterpret_cast<const int16_t*>(buffer + i)));
        }
    } else if (input->datatype() == "INT8") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(int8_t)) {
            auto value = input->mutable_contents()->mutable_int_contents()->Add();
            *value = (*(reinterpret_cast<const int8_t*>(buffer + i)));
        }
    } else if (input->datatype() == "UINT64") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(uint64_t)) {
            auto value = input->mutable_contents()->mutable_uint64_contents()->Add();
            *value = (*(reinterpret_cast<const uint64_t*>(buffer + i)));
        }
    } else if (input->datatype() == "UINT32") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(uint32_t)) {
            auto value = input->mutable_contents()->mutable_uint_contents()->Add();
            *value = (*(reinterpret_cast<const uint32_t*>(buffer + i)));
        }
    } else if (input->datatype() == "UINT16") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(uint16_t)) {
            auto value = input->mutable_contents()->mutable_uint_contents()->Add();
            *value = (*(reinterpret_cast<const uint16_t*>(buffer + i)));
        }
    } else if (input->datatype() == "UINT8") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(uint8_t)) {
            auto value = input->mutable_contents()->mutable_uint_contents()->Add();
            *value = (*(reinterpret_cast<const uint8_t*>(buffer + i)));
        }
    } else if (input->datatype() == "FP64") {
        for (size_t i = 0; i < binary_input_size; i += sizeof(double)) {
            auto value = input->mutable_contents()->mutable_fp64_contents()->Add();
            *value = (*(reinterpret_cast<const double*>(buffer + i)));
        }
    } else if (input->datatype() == "BYTES") {
        input->mutable_contents()->add_bytes_contents(buffer, binary_input_size);
    } else {
        return StatusCode::REST_UNSUPPORTED_PRECISION;
    }

    return StatusCode::OK;
}

#define CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE " contents is not empty. Content field should be empty when using binary inputs extension."

Status validateContentFieldsEmptiness(::inference::ModelInferRequest_InferInputTensor* input) {
    if (input->datatype() == "FP32") {
        if (input->contents().fp32_contents_size() > 0) {
            SPDLOG_DEBUG("FP32" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "INT64") {
        if (input->contents().int64_contents_size() > 0) {
            SPDLOG_DEBUG("INT64" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "INT32") {
        if (input->contents().int_contents_size() > 0) {
            SPDLOG_DEBUG("INT32" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "INT16") {
        if (input->contents().int_contents_size() > 0) {
            SPDLOG_DEBUG("INT16" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "INT8") {
        if (input->contents().int_contents_size() > 0) {
            SPDLOG_DEBUG("INT8" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "UINT64") {
        if (input->contents().uint64_contents_size() > 0) {
            SPDLOG_DEBUG("UINT64" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "UINT32") {
        if (input->contents().uint_contents_size() > 0) {
            SPDLOG_DEBUG("UINT32" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "UINT16") {
        if (input->contents().uint_contents_size() > 0) {
            SPDLOG_DEBUG("UINT16" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "UINT8") {
        if (input->contents().uint_contents_size() > 0) {
            SPDLOG_DEBUG("UINT8" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "FP64") {
        if (input->contents().fp64_contents_size() > 0) {
            SPDLOG_DEBUG("FP64" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else if (input->datatype() == "BYTES") {
        if (input->contents().bytes_contents_size() > 0) {
            SPDLOG_DEBUG("BYTES" CONTENT_FIELD_NOT_EMPTY_ERROR_MESSAGE);
            return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
        }
    } else {
        return StatusCode::REST_UNSUPPORTED_PRECISION;
    }

    return StatusCode::OK;
}

Status handleBinaryInputs(::inference::ModelInferRequest& grpc_request, const std::string request_body, size_t endOfJson) {
    const char* binary_inputs = &(request_body[endOfJson]);
    size_t binary_inputs_size = request_body.length() - endOfJson;

    size_t binary_input_offset = 0;
    for (int i = 0; i < grpc_request.mutable_inputs()->size(); i++) {
        auto input = grpc_request.mutable_inputs()->Mutable(i);
        auto binary_data_size_parameter = input->parameters().find("binary_data_size");
        if (binary_data_size_parameter != input->parameters().end()) {
            auto status = validateContentFieldsEmptiness(input);
            if (!status.ok()) {
                return status;
            }
            if (binary_data_size_parameter->second.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                auto binary_input_size = binary_data_size_parameter->second.int64_param();
                if (binary_input_offset + binary_input_size > binary_inputs_size) {
                    SPDLOG_DEBUG("Binary inputs size exceeds provided buffer size {}", binary_inputs_size);
                    return StatusCode::REST_BINARY_BUFFER_EXCEEDED;
                }
                status = parseBinaryInput(input, binary_input_size, binary_inputs + binary_input_offset);
                if (!status.ok()) {
                    return status;
                }
                binary_input_offset += binary_input_size;
            } else if (binary_data_size_parameter->second.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
                std::vector<int> binary_inputs_sizes;
                status = convertStringToVectorOfSizes(binary_data_size_parameter->second.string_param(), binary_inputs_sizes);
                if (!status.ok()) {
                    return status;
                }
                for (auto size : binary_inputs_sizes) {
                    if (binary_input_offset + size > binary_inputs_size) {
                        SPDLOG_DEBUG("Binary inputs size exceeds provided buffer size {}", binary_inputs_size);
                        return StatusCode::REST_BINARY_BUFFER_EXCEEDED;
                    }
                    status = parseBinaryInput(input, size, binary_inputs + binary_input_offset);
                    if (!status.ok()) {
                        return status;
                    }
                    binary_input_offset += size;
                }
            } else {
                SPDLOG_DEBUG("binary_data_size parameter type should be int64 or string");
                return StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID;
            }
        }
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::prepareGrpcRequest(const std::string modelName, const std::string modelVersion, const std::string& request_body, ::inference::ModelInferRequest& grpc_request, std::optional<int>& inferenceHeaderContentLength) {
    KFSRestParser requestParser;

    size_t endOfJson = inferenceHeaderContentLength.value_or(request_body.length());
    auto status = requestParser.parse(request_body.substr(0, endOfJson).c_str());
    if (!status.ok()) {
        // modelInstance->getMetricReporter().requestFailRestPredict->increment();
        return status;
    }
    grpc_request = requestParser.getProto();
    status = handleBinaryInputs(grpc_request, request_body, endOfJson);
    if (!status.ok()) {
        return status;
    }
    grpc_request.set_model_name(modelName);
    grpc_request.set_model_version(modelVersion);
    return StatusCode::OK;
}

Status HttpRestApiHandler::processInferKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    Timer timer;
    timer.start("total");
    ServableMetricReporter* reporter = nullptr;
    std::string modelName(request_components.model_name);
    std::string modelVersion(std::to_string(request_components.model_version.value_or(0)));
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersion);
    ::inference::ModelInferRequest grpc_request;
    Timer timer;
    timer.start("prepareGrpcRequest");
    using std::chrono::microseconds;
    auto status = prepareGrpcRequest(modelName, modelVersion, request_body, grpc_request, request_components.inferenceHeaderContentLength);
    if (!status.ok()) {
        SPDLOG_DEBUG("REST to GRPC request conversion failed dor model: {}", modelName);
        return status;
    }
    timer.stop("prepareGrpcRequest");
    SPDLOG_DEBUG("Preparing grpc request time: {} ms", timer.elapsed<std::chrono::microseconds>("prepareGrpcRequest") / 1000);
    ::inference::ModelInferResponse grpc_response;
    const Status gstatus = kfsGrpcImpl.ModelInferImpl(nullptr, &grpc_request, &grpc_response, ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::ModelInfer},
        reporter);
    if (!gstatus.ok()) {
        return gstatus;
    }
    std::string output;
    google::protobuf::util::JsonPrintOptions opts_out;
    status = ovms::makeJsonFromPredictResponse(grpc_response, &output);
    if (!status.ok()) {
        return status;
    }
    response = output;
    timer.stop("total");
    double totalTime = timer.elapsed<std::chrono::microseconds>("total");
    SPDLOG_DEBUG("Total REST request processing time: {} ms", totalTime / 1000);
    OBSERVE_IF_ENABLED(reporter->requestTimeRest, totalTime);
    return StatusCode::OK;
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

Status HttpRestApiHandler::processMetrics(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    auto module = this->ovmsServer.getModule(METRICS_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::INTERNAL_ERROR;  // TODO: Return proper code when metric endpoint is disabled (missing module).
    }
    auto& metricConfig = this->modelManager.getMetricConfig();

    if (!metricConfig.metricsEnabled) {
        return StatusCode::REST_INVALID_URL;
    }

    auto metricModule = dynamic_cast<const MetricModule*>(module);
    response = metricModule->getRegistry().collect();

    return StatusCode::OK;
}

Status HttpRestApiHandler::processModelReadyKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::inference::ModelReadyRequest grpc_request;
    ::inference::ModelReadyResponse grpc_response;
    std::string modelName(request_components.model_name);
    std::string modelVersion(std::to_string(request_components.model_version.value_or(0)));
    grpc_request.set_name(modelName);
    grpc_request.set_version(modelVersion);
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersion);

    Status status = kfsGrpcImpl.ModelReadyImpl(nullptr, &grpc_request, &grpc_response, ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::ModelReady});
    if (!status.ok()) {
        return status;
    }

    if (grpc_response.ready()) {
        return StatusCode::OK;
    }
    return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
}

void HttpRestApiHandler::convertShapeType(Value& scope, Document& doc) {
    for (SizeType i = 0; i < scope.Size(); i++) {
        Value data = scope[i].GetObject()["shape"].GetArray();
        Value shape(rapidjson::kArrayType);
        for (SizeType j = 0; j < data.Size(); j++) {
            shape.PushBack(atoi(data[j].GetString()), doc.GetAllocator());
        }
        scope[i].GetObject()["shape"] = shape;
    }
}

Status HttpRestApiHandler::processModelMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::inference::ModelMetadataRequest grpc_request;
    ::inference::ModelMetadataResponse grpc_response;
    std::string modelName(request_components.model_name);
    std::string modelVersion(std::to_string(request_components.model_version.value_or(0)));
    grpc_request.set_name(modelName);
    grpc_request.set_version(modelVersion);
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersion);
    Status gstatus = kfsGrpcImpl.ModelMetadataImpl(nullptr, &grpc_request, &grpc_response, ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::ModelMetadata});
    if (!gstatus.ok()) {
        return gstatus;
    }
    std::string output;
    google::protobuf::util::JsonPrintOptions opts;
    google::protobuf::util::Status status = google::protobuf::util::MessageToJsonString(grpc_response, &output, opts);
    if (!status.ok()) {
        return StatusCode::JSON_SERIALIZATION_ERROR;
    }

    Document doc;
    doc.Parse(output.c_str());

    convertShapeType(doc["inputs"], doc);
    convertShapeType(doc["outputs"], doc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    response = buffer.GetString();
    return StatusCode::OK;
}

Status parseInferenceHeaderContentLength(HttpRequestComponents& requestComponents,
    const std::vector<std::pair<std::string, std::string>>& headers) {
    for (auto header : headers) {
        if (header.first == "Inference-Header-Content-Length") {
            requestComponents.inferenceHeaderContentLength = stoi32(header.second);
            if (!requestComponents.inferenceHeaderContentLength.has_value() || requestComponents.inferenceHeaderContentLength.value() < 0) {
                return StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID;
            }
        }
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::parseRequestComponents(HttpRequestComponents& requestComponents,
    const std::string_view http_method,
    const std::string& request_path,
    const std::vector<std::pair<std::string, std::string>>& headers) {
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
        if (std::regex_match(request_path, sm, kfs_inferRegex, std::regex_constants::match_any)) {
            requestComponents.type = KFS_Infer;
            requestComponents.model_name = sm[1];
            std::string model_version_str = sm[2];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;

            status = parseInferenceHeaderContentLength(requestComponents, headers);
            if (!status.ok())
                return status;
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
        if (std::regex_match(request_path, sm, kfs_serverliveRegex)) {
            requestComponents.type = KFS_GetServerLive;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, kfs_serverreadyRegex)) {
            requestComponents.type = KFS_GetServerReady;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, kfs_servermetadataRegex)) {
            requestComponents.type = KFS_GetServerMetadata;
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
        if (std::regex_match(request_path, sm, metricsRegex)) {
            requestComponents.type = Metrics;
            return StatusCode::OK;
        }
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

    HttpRequestComponents requestComponents;
    auto status = parseRequestComponents(requestComponents, http_method, request_path_str, *headers);

    headers->clear();
    response->clear();
    headers->push_back({"Content-Type", "application/json"});

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

    Order requestOrder;
    tensorflow::serving::PredictResponse responseProto;
    Status status;

    ServableMetricReporter* reporterOut = nullptr;
    if (this->modelManager.modelExists(modelName)) {
        SPDLOG_DEBUG("Found model with name: {}. Searching for requested version...", modelName);
        status = processSingleModelRequest(modelName, modelVersion, request, requestOrder, responseProto, reporterOut);
    } else if (this->modelManager.pipelineDefinitionExists(modelName)) {
        SPDLOG_DEBUG("Found pipeline with name: {}", modelName);
        status = processPipelineRequest(modelName, request, requestOrder, responseProto, reporterOut);
    } else {
        SPDLOG_WARN("Model or pipeline matching request parameters not found - name: {}, version: {}", modelName, modelVersion.value_or(0));
        status = StatusCode::MODEL_NAME_MISSING;
    }
    if (!status.ok())
        return status;
    if (!reporterOut) {
        return StatusCode::INTERNAL_ERROR;  // should not happen
    }

    status = makeJsonFromPredictResponse(responseProto, response, requestOrder);
    if (!status.ok())
        return status;

    timer.stop("total");
    double requestTime = timer.elapsed<std::chrono::microseconds>("total");
    OBSERVE_IF_ENABLED(reporterOut->requestTimeRest, requestTime);
    SPDLOG_DEBUG("Total REST request processing time: {} ms", requestTime / 1000);
    return StatusCode::OK;
}

Status HttpRestApiHandler::processSingleModelRequest(const std::string& modelName,
    const std::optional<int64_t>& modelVersion,
    const std::string& request,
    Order& requestOrder,
    tensorflow::serving::PredictResponse& responseProto,
    ServableMetricReporter*& reporterOut) {

    std::shared_ptr<ModelInstance> modelInstance;
    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = this->modelManager.getModelInstance(
        modelName,
        modelVersion.value_or(0),
        modelInstance,
        modelInstanceUnloadGuard);

    if (!status.ok()) {
        if (modelInstance) {
            INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailRestPredict);
        }
        SPDLOG_WARN("Requested model instance - name: {}, version: {} - does not exist.", modelName, modelVersion.value_or(0));
        return status;
    }
    reporterOut = &modelInstance->getMetricReporter();
    Timer timer;
    timer.start("parse");
    TFSRestParser requestParser(modelInstance->getInputsInfo());
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailRestPredict);
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
    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getInferRequestMetric(ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::Predict}, status.ok()));
    return status;
}

Status HttpRestApiHandler::getPipelineInputs(const std::string& modelName, ovms::tensor_map_t& inputs) {
    auto pipelineDefinition = this->modelManager.getPipelineFactory().findDefinitionByName(modelName);
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
    tensorflow::serving::PredictResponse& responseProto,
    ServableMetricReporter*& reporterOut) {

    std::unique_ptr<Pipeline> pipelinePtr;

    Timer timer;
    timer.start("parse");
    ovms::tensor_map_t inputs;
    auto status = getPipelineInputs(modelName, inputs);
    if (!status.ok()) {
        return status;
    }

    TFSRestParser requestParser(inputs);
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop("parse");
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>("parse") / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    status = this->modelManager.createPipeline(pipelinePtr, modelName, &requestProto, &responseProto);
    if (!status.ok()) {
        return status;
    }
    ExecutionContext executionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::Predict};
    reporterOut = &pipelinePtr->getMetricReporter();
    status = pipelinePtr->execute(executionContext);
    INCREMENT_IF_ENABLED(pipelinePtr->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
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
    status = grpcGetModelMetadataImpl.getModelStatus(&grpc_request, &grpc_response, ExecutionContext(ExecutionContext::Interface::REST, ExecutionContext::Method::GetModelMetadata));
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
    status = GetModelStatusImpl::getModelStatus(&grpc_request, &grpc_response, this->modelManager, ExecutionContext(ExecutionContext::Interface::REST, ExecutionContext::Method::GetModelStatus));
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
    status = GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, manager, ExecutionContext(ExecutionContext::Interface::REST, ExecutionContext::Method::ConfigReload));
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
    status = GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, manager, ExecutionContext(ExecutionContext::Interface::REST, ExecutionContext::Method::ConfigStatus));
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
