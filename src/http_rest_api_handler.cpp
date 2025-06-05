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

#include <cctype>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <curl/curl.h>
#endif
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

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

#if (MEDIAPIPE_DISABLE == 0)
#include "http_frontend/http_client_connection.hpp"
#include "http_frontend/http_graph_executor_impl.hpp"
#include "mediapipe_internal/mediapipegraphexecutor.hpp"
#endif

#include "tfs_frontend/tfs_utils.hpp"
#include "tfs_frontend/deserialization.hpp"
#include "deserialization_main.hpp"
#include "inference_executor.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using rapidjson::Document;
using rapidjson::SizeType;
using rapidjson::Value;

namespace {
enum : unsigned int {
    TOTAL,
    PREPARE_GRPC_REQUEST,
    TIMER_END
};
const std::string DEFAULT_VERSION = "DEFAULT";
}  // namespace

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

const std::string HttpRestApiHandler::v3_RegexExp =
    R"(/v3/.*?(/|$))";

const std::string HttpRestApiHandler::metricsRegexExp = R"((.?)\/metrics(\?(.*))?)";

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
    v3_Regex(v3_RegexExp),
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
        } catch (std::out_of_range const&) {
            return StatusCode::MODEL_VERSION_MISSING;
        } catch (std::exception&) {
            SPDLOG_DEBUG("Couldn't parse model version {}", model_version_str);
            return StatusCode::REST_COULD_NOT_PARSE_VERSION;
        }
    }
    return StatusCode::OK;
}

void HttpRestApiHandler::registerHandler(RequestType type, HandlerCallbackFn f) {
    handlers[type] = std::move(f);
}

void HttpRestApiHandler::registerAll() {
    registerHandler(Predict, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        if (request_components.processing_method == "predict") {
            return processPredictRequest(request_components.model_name, request_components.model_version,
                request_components.model_version_label, request_body, &response);
        } else {
            SPDLOG_DEBUG("Requested REST resource not found");
            return StatusCode::REST_NOT_FOUND;
        }
    });

    registerHandler(GetModelMetadata, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) {
        return processModelMetadataRequest(request_components.model_name, request_components.model_version,
            request_components.model_version_label, &response);
    });
    registerHandler(GetModelStatus, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) {
        return processModelStatusRequest(request_components.model_name, request_components.model_version,
            request_components.model_version_label, &response);
    });
    registerHandler(ConfigReload, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processConfigReloadRequest(response, this->modelManager);
    });
    registerHandler(ConfigStatus, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processConfigStatusRequest(response, this->modelManager);
    });
    registerHandler(KFS_GetModelReady, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processModelReadyKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetModelMetadata, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processModelMetadataKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_Infer, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processInferKFSRequest(request_components, response, request_body, response_components.inferenceHeaderContentLength);
    });
    registerHandler(KFS_GetServerReady, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processServerReadyKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetServerLive, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processServerLiveKFSRequest(request_components, response, request_body);
    });
    registerHandler(KFS_GetServerMetadata, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        return processServerMetadataKFSRequest(request_components, response, request_body);
    });

    registerHandler(V3, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
        OVMS_PROFILE_FUNCTION();
        return processV3(uri, request_components, response, request_body, std::move(serverReaderWriter), std::move(multiPartParser));
    });
    registerHandler(Metrics, [this](const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, HttpResponseComponents& response_components, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) -> Status {
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
    bool isLive = this->ovmsServer.isLive(HTTP_SERVER_MODULE_NAME);
    SPDLOG_DEBUG("Requested Server liveness state: {}", isLive);
    if (isLive) {
        return StatusCode::OK;
    }
    return StatusCode::INTERNAL_ERROR;
}

Status HttpRestApiHandler::processServerMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::KFSServerMetadataRequest grpc_request;
    ::KFSServerMetadataResponse grpc_response;
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
    response = std::move(output);
    return StatusCode::OK;
}

static bool isInputEmpty(const ::KFSRequest::InferInputTensor& input) {
    if (input.datatype() == "FP32")
        return input.contents().fp32_contents_size() == 0;
    if (input.datatype() == "INT64")
        return input.contents().int64_contents_size() == 0;
    if (input.datatype() == "INT32")
        return input.contents().int_contents_size() == 0;
    if (input.datatype() == "INT16")
        return input.contents().int_contents_size() == 0;
    if (input.datatype() == "INT8")
        return input.contents().int_contents_size() == 0;
    if (input.datatype() == "UINT64")
        return input.contents().uint64_contents_size() == 0;
    if (input.datatype() == "UINT32")
        return input.contents().uint_contents_size() == 0;
    if (input.datatype() == "UINT16")
        return input.contents().uint_contents_size() == 0;
    if (input.datatype() == "UINT8")
        return input.contents().uint_contents_size() == 0;
    if (input.datatype() == "FP64")
        return input.contents().fp64_contents_size() == 0;
    if (input.datatype() == "BYTES")
        return input.contents().bytes_contents_size() == 0;
    if (input.datatype() == "BOOL")
        return input.contents().bool_contents_size() == 0;
    return true;
}

static Status handleBinaryInput(const int binary_input_size, size_t& binary_input_offset, const size_t binary_buffer_size, const char* binary_inputs_buffer, ::KFSRequest::InferInputTensor& input, std::string* rawInputContentsBuffer) {
    if (binary_input_offset + binary_input_size > binary_buffer_size) {
        SPDLOG_DEBUG("Binary inputs size exceeds provided buffer size {}, binary input offset {}, binary_input size {}",
            binary_buffer_size,
            binary_input_offset,
            binary_input_size);
        return StatusCode::REST_BINARY_BUFFER_EXCEEDED;
    }
    rawInputContentsBuffer->assign(binary_inputs_buffer + binary_input_offset, binary_input_size);
    binary_input_offset += binary_input_size;
    return StatusCode::OK;
}

static size_t calculateBinaryDataSize(::KFSRequest::InferInputTensor& input) {
    auto element_size = KFSDataTypeSize(input.datatype());
    size_t elements_number = std::accumulate(std::begin(input.shape()), std::end(input.shape()), 1, std::multiplies<size_t>());
    size_t binary_data_size = elements_number * element_size;
    return binary_data_size;
}

static Status handleBinaryInputs(::KFSRequest& grpc_request, const std::string& request_body, size_t endOfJson) {
    const char* binary_inputs_buffer = &(request_body[endOfJson]);
    size_t binary_buffer_size = request_body.length() - endOfJson;

    size_t binary_input_offset = 0;
    for (int i = 0; i < grpc_request.mutable_inputs()->size(); i++) {
        auto input = grpc_request.mutable_inputs()->Mutable(i);
        auto binary_data_size_parameter = input->parameters().find("binary_data_size");
        size_t binary_input_size = 0;
        if (binary_data_size_parameter != input->parameters().end()) {
            if (!isInputEmpty(*input)) {
                SPDLOG_DEBUG("Request contains both data in json and binary inputs");
                return StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY;
            }
            if (binary_data_size_parameter->second.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                binary_input_size = binary_data_size_parameter->second.int64_param();
            } else {
                SPDLOG_DEBUG("binary_data_size parameter type should be int64");
                return StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID;
            }
        } else {
            if (!isInputEmpty(*input))
                continue;
            if (grpc_request.mutable_inputs()->size() == 1 && input->datatype() == "BYTES") {
                binary_input_size = binary_buffer_size;
            } else {
                binary_input_size = calculateBinaryDataSize(*input);
            }
        }
        auto status = handleBinaryInput(binary_input_size, binary_input_offset, binary_buffer_size, binary_inputs_buffer, *input, grpc_request.add_raw_input_contents());
        if (!status.ok()) {
            SPDLOG_DEBUG("Error handling binary input");
            return status;
        }
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::prepareGrpcRequest(const std::string modelName, const std::optional<int64_t>& modelVersion, const std::string& request_body, ::KFSRequest& grpc_request, const std::optional<int>& inferenceHeaderContentLength) {
    KFSRestParser requestParser;

    size_t endOfJson = inferenceHeaderContentLength.value_or(request_body.length());
    if (endOfJson > request_body.length()) {
        SPDLOG_DEBUG("Inference header content length exceeded JSON size");
        return StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_EXCEEDED;
    }
    auto status = requestParser.parse(request_body.substr(0, endOfJson).c_str());
    if (!status.ok()) {
        SPDLOG_DEBUG("Parsing http request failed");
        return status;
    }
    grpc_request = requestParser.getProto();
    status = handleBinaryInputs(grpc_request, request_body, endOfJson);
    if (!status.ok()) {
        SPDLOG_DEBUG("Error handling binary inputs");
        return status;
    }
    grpc_request.set_model_name(modelName);
    if (modelVersion.has_value()) {
        grpc_request.set_model_version(std::to_string(modelVersion.value()));
    }
    return StatusCode::OK;
}

static std::set<std::string> getRequestedBinaryOutputsNames(::KFSRequest& grpc_request) {
    std::set<std::string> binaryOutputs;
    bool byDefaultBinaryOutpuRequested = false;
    for (auto& parameter : grpc_request.parameters()) {
        if (parameter.second.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            if (parameter.first == "binary_data_output") {
                byDefaultBinaryOutpuRequested = parameter.second.bool_param();
                break;
            }
        }
    }
    for (const inference::ModelInferRequest_InferRequestedOutputTensor& output : grpc_request.outputs()) {
        bool specificBinaryOutputRequested = byDefaultBinaryOutpuRequested;
        for (auto& parameter : output.parameters()) {
            if ((parameter.second.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) &&
                (parameter.first == "binary_data")) {
                specificBinaryOutputRequested = parameter.second.bool_param();
                break;
            }
        }
        if (specificBinaryOutputRequested) {
            binaryOutputs.insert(output.name());
        }
    }
    return binaryOutputs;
}

Status HttpRestApiHandler::processInferKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, std::optional<int>& inferenceHeaderContentLength) {
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    ServableMetricReporter* reporter = nullptr;
    std::string modelName(request_components.model_name);
    std::string modelVersionLog = request_components.model_version.has_value() ? std::to_string(request_components.model_version.value()) : DEFAULT_VERSION;
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersionLog);
    ::KFSRequest grpc_request;
    timer.start(PREPARE_GRPC_REQUEST);
    using std::chrono::microseconds;
    auto status = prepareGrpcRequest(modelName, request_components.model_version, request_body, grpc_request, request_components.inferenceHeaderContentLength);
    ExecutionContext executionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::ModelInfer};
    if (!status.ok()) {
        auto pstatus = this->getReporter(request_components, reporter);
        if (pstatus.ok()) {
            INCREMENT_IF_ENABLED(reporter->getInferRequestMetric(executionContext, status.ok()));
        }
        SPDLOG_DEBUG("REST to GRPC request conversion failed for model: {}", modelName);
        return status;
    }
    timer.stop(PREPARE_GRPC_REQUEST);
    SPDLOG_DEBUG("Preparing grpc request time: {} ms", timer.elapsed<std::chrono::microseconds>(PREPARE_GRPC_REQUEST) / 1000);
    ::KFSResponse grpc_response;
    const Status gstatus = kfsGrpcImpl.ModelInferImpl(nullptr, &grpc_request, &grpc_response, executionContext, reporter);
    if (!gstatus.ok()) {
        return gstatus;
    }
    std::set<std::string> requestedBinaryOutputsNames = getRequestedBinaryOutputsNames(grpc_request);
    std::string output;
    status = ovms::makeJsonFromPredictResponse(grpc_response, &output, inferenceHeaderContentLength, requestedBinaryOutputsNames);
    if (!status.ok()) {
        return status;
    }
    response = std::move(output);
    timer.stop(TOTAL);
    double totalTime = timer.elapsed<std::chrono::microseconds>(TOTAL);
    SPDLOG_DEBUG("Total REST request processing time: {} ms", totalTime / 1000);

    if (!reporter) {
        return StatusCode::OK;
        // There is no request time metric for MediaPipe endpoints
    }
    OBSERVE_IF_ENABLED(reporter->requestTimeRest, totalTime);
    return StatusCode::OK;
}

Status HttpRestApiHandler::dispatchToProcessor(
    const std::string_view uri,
    const std::string& request_body,
    std::string* response,
    const HttpRequestComponents& request_components,
    HttpResponseComponents& response_components,
    std::shared_ptr<HttpAsyncWriter> serverReaderWriter,
    std::shared_ptr<MultiPartParser> multiPartParser) {

    auto handler = handlers.find(request_components.type);
    if (handler != handlers.end()) {
        return handler->second(uri, request_components, *response, request_body, response_components, std::move(serverReaderWriter), std::move(multiPartParser));
    } else {
        return StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE;
    }
    return StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE;
}

#if (MEDIAPIPE_DISABLE == 0)
static void ensureJsonParserInErrorState(std::shared_ptr<Document>& parsedJson) {
    // Hack to set json parser in invalid state in order to get HasParseError to respond with true
    parsedJson->Parse("error");
}

static Status createV3HttpPayload(
    const std::string_view uri,
    const HttpRequestComponents& request_components,
    std::string& response,
    const std::string& request_body,
    std::shared_ptr<HttpAsyncWriter> serverReaderWriter,
    std::shared_ptr<MultiPartParser> multiPartParser,
    HttpPayload& request,
    std::string& modelName,
    bool& streamFieldVal) {
    OVMS_PROFILE_SCOPE("createV3HttpPayload");

    std::shared_ptr<Document> parsedJson = std::make_shared<Document>();

    auto it = request_components.headers.find("content-type");
    bool isApplicationJson = it != request_components.headers.end() && it->second.find("application/json") != std::string::npos;
    bool isMultiPart = it != request_components.headers.end() && it->second.find("multipart/form-data") != std::string::npos;
    bool isUriBasedRouting = !isApplicationJson && !isMultiPart;  // For content types other than "application/json" and "multipart/form-data", we look for model information in the URI

    if (isMultiPart) {
        OVMS_PROFILE_SCOPE("multipart parse");
        if (!multiPartParser->parse()) {
            SPDLOG_DEBUG("Failed to parse multipart content type request");
            return StatusCode::FAILED_TO_PARSE_MULTIPART_CONTENT_TYPE;
        }
        modelName = multiPartParser->getFieldByName("model");
        if (modelName.empty()) {
            isUriBasedRouting = true;
        } else {
            SPDLOG_DEBUG("Model name from deduced from MultiPart field: {}", modelName);
        }
        ensureJsonParserInErrorState(parsedJson);
    } else if (isApplicationJson) {
        {
            OVMS_PROFILE_SCOPE("rapidjson parse");
            parsedJson->Parse(request_body.c_str());
        }
        OVMS_PROFILE_SCOPE("rapidjson validate");
        if (parsedJson->HasParseError()) {
            return Status(StatusCode::JSON_INVALID, "Cannot parse JSON body");
        }

        if (!parsedJson->IsObject()) {
            return Status(StatusCode::JSON_INVALID, "JSON body must be an object");
        }

        auto modelNameIt = parsedJson->FindMember("model");
        if (modelNameIt == parsedJson->MemberEnd()) {
            return Status(StatusCode::JSON_INVALID, "model field is missing in JSON body");
        }

        if (!modelNameIt->value.IsString()) {
            return Status(StatusCode::JSON_INVALID, "model field is not a string");
        }

        bool isTextGenerationEndpoint = uri.find("completions") != std::string_view::npos;
        if (isTextGenerationEndpoint) {
            auto streamIt = parsedJson->FindMember("stream");
            if (streamIt != parsedJson->MemberEnd()) {
                if (!streamIt->value.IsBool()) {
                    return Status(StatusCode::JSON_INVALID, "stream field is not a boolean");
                }
                streamFieldVal = streamIt->value.GetBool();
            }
        }

        modelName = modelNameIt->value.GetString();
        if (modelName.empty()) {
            isUriBasedRouting = true;
        } else {
            SPDLOG_DEBUG("Model name from deduced from JSON: {}", modelName);
        }
    }

    // Deduce Graph Name from URI since there is no info in JSON or MultiPart
    if (isUriBasedRouting) {
        if (uri.size() <= 4) {  // nothing after "/v3/..."
            SPDLOG_DEBUG("Failed to deduce model name from URI");
            return StatusCode::FAILED_TO_DEDUCE_MODEL_NAME_FROM_URI;
        }
        modelName = std::string(uri.substr(4));
        SPDLOG_DEBUG("Model name from deduced from URI: {}", modelName);
        // Set json parser in invalid state in order to get HasParseError to respond with true
        ensureJsonParserInErrorState(parsedJson);
    }

    request.headers = request_components.headers;
    request.body = request_body;
    request.parsedJson = std::move(parsedJson);
    request.uri = std::string(uri);
    request.client = std::make_shared<HttpClientConnection>(serverReaderWriter);
    request.multipartParser = std::move(multiPartParser);

    return StatusCode::OK;
}
#endif

Status HttpRestApiHandler::processV3(const std::string_view uri, const HttpRequestComponents& request_components, std::string& response, const std::string& request_body, std::shared_ptr<HttpAsyncWriter> serverReaderWriter, std::shared_ptr<MultiPartParser> multiPartParser) {
#if (MEDIAPIPE_DISABLE == 0)
    OVMS_PROFILE_FUNCTION();

    HttpPayload request;
    std::string modelName;
    bool streamFieldVal = false;

    auto status = createV3HttpPayload(uri, request_components, response, request_body, serverReaderWriter, std::move(multiPartParser), request, modelName, streamFieldVal);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to create V3 payload: {}", status.string());
        return status;
    }

    std::shared_ptr<MediapipeGraphExecutor> executor;
    status = this->modelManager.createPipeline(executor, modelName);
    if (!status.ok()) {
        return status;
    }

    if (streamFieldVal == false) {
        ExecutionContext executionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::V3Unary};
        return executor->infer(&request, &response, executionContext);
    } else {
        serverReaderWriter->OverwriteResponseHeader("Content-Type", "text/event-stream");
        serverReaderWriter->OverwriteResponseHeader("Cache-Control", "no-cache");
        serverReaderWriter->OverwriteResponseHeader("Connection", "keep-alive");
        serverReaderWriter->PartialReplyBegin([executor = std::move(executor), serverReaderWriter, request = std::move(request)] {
            ExecutionContext executionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::V3Stream};
            auto status = executor->inferStream(request, *serverReaderWriter, executionContext);
            if (!status.ok()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                writer.StartObject();
                writer.String("error");
                writer.String(status.string().c_str());
                writer.EndObject();
                serverReaderWriter->PartialReplyWithStatus(buffer.GetString(), HTTPStatusCode::BAD_REQUEST);
            }
            serverReaderWriter->PartialReplyEnd();
        });
        return StatusCode::PARTIAL_END;
    }
#else
    SPDLOG_DEBUG("Mediapipe support was disabled during build process...");
    return StatusCode::NOT_IMPLEMENTED;
#endif
}

Status HttpRestApiHandler::processMetrics(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    auto module = this->ovmsServer.getModule(METRICS_MODULE_NAME);
    if (nullptr == module) {
        SPDLOG_ERROR("Failed to process metrics - metrics module is missing");
        return StatusCode::INTERNAL_ERROR;
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
    ::KFSGetModelStatusRequest grpc_request;
    ::KFSGetModelStatusResponse grpc_response;
    std::string modelName(request_components.model_name);
    grpc_request.set_name(modelName);
    if (request_components.model_version.has_value()) {
        grpc_request.set_version(std::to_string(request_components.model_version.value()));
    }
    std::string modelVersionLog = request_components.model_version.has_value() ? std::to_string(request_components.model_version.value()) : DEFAULT_VERSION;
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersionLog);

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

void HttpRestApiHandler::convertRTInfo(Value& scope, Document& doc, ov::AnyMap& rtInfo) {
    scope.SetObject();
    for (auto& [key, value] : rtInfo) {
        SPDLOG_DEBUG("building rest response: rt_info: key: {}; value: {}", key, value.as<std::string>());
        rapidjson::Value rtInfoKey, rtInfoValue, subScope;
        rtInfoKey.SetString(key.c_str(), doc.GetAllocator());
        if (value.is<ov::AnyMap>()) {
            SPDLOG_DEBUG("building submap rest response : key: {};", key);
            subScope.SetObject();
            convertRTInfo(subScope, doc, value.as<ov::AnyMap>());
            scope.AddMember(rtInfoKey, subScope, doc.GetAllocator());
        } else {
            try {
                rtInfoValue.SetString(value.as<std::string>().c_str(), doc.GetAllocator());
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Error converting RT info value to string: {}", e.what());
                rtInfoValue.SetString("Error converting value", doc.GetAllocator());
            }
            scope.AddMember(rtInfoKey, rtInfoValue, doc.GetAllocator());
        }
    }
}

Status HttpRestApiHandler::processModelMetadataKFSRequest(const HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
    ::KFSModelMetadataRequest grpc_request;
    ::KFSModelMetadataResponse grpc_response;
    std::string modelName(request_components.model_name);
    grpc_request.set_name(modelName);
    if (request_components.model_version.has_value()) {
        grpc_request.set_version(std::to_string(request_components.model_version.value()));
    }
    std::string modelVersionLog = request_components.model_version.has_value() ? std::to_string(request_components.model_version.value()) : DEFAULT_VERSION;
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}", modelName, modelVersionLog);
    KFSModelExtraMetadata extraMetadata;
    Status gstatus = kfsGrpcImpl.ModelMetadataImpl(nullptr, &grpc_request, &grpc_response, ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::ModelMetadata}, extraMetadata);
    if (!gstatus.ok()) {
        return gstatus;
    }
    std::string output;
    google::protobuf::util::JsonPrintOptions opts;
    // This parameter forces JSON writer to not omit empty shape in case of scalar tensor
    opts.always_print_primitive_fields = true;
    google::protobuf::util::Status status = google::protobuf::util::MessageToJsonString(grpc_response, &output, opts);
    if (!status.ok()) {
        return StatusCode::JSON_SERIALIZATION_ERROR;
    }

    Document doc;
    doc.Parse(output.c_str());

    convertShapeType(doc["inputs"], doc);
    convertShapeType(doc["outputs"], doc);
    if (extraMetadata.rt_info.count("model_info")) {
        rapidjson::Value modelInfoScope, rtInfoScope;
        modelInfoScope.SetObject();
        rtInfoScope.SetObject();
        try {
            convertRTInfo(modelInfoScope, doc, extraMetadata.rt_info["model_info"].as<ov::AnyMap>());
        } catch (const std::exception& e) {
            SPDLOG_DEBUG("Error converting RT info: {}", e.what());
            return StatusCode::INTERNAL_ERROR;
        }
        rtInfoScope.AddMember("model_info", modelInfoScope, doc.GetAllocator());
        doc.AddMember("rt_info", rtInfoScope, doc.GetAllocator());
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    response = buffer.GetString();
    return StatusCode::OK;
}

static Status parseInferenceHeaderContentLength(HttpRequestComponents& requestComponents,
    const std::unordered_map<std::string, std::string>& headers) {
    for (auto& header : headers) {
        if (toLower(header.first) == "inference-header-content-length") {  // drogon automatically converts all headers to lowercase, net_http does not
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
    const std::unordered_map<std::string, std::string>& headers) {
    std::smatch sm;
    requestComponents.http_method = http_method;
    if (http_method != "POST" && http_method != "GET") {
        return StatusCode::REST_UNSUPPORTED_METHOD;
    }

    if (FileSystem::isPathEscaped(request_path)) {
        SPDLOG_DEBUG("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    if (http_method == "POST") {
        if (std::regex_match(request_path, sm, predictionRegex)) {
            requestComponents.type = Predict;
            requestComponents.model_name = urlDecode(sm[2]);

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
        if (std::regex_match(request_path, sm, kfs_inferRegex, std::regex_constants::match_any)) {
            requestComponents.type = KFS_Infer;
            requestComponents.model_name = urlDecode(sm[1]);
            std::string model_version_str = sm[2];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;

            status = parseInferenceHeaderContentLength(requestComponents, headers);
            if (!status.ok())
                return status;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, v3_Regex)) {
            requestComponents.type = V3;
            auto status = parseInferenceHeaderContentLength(requestComponents, headers);
            if (!status.ok())
                return status;
            requestComponents.headers = headers;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, configReloadRegex)) {
            requestComponents.type = ConfigReload;
            return StatusCode::OK;
        }
        return (std::regex_match(request_path, sm, modelstatusRegex) ||
                   std::regex_match(request_path, sm, kfs_serverliveRegex) ||
                   std::regex_match(request_path, sm, configStatusRegex) ||
                   std::regex_match(request_path, sm, kfs_serverreadyRegex) ||
                   std::regex_match(request_path, sm, kfs_servermetadataRegex) ||
                   std::regex_match(request_path, sm, kfs_modelmetadataRegex) ||
                   std::regex_match(request_path, sm, kfs_modelreadyRegex) ||
                   std::regex_match(request_path, sm, metricsRegex))
                   ? StatusCode::REST_UNSUPPORTED_METHOD
                   : StatusCode::REST_INVALID_URL;

    } else if (http_method == "GET") {
        if (std::regex_match(request_path, sm, modelstatusRegex)) {
            requestComponents.model_name = urlDecode(sm[2]);
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
            requestComponents.model_name = urlDecode(sm[1]);
            std::string model_version_str = sm[2];
            auto status = parseModelVersion(model_version_str, requestComponents.model_version);
            if (!status.ok())
                return status;
            requestComponents.type = KFS_GetModelMetadata;
            return StatusCode::OK;
        }
        if (std::regex_match(request_path, sm, kfs_modelreadyRegex)) {
            requestComponents.model_name = urlDecode(sm[1]);
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
            std::string params = sm[3];
            if (!params.empty()) {
                SPDLOG_DEBUG("Discarded following url parameters: {}", params);
            }
            requestComponents.type = Metrics;
            return StatusCode::OK;
        }
        return (std::regex_match(request_path, sm, predictionRegex) ||
                   std::regex_match(request_path, sm, kfs_inferRegex, std::regex_constants::match_any) ||
                   std::regex_match(request_path, sm, configReloadRegex))
                   ? StatusCode::REST_UNSUPPORTED_METHOD
                   : StatusCode::REST_INVALID_URL;
    }
    return StatusCode::REST_INVALID_URL;
}

Status HttpRestApiHandler::processRequest(
    const std::string_view http_method,
    const std::string_view request_path,
    const std::string& request_body,
    std::unordered_map<std::string, std::string>* headers,
    std::string* response,
    HttpResponseComponents& responseComponents,
    std::shared_ptr<HttpAsyncWriter> serverReaderWriter,
    std::shared_ptr<MultiPartParser> multiPartParser) {

    std::smatch sm;
    std::string request_path_str(request_path);
    if (FileSystem::isPathEscaped(request_path_str)) {
        SPDLOG_DEBUG("Path {} escape with .. is forbidden.", request_path);
        return StatusCode::PATH_INVALID;
    }

    HttpRequestComponents requestComponents;
    auto status = parseRequestComponents(requestComponents, http_method, request_path_str, *headers);

    if (!status.ok())
        return status;

    response->clear();
    return dispatchToProcessor(request_path, request_body, response, requestComponents, responseComponents, std::move(serverReaderWriter), std::move(multiPartParser));
}

Status HttpRestApiHandler::processPredictRequest(
    const std::string& modelName,
    const std::optional<int64_t>& modelVersion,
    const std::optional<std::string_view>& modelVersionLabel,
    const std::string& request,
    std::string* response) {
    // model_version_label currently is not in use

    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    using std::chrono::microseconds;

    std::string modelVersionLog = modelVersion.has_value() ? std::to_string(modelVersion.value()) : DEFAULT_VERSION;
    SPDLOG_DEBUG("Processing REST request for model: {}; version: {}",
        modelName, modelVersionLog);
    Order requestOrder = Order::UNKNOWN;
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
        SPDLOG_DEBUG("Model or pipeline matching request parameters not found - name: {}, version: {}", modelName, modelVersionLog);
        status = StatusCode::MODEL_NAME_MISSING;
    }
    if (!status.ok())
        return status;
    status = makeJsonFromPredictResponse(responseProto, response, requestOrder);
    if (!status.ok())
        return status;

    timer.stop(TOTAL);
    double requestTime = timer.elapsed<std::chrono::microseconds>(TOTAL);
    SPDLOG_DEBUG("Total REST request processing time: {} ms", requestTime / 1000);
    if (!reporterOut) {
        return StatusCode::OK;
        // There is no request time metric for MediaPipe endpoints
    }
    OBSERVE_IF_ENABLED(reporterOut->requestTimeRest, requestTime);
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
        std::string modelVersionLog = modelVersion.has_value() ? std::to_string(modelVersion.value()) : DEFAULT_VERSION;
        SPDLOG_DEBUG("Requested model instance - name: {}, version: {} - does not exist.", modelName, modelVersionLog);
        return status;
    }
    reporterOut = &modelInstance->getMetricReporter();
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    TFSRestParser requestParser(modelInstance->getInputsInfo());
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailRestPredict);
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop(TOTAL);
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>(TOTAL) / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    if (modelVersion.has_value()) {
        requestProto.mutable_model_spec()->mutable_version()->set_value(modelVersion.value());
    }
    status = infer(*modelInstance, &requestProto, &responseProto, modelInstanceUnloadGuard);
    INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getInferRequestMetric(ExecutionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::Predict}, status.ok()));
    return status;
}

Status HttpRestApiHandler::getReporter(const HttpRequestComponents& components, ovms::ServableMetricReporter*& reporter) {
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;
    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = this->modelManager.getModelInstance(components.model_name, components.model_version.value_or(0), modelInstance, modelInstanceUnloadGuard);
    if (status == StatusCode::MODEL_NAME_MISSING) {
        auto pipelineDefinition = this->modelManager.getPipelineFactory().findDefinitionByName(components.model_name);
        if (!pipelineDefinition) {
            return StatusCode::MODEL_MISSING;
        }
        reporter = &pipelineDefinition->getMetricReporter();
    } else if (status.ok()) {
        reporter = &modelInstance->getMetricReporter();
    } else {
        return StatusCode::MODEL_MISSING;
    }
    return StatusCode::OK;
}

Status HttpRestApiHandler::getPipelineInputsAndReporter(const std::string& modelName, ovms::tensor_map_t& inputs, ovms::ServableMetricReporter*& reporter) {
    auto pipelineDefinition = this->modelManager.getPipelineFactory().findDefinitionByName(modelName);
    if (!pipelineDefinition) {
        return StatusCode::MODEL_MISSING;
    }
    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;
    Status status = pipelineDefinition->waitForLoaded(unloadGuard);
    if (!status.ok()) {
        return status;
    }
    reporter = &pipelineDefinition->getMetricReporter();
    inputs = pipelineDefinition->getInputsInfo();
    return StatusCode::OK;
}

Status HttpRestApiHandler::processPipelineRequest(const std::string& modelName,
    const std::string& request,
    Order& requestOrder,
    tensorflow::serving::PredictResponse& responseProto,
    ServableMetricReporter*& reporterOut) {
    ExecutionContext executionContext{ExecutionContext::Interface::REST, ExecutionContext::Method::Predict};
    std::unique_ptr<Pipeline> pipelinePtr;

    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    ovms::tensor_map_t inputs;
    auto status = getPipelineInputsAndReporter(modelName, inputs, reporterOut);
    if (!status.ok()) {
        if (reporterOut) {
            INCREMENT_IF_ENABLED(reporterOut->getInferRequestMetric(executionContext, false));
        }
        return status;
    }

    TFSRestParser requestParser(inputs);
    status = requestParser.parse(request.c_str());
    if (!status.ok()) {
        INCREMENT_IF_ENABLED(reporterOut->getInferRequestMetric(executionContext, false));
        return status;
    }
    requestOrder = requestParser.getOrder();
    timer.stop(TOTAL);
    SPDLOG_DEBUG("JSON request parsing time: {} ms", timer.elapsed<std::chrono::microseconds>(TOTAL) / 1000);

    tensorflow::serving::PredictRequest& requestProto = requestParser.getProto();
    requestProto.mutable_model_spec()->set_name(modelName);
    status = this->modelManager.createPipeline(pipelinePtr, modelName, &requestProto, &responseProto);
    if (!status.ok()) {
        INCREMENT_IF_ENABLED(reporterOut->getInferRequestMetric(executionContext, false));
        return status;
    }
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

inline static std::string createErrorJsonWithMessage(std::string message) {
    return "{\n\t\"error\": \"" + message + "\"\n}";
}

Status HttpRestApiHandler::processConfigReloadRequest(std::string& response, ModelManager& manager) {
    SPDLOG_DEBUG("Processing config reload request started.");
    Status status;
    auto& config = ovms::Config::instance();

    bool reloadNeeded = false;
    if (manager.isStartedWithConfigFile()) {
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

std::string urlDecode(const std::string& encoded) {
    std::ostringstream decoded;
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] == '%') {
            // Check if the next two characters are valid hex digits
            if (i + 2 < encoded.size() &&
                std::isxdigit(static_cast<unsigned char>(encoded[i + 1])) &&
                std::isxdigit(static_cast<unsigned char>(encoded[i + 2]))) {
                // Convert the two hexadecimal digits to a character
                int value = 0;
                std::stringstream hex_value;
                hex_value << encoded.substr(i + 1, 2);
                hex_value >> std::hex >> value;
                decoded << static_cast<char>(value);
                i += 2;  // Skip the next two characters
            } else {
                // Invalid escape sequence, copy '%' as is
                decoded << '%';
            }
        } else {
            // Regular character, just add it
            decoded << encoded[i];
        }
    }
    return decoded.str();
}

}  // namespace ovms
