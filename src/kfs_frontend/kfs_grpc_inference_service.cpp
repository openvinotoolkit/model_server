//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "kfs_grpc_inference_service.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../dags/pipeline.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../dags/pipelinedefinitionstatus.hpp"
#include "../dags/pipelinedefinitionunloadguard.hpp"
#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../grpc_utils.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../mediapipe_internal/mediapipedemo.hpp"
#include "../metric.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../ovinferrequestsqueue.hpp"
#include "../prediction_service_utils.hpp"
#include "../serialization.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};
}

namespace ovms {

Status KFSInferenceServiceImpl::getModelInstance(const KFSRequest* request,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    model_version_t requestedVersion = 0;
    if (!request->model_version().empty()) {
        auto versionRead = stoi64(request->model_version());
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("requested model: name {}; with version in invalid format: {}", request->model_name(), request->model_version());
            return StatusCode::MODEL_VERSION_INVALID_FORMAT;
        }
    }
    return this->modelManager.getModelInstance(request->model_name(), requestedVersion, modelInstance, modelInstanceUnloadGuardPtr);
}

Status KFSInferenceServiceImpl::getPipeline(const KFSRequest* request,
    KFSResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    return this->modelManager.createPipeline(pipelinePtr, request->model_name(), request, response);
}

const std::string PLATFORM = "OpenVINO";

::grpc::Status KFSInferenceServiceImpl::ServerLive(::grpc::ServerContext* context, const ::inference::ServerLiveRequest* request, ::inference::ServerLiveResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    bool isLive = this->ovmsServer.isLive();
    SPDLOG_DEBUG("Requested Server liveness state: {}", isLive);
    response->set_live(isLive);
    return grpc::Status::OK;
}

::grpc::Status KFSInferenceServiceImpl::ServerReady(::grpc::ServerContext* context, const ::inference::ServerReadyRequest* request, ::inference::ServerReadyResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    bool isReady = this->ovmsServer.isReady();
    SPDLOG_DEBUG("Requested Server readiness state: {}", isReady);
    response->set_ready(isReady);
    return grpc::Status::OK;
}

Status KFSInferenceServiceImpl::getModelReady(const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response, const ModelManager& manager, ExecutionContext executionContext) {
    // Return in response true/false
    // if no version requested give response for default version
    const auto& name = request->name();
    const auto& versionString = request->version();
    auto model = manager.findModelByName(name);
    SPDLOG_DEBUG("ModelReady requested name: {}, version: {}", name, versionString);
    if (model == nullptr) {
        SPDLOG_DEBUG("ModelReady requested model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return Status(StatusCode::MODEL_NAME_MISSING);
        }
        auto status = buildResponse(*pipelineDefinition, response);
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getModelReadyMetric(executionContext, status.ok()));
        return status;
    }
    std::shared_ptr<ModelInstance> instance = nullptr;
    if (!versionString.empty()) {
        SPDLOG_DEBUG("ModelReady requested model: name {}; version {}", name, versionString);
        model_version_t requestedVersion = 0;
        auto versionRead = stoi64(versionString);
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("ModelReady requested model: name {}; with version in invalid format: {}", name, versionString);
            return Status(StatusCode::MODEL_VERSION_INVALID_FORMAT);
        }
        instance = model->getModelInstanceByVersion(requestedVersion);
        if (instance == nullptr) {
            SPDLOG_DEBUG("ModelReady requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    } else {
        SPDLOG_DEBUG("ModelReady requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("ModelReady requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    }
    auto status = buildResponse(instance, response);
    INCREMENT_IF_ENABLED(instance->getMetricReporter().getModelReadyMetric(executionContext, status.ok()));
    return status;
}

::grpc::Status KFSInferenceServiceImpl::ModelReady(::grpc::ServerContext* context, const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response) {
    return grpc(ModelReadyImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelReady}));
}

Status KFSInferenceServiceImpl::ModelReadyImpl(::grpc::ServerContext* context, const KFSGetModelStatusRequest* request, KFSGetModelStatusResponse* response, ExecutionContext executionContext) {
    (void)context;
    return this->getModelReady(request, response, this->modelManager, executionContext);
}

::grpc::Status KFSInferenceServiceImpl::ServerMetadata(::grpc::ServerContext* context, const KFSServerMetadataRequest* request, KFSServerMetadataResponse* response) {
    return grpc(ServerMetadataImpl(context, request, response));
}

Status KFSInferenceServiceImpl::ServerMetadataImpl(::grpc::ServerContext* context, const KFSServerMetadataRequest* request, KFSServerMetadataResponse* response) {
    (void)context;
    (void)request;
    (void)response;
    response->set_name(PROJECT_NAME);
    response->set_version(PROJECT_VERSION);
    return StatusCode::OK;
}

::grpc::Status KFSInferenceServiceImpl::ModelMetadata(::grpc::ServerContext* context, const KFSModelMetadataRequest* request, KFSModelMetadataResponse* response) {
    return grpc(ModelMetadataImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelMetadata}));
}

Status KFSInferenceServiceImpl::ModelMetadataImpl(::grpc::ServerContext* context, const KFSModelMetadataRequest* request, KFSModelMetadataResponse* response, ExecutionContext executionContext) {
    const auto& name = request->name();
    const auto& versionString = request->version();

    auto model = this->modelManager.findModelByName(name);
    if (model == nullptr) {
        SPDLOG_DEBUG("GetModelMetadata: Model {} is missing, trying to find pipeline with such name", name);
        auto pipelineDefinition = this->modelManager.getPipelineFactory().findDefinitionByName(name);
        if (!pipelineDefinition) {
            return Status(StatusCode::MODEL_NAME_MISSING);
        }
        auto status = buildResponse(*pipelineDefinition, response);
        INCREMENT_IF_ENABLED(pipelineDefinition->getMetricReporter().getModelMetadataMetric(executionContext, status.ok()));
        return status;
    }
    std::shared_ptr<ModelInstance> instance = nullptr;
    if (!versionString.empty()) {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; version {}", name, versionString);
        model_version_t requestedVersion = 0;
        auto versionRead = stoi64(versionString);
        if (versionRead) {
            requestedVersion = versionRead.value();
        } else {
            SPDLOG_DEBUG("GetModelMetadata requested model: name {}; with version in invalid format: {}", name, versionString);
            return Status(StatusCode::MODEL_VERSION_INVALID_FORMAT);
        }
        instance = model->getModelInstanceByVersion(requestedVersion);
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    } else {
        SPDLOG_DEBUG("GetModelMetadata requested model: name {}; default version", name);
        instance = model->getDefaultModelInstance();
        if (instance == nullptr) {
            SPDLOG_DEBUG("GetModelMetadata requested model {}; version {} is missing", name, versionString);
            return Status(StatusCode::MODEL_VERSION_MISSING);
        }
    }
    auto status = buildResponse(*model, *instance, response);
    INCREMENT_IF_ENABLED(instance->getMetricReporter().getModelMetadataMetric(executionContext, status.ok()));

    return status;
}

::grpc::Status KFSInferenceServiceImpl::ModelInfer(::grpc::ServerContext* context, const KFSRequest* request, KFSResponse* response) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
    SPDLOG_DEBUG("Processing gRPC request for model: {}; version: {}",
        request->model_name(),
        request->model_version());
    ServableMetricReporter* reporter = nullptr;
    auto status = this->ModelInferImpl(context, request, response, ExecutionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelInfer}, reporter);
    timer.stop(TOTAL);
    if (!status.ok()) {
        return grpc(status);
    }
    const std::string ServableName = request->model_name();
    if (0 == ServableName.rfind("mediapipe", 0)) {
        return grpc(Status(StatusCode::OK));
    }
    if (!reporter) {
        SPDLOG_ERROR("If this is mediapipe test you need to exclude it from this check");
        return grpc(Status(StatusCode::INTERNAL_ERROR));  // should not happen
    }
    double requestTotal = timer.elapsed<std::chrono::microseconds>(TOTAL);
    SPDLOG_DEBUG("Total gRPC request processing time: {} ms", requestTotal / 1000);
    OBSERVE_IF_ENABLED(reporter->requestTimeGrpc, requestTotal);
    return grpc(status);
}

static ov::Tensor bruteForceDeserialize(const std::string& requestedName, const KFSRequest* request) {
    auto requestInputItr = std::find_if(request->inputs().begin(), request->inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request->inputs().end()) {
        return ov::Tensor();  // TODO
    }
    auto inputIndex = requestInputItr - request->inputs().begin();
    bool deserializeFromSharedInputContents = request->raw_input_contents().size() > 0;
    auto bufferLocation = deserializeFromSharedInputContents ? &request->raw_input_contents()[inputIndex] : nullptr;
    if (!bufferLocation)
        throw 42;
    ov::Shape shape;
    for (int i = 0; i < requestInputItr->shape().size(); i++) {
        shape.push_back(requestInputItr->shape()[i]);
    }
    ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
    // TODO handle both KFS input handling ways
    try {
        auto outTensor = ov::Tensor(precision, shape, const_cast<void*>((const void*)bufferLocation->c_str()));
        return outTensor;
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Kserve mediapipe request deserialization failed:{}", e.what());
    } catch (...) {
        SPDLOG_DEBUG("KServe mediapipe request deserialization failed");
    }
    return ov::Tensor();
}

class MediapipeGraphExecutor {
    ::mediapipe::CalculatorGraph graph;
    ::mediapipe::CalculatorGraphConfig config;
    const std::string OUTPUT_NAME = "out";
    const std::string INPUT_NAME = "in";
    const std::string servableName;
    const uint16_t SERVABLE_VERSION = 1;

public:
    MediapipeGraphExecutor(const std::string servableName) :
        servableName(servableName) {
        std::string chosenConfig;
        if (servableName == "mediapipeDummy") {
            chosenConfig = DUMMY_MEDIAPIPE_GRAPH;
        } else if (servableName == "mediapipeAdd") {
            chosenConfig = ADD_MEDIAPIPE_GRAPH;
        } else if (servableName == "mediapipeDummyADAPT") {
            chosenConfig = DUMMY_MEDIAPIPE_GRAPH_ADAPT;
        } else if (servableName == "mediapipeAddADAPT") {
            chosenConfig = ADD_MEDIAPIPE_GRAPH_ADAPT;
        } else if (servableName == "mediapipeAddADAPTFULL") {
            chosenConfig = ADD_MEDIAPIPE_GRAPH_ADAPT_FULL;
        } else {
            throw 42;  // FIXME
        }
        config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(chosenConfig);
    }
    Status infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) {
        SPDLOG_DEBUG("Start KServe request mediapipe graph:{} execution", request->model_name());
        auto ret = graph.Initialize(config);
        if (!ret.ok()) {
            SPDLOG_DEBUG("KServe request for mediapipe graph:{} execution failed with message: {}", request->model_name(), ret.message());
        }

        std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
        // TODO validate number of inputs
        // TODO validate input names against input streams
        //       std::vector<std::string> outputNames{OUTPUT_NAME};
        auto outputNames = config.output_stream();
        for (auto name : outputNames) {
            auto absStatus = graph.AddOutputStreamPoller(OUTPUT_NAME);
            if (!absStatus.ok()) {
                return StatusCode::NOT_IMPLEMENTED;
            }
            outputPollers.emplace(name, std::move(absStatus).value());
        }
        auto inputNames = config.input_stream();
        auto ret2 = graph.StartRun({});  // TODO retcode
        if (!ret2.ok()) {
            SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), ret2.message());
            throw 43;  // TODO retcode
        }
        for (auto name : inputNames) {
            SPDLOG_TRACE("Tensor to deserialize:\"{}\"", name);
            ov::Tensor input_tensor = bruteForceDeserialize(name, request);
            auto abstatus = graph.AddPacketToInputStream(
                name, ::mediapipe::MakePacket<ov::Tensor>(std::move(input_tensor)).At(::mediapipe::Timestamp(0)));
            if (!abstatus.ok()) {
                SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {}. Error message: {}, error code: {}",
                    name, request->model_name(), abstatus.message(), abstatus.raw_code());
                throw 44;
            }
            abstatus = graph.CloseInputStream(name);  // TODO retcode
            if (!abstatus.ok()) {
                SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {}. Error message: {}, error code: {}",
                    name, request->model_name(), abstatus.message(), abstatus.raw_code());
                throw 45;
            }
            SPDLOG_TRACE("Tensor to deserialize:\"{}\"", name);
        }
        // receive outputs
        ::mediapipe::Packet packet;
        for (auto& [outputStreamName, poller] : outputPollers) {
            SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
            while (poller.Next(&packet)) {
                SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
                auto received = packet.Get<ov::Tensor>();
                float* dataOut = (float*)received.data();
                auto timestamp = packet.Timestamp();
                std::stringstream ss;
                ss << "ServiceImpl Received tensor: [";
                for (int x = 0; x < 10; ++x) {
                    ss << dataOut[x] << " ";
                }
                ss << " ]  timestamp: " << timestamp.DebugString();
                SPDLOG_DEBUG(ss.str());
                auto* output = response->add_outputs();
                output->clear_shape();
                output->set_name(outputStreamName);
                auto* outputContentString = response->add_raw_output_contents();
                auto ovDtype = received.get_element_type();
                auto outputDtype = ovmsPrecisionToKFSPrecision(ovElementTypeToOvmsPrecision(ovDtype));
                output->set_datatype(outputDtype);
                auto shape = received.get_shape();
                for (const auto& dim : shape) {
                    output->add_shape(dim);
                }
                float* data = (float*)received.data();
                std::stringstream ss2;
                ss2 << "[ ";
                for (size_t i = 0; i < 10; ++i) {
                    ss2 << data[i] << " ";
                }
                ss2 << "]";
                SPDLOG_DEBUG("ServiceImpl OutputData: {}", ss2.str());
                outputContentString->assign((char*)received.data(), received.get_byte_size());
            }
        }
        SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
        response->set_model_name(servableName);
        response->set_id("1");  // TODO later
        response->set_model_version(std::to_string(SERVABLE_VERSION));
        return StatusCode::OK;
    }
};

Status KFSInferenceServiceImpl::ModelInferImpl(::grpc::ServerContext* context, const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) {
    OVMS_PROFILE_FUNCTION();
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(request, modelInstance, modelInstanceUnloadGuard);
    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", request->model_name());
        status = getPipeline(request, response, pipelinePtr);
    }
    if (!status.ok()) {
        const std::string ServableName = request->model_name();
        if (modelInstance) {
            INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailGrpcModelInfer);
        } else if (0 == ServableName.rfind("mediapipe", 0)) {
            // TODO mediapipe metrics
            MediapipeGraphExecutor executor(request->model_name());
            auto status = executor.infer(request, response, executionContext, reporterOut);
            return status;
        }
        SPDLOG_DEBUG("Getting modelInstance or pipeline failed. {}", status.string());
        return status;
    }
    if (pipelinePtr) {
        reporterOut = &pipelinePtr->getMetricReporter();
        status = pipelinePtr->execute(executionContext);
    } else if (modelInstance) {
        reporterOut = &modelInstance->getMetricReporter();
        status = modelInstance->infer(request, response, modelInstanceUnloadGuard);
    }
    INCREMENT_IF_ENABLED(reporterOut->getInferRequestMetric(executionContext, status.ok()));
    if (!status.ok()) {
        return status;
    }
    response->set_id(request->id());
    return StatusCode::OK;
}

Status KFSInferenceServiceImpl::buildResponse(
    std::shared_ptr<ModelInstance> instance,
    KFSGetModelStatusResponse* response) {
    response->set_ready(instance->getStatus().getState() == ModelVersionState::AVAILABLE);
    return StatusCode::OK;
}

Status KFSInferenceServiceImpl::buildResponse(
    PipelineDefinition& pipelineDefinition,
    KFSGetModelStatusResponse* response) {
    response->set_ready(pipelineDefinition.getStatus().isAvailable());
    return StatusCode::OK;
}

static void addReadyVersions(Model& model,
    KFSModelMetadataResponse* response) {
    auto modelVersions = model.getModelVersionsMapCopy();
    for (auto& [modelVersion, modelInstance] : modelVersions) {
        if (modelInstance.getStatus().getState() == ModelVersionState::AVAILABLE)
            response->add_versions(std::to_string(modelVersion));
    }
}

Status KFSInferenceServiceImpl::buildResponse(
    Model& model,
    ModelInstance& instance,
    KFSModelMetadataResponse* response) {

    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = instance.waitForLoaded(0, unloadGuard);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->set_name(instance.getName());
    addReadyVersions(model, response);
    response->set_platform(PLATFORM);

    for (const auto& input : instance.getInputsInfo()) {
        convert(input, response->add_inputs());
    }

    for (const auto& output : instance.getOutputsInfo()) {
        convert(output, response->add_outputs());
    }

    return StatusCode::OK;
}

KFSInferenceServiceImpl::KFSInferenceServiceImpl(const Server& server) :
    ovmsServer(server),
    modelManager(dynamic_cast<const ServableManagerModule*>(this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME)) {
        const char* message = "Tried to create kserve inference service impl without servable manager module";
        SPDLOG_ERROR(message);
        throw std::logic_error(message);
    }
}

Status KFSInferenceServiceImpl::buildResponse(
    PipelineDefinition& pipelineDefinition,
    KFSModelMetadataResponse* response) {

    std::unique_ptr<PipelineDefinitionUnloadGuard> unloadGuard;

    // 0 meaning immediately return unload guard if possible, otherwise do not wait for available state
    auto status = pipelineDefinition.waitForLoaded(unloadGuard, 0);
    if (!status.ok()) {
        return status;
    }

    response->Clear();
    response->set_name(pipelineDefinition.getName());
    response->add_versions("1");
    response->set_platform(PLATFORM);

    for (const auto& input : pipelineDefinition.getInputsInfo()) {
        convert(input, response->add_inputs());
    }

    for (const auto& output : pipelineDefinition.getOutputsInfo()) {
        convert(output, response->add_outputs());
    }

    return StatusCode::OK;
}

void KFSInferenceServiceImpl::convert(
    const std::pair<std::string, std::shared_ptr<TensorInfo>>& from,
    KFSModelMetadataResponse::TensorMetadata* to) {
    to->set_name(from.first);
    to->set_datatype(ovmsPrecisionToKFSPrecision(from.second->getPrecision()));
    for (auto& dim : from.second->getShape()) {
        if (dim.isStatic()) {
            to->add_shape(dim.getStaticValue());
        } else {
            to->add_shape(DYNAMIC_DIMENSION);
        }
    }
}

}  // namespace ovms
