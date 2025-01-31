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
#include "model_metric_reporter.hpp"

#include <cmath>
#include <exception>

#include "execution_context.hpp"
#include "logging.hpp"
#include "metric_config.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {

constexpr int NUMBER_OF_BUCKETS = 33;
constexpr double BUCKET_POWER_BASE = 1.8;
constexpr double BUCKET_MULTIPLIER = 10;

#define THROW_IF_NULL(VAR, MESSAGE)                        \
    if (VAR == nullptr) {                                  \
        SPDLOG_LOGGER_ERROR(modelmanager_logger, MESSAGE); \
        throw std::logic_error(MESSAGE);                   \
    }

ServableMetricReporter::~ServableMetricReporter() = default;

ServableMetricReporter::ServableMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion) :
    registry(registry) {
    if (!registry) {
        return;
    }

    if (!metricConfig || !metricConfig->metricsEnabled) {
        return;
    }

    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        this->buckets.emplace_back(floor(BUCKET_MULTIPLIER * pow(BUCKET_POWER_BASE, i)));
    }

    std::string familyName = METRIC_NAME_REQUESTS_SUCCESS;
    auto family = registry->createFamily<MetricCounter>(familyName,
        "Number of successful requests to a model or a DAG.");
    THROW_IF_NULL(family, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        // TFS
        this->requestSuccessGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcPredict, "cannot create metric");

        this->requestSuccessGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcGetModelMetadata, "cannot create metric");

        this->requestSuccessGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcGetModelStatus, "cannot create metric");

        this->requestSuccessRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestPredict, "cannot create metric");

        this->requestSuccessRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestGetModelMetadata, "cannot create metric");

        this->requestSuccessRestGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestGetModelStatus, "cannot create metric");

        // KFS
        this->requestSuccessGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelInfer, "cannot create metric");

        this->requestSuccessGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelMetadata, "cannot create metric");

        this->requestSuccessGrpcModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelReady, "cannot create metric");

        this->requestSuccessRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelInfer, "cannot create metric");

        this->requestSuccessRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelMetadata, "cannot create metric");

        this->requestSuccessRestModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelReady, "cannot create metric");
    }

    familyName = METRIC_NAME_REQUESTS_FAIL;
    family = registry->createFamily<MetricCounter>(familyName,
        "Number of failed requests to a model or a DAG.");
    THROW_IF_NULL(family, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        // TFS
        this->requestFailGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcPredict, "cannot create metric");

        this->requestFailGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcGetModelMetadata, "cannot create metric");

        this->requestFailGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcGetModelStatus, "cannot create metric");

        this->requestFailRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestPredict, "cannot create metric");

        this->requestFailRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestGetModelMetadata, "cannot create metric");

        this->requestFailRestGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestGetModelStatus, "cannot create metric");

        // KFS
        this->requestFailGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelInfer, "cannot create metric");

        this->requestFailGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelMetadata, "cannot create metric");

        this->requestFailGrpcModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelReady, "cannot create metric");

        this->requestFailRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelInfer, "cannot create metric");

        this->requestFailRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelMetadata, "cannot create metric");

        this->requestFailRestModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelReady, "cannot create metric");
    }

    familyName = METRIC_NAME_REQUEST_TIME;
    auto requestTimeFamily = registry->createFamily<MetricHistogram>(familyName,
        "Processing time of requests to a model or a DAG.");
    THROW_IF_NULL(requestTimeFamily, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        this->requestTimeGrpc = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "gRPC"}},
            this->buckets);
        THROW_IF_NULL(this->requestTimeGrpc, "cannot create metric");

        this->requestTimeRest = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "REST"}},
            this->buckets);
        THROW_IF_NULL(this->requestTimeRest, "cannot create metric");
    }
}

ModelMetricReporter::ModelMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion) :
    ServableMetricReporter(metricConfig, registry, modelName, modelVersion) {
    if (!registry) {
        return;
    }

    if (!metricConfig || !metricConfig->metricsEnabled) {
        return;
    }

    std::string familyName = METRIC_NAME_INFERENCE_TIME;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricHistogram>(familyName,
            "Inference execution time in the OpenVINO backend.");
        THROW_IF_NULL(family, "cannot create family");
        this->inferenceTime = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
            this->buckets);
        THROW_IF_NULL(this->inferenceTime, "cannot create metric");
    }

    familyName = METRIC_NAME_WAIT_FOR_INFER_REQ_TIME;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricHistogram>(familyName,
            "Request waiting time in the scheduling queue.");
        THROW_IF_NULL(family, "cannot create family");
        this->waitForInferReqTime = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
            this->buckets);
        THROW_IF_NULL(this->waitForInferReqTime, "cannot create metric");
    }

    familyName = METRIC_NAME_STREAMS;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricGauge>(familyName,
            "Number of OpenVINO execution streams.");
        THROW_IF_NULL(family, "cannot create family");
        this->streams = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->streams, "cannot create metric");
    }

    familyName = METRIC_NAME_INFER_REQ_QUEUE_SIZE;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricGauge>(familyName,
            "Inference request queue size (nireq).");
        THROW_IF_NULL(family, "cannot create family");
        this->inferReqQueueSize = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->inferReqQueueSize, "cannot create metric");
    }

    familyName = METRIC_NAME_INFER_REQ_ACTIVE;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricGauge>(familyName,
            "Number of currently consumed inference request from the processing queue.");
        THROW_IF_NULL(family, "cannot create family");
        this->inferReqActive = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->inferReqActive, "cannot create metric");
    }

    familyName = METRIC_NAME_CURRENT_REQUESTS;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricGauge>(familyName,
            "Number of inference requests currently in process.");
        THROW_IF_NULL(family, "cannot create family");
        this->currentRequests = family->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->currentRequests, "cannot create metric");
    }
}

MediapipeServableMetricReporter::MediapipeServableMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& graphName) :
    registry(registry) {
    if (!registry) {
        return;
    }

    if (!metricConfig || !metricConfig->metricsEnabled) {
        return;
    }

    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        this->buckets.emplace_back(floor(BUCKET_MULTIPLIER * pow(BUCKET_POWER_BASE, i)));
    }

    auto familyName = METRIC_NAME_CURRENT_GRAPHS;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricGauge>(familyName,
            "Number of MediaPipe graphs in process.");
        THROW_IF_NULL(family, "cannot create family");
        this->currentGraphs = family->addMetric(
            {{"name", graphName}});
        THROW_IF_NULL(this->currentGraphs, "cannot create metric");
    } else {
        SPDLOG_INFO("DISABLED {}", METRIC_NAME_CURRENT_GRAPHS);
    }

    familyName = METRIC_NAME_REQUESTS_ACCEPTED;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of accepted requests which ended up inserting packet(s) into the MediaPipe graph.");
        THROW_IF_NULL(family, "cannot create family");

        // KFS
        this->requestAcceptedGrpcModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestAcceptedGrpcModelInfer, "cannot create metric");

        this->requestAcceptedGrpcModelInferStream = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInferStream"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestAcceptedGrpcModelInfer, "cannot create metric");

        this->requestAcceptedRestModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestAcceptedRestModelInfer, "cannot create metric");

        this->requestAcceptedRestV3Unary = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Unary"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestAcceptedRestV3Unary, "cannot create metric");

        this->requestAcceptedRestV3Stream = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Stream"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestAcceptedRestV3Stream, "cannot create metric");
    }

    familyName = METRIC_NAME_REQUESTS_REJECTED;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of rejected requests which did not end up being inserted into the MediaPipe graph.");
        THROW_IF_NULL(family, "cannot create family");

        // KFS
        this->requestRejectedGrpcModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestRejectedGrpcModelInfer, "cannot create metric");

        this->requestRejectedGrpcModelInferStream = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInferStream"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestRejectedGrpcModelInfer, "cannot create metric");

        this->requestRejectedRestModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestRejectedRestModelInfer, "cannot create metric");

        this->requestRejectedRestV3Unary = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Unary"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestRejectedRestV3Unary, "cannot create metric");

        this->requestRejectedRestV3Stream = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Stream"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestRejectedRestV3Stream, "cannot create metric");
    }
    familyName = METRIC_NAME_GRAPH_ERROR;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of errors generated by the MediaPipe graph.");
        THROW_IF_NULL(family, "cannot create family");
        this->requestErrorGrpcModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestErrorGrpcModelInfer, "cannot create metric");
        this->requestErrorGrpcModelInferStream = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInferStream"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestErrorGrpcModelInferStream, "cannot create metric");
        this->requestErrorRestModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestErrorRestModelInfer, "cannot create metric");
        this->requestErrorRestV3Unary = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Unary"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestErrorRestV3Unary, "cannot create metric");
        this->requestErrorRestV3Stream = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Stream"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestErrorRestV3Stream, "cannot create metric");
    }

    familyName = METRIC_NAME_RESPONSES;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of responses generated the MediaPipe graph.");
        THROW_IF_NULL(family, "cannot create family");

        // KFS
        this->responseGrpcModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->responseGrpcModelInfer, "cannot create metric");

        this->responseGrpcModelInferStream = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInferStream"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->responseGrpcModelInfer, "cannot create metric");

        this->responseRestModelInfer = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->responseRestModelInfer, "cannot create metric");

        // V3
        this->responseRestV3Unary = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Unary"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->responseRestV3Unary, "cannot create metric");

        this->responseRestV3Stream = family->addMetric({{"name", graphName},
            {"api", "V3"},
            {"method", "Stream"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->responseRestV3Stream, "cannot create metric");
    }

    familyName = METRIC_NAME_REQUESTS_FAIL;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of failed requests to a mediapipe.");
        THROW_IF_NULL(family, "cannot create family");

        this->requestFailGrpcModelMetadata = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelMetadata, "cannot create metric");

        this->requestFailGrpcModelReady = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelReady, "cannot create metric");

        this->requestFailRestModelMetadata = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelMetadata, "cannot create metric");

        this->requestFailRestModelReady = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelReady, "cannot create metric");
    }

    familyName = METRIC_NAME_REQUESTS_SUCCESS;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricCounter>(familyName,
            "Number of successful requests to a mediapipe.");
        THROW_IF_NULL(family, "cannot create family");

        this->requestSuccessGrpcModelMetadata = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelMetadata, "cannot create metric");

        this->requestSuccessGrpcModelReady = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelReady, "cannot create metric");

        this->requestSuccessRestModelMetadata = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelMetadata, "cannot create metric");

        this->requestSuccessRestModelReady = family->addMetric({{"name", graphName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelReady, "cannot create metric");
    }
    familyName = METRIC_NAME_PROCESSING_TIME;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricHistogram>(familyName,
            "Time packet spent in the MediaPipe graph.");
        THROW_IF_NULL(family, "cannot create family");

        // KFS
        this->processingTimeGrpcModelInfer = family->addMetric({{"name", graphName},
                                                                   {"method", "ModelInfer"}},
            this->buckets);
        THROW_IF_NULL(this->processingTimeGrpcModelInfer, "cannot create metric");

        this->processingTimeGrpcModelInferStream = family->addMetric({{"name", graphName},
                                                                         {"method", "ModelInferStream"}},
            this->buckets);
        THROW_IF_NULL(this->processingTimeGrpcModelInfer, "cannot create metric");

        // V3
        this->processingTimeRestV3Unary = family->addMetric({{"name", graphName},
                                                                {"method", "Unary"}},
            this->buckets);
        THROW_IF_NULL(this->processingTimeRestV3Unary, "cannot create metric");

        this->processingTimeRestV3Stream = family->addMetric({{"name", graphName},
                                                                 {"method", "Stream"}},
            this->buckets);
        THROW_IF_NULL(this->processingTimeRestV3Stream, "cannot create metric");
    }
    familyName = METRIC_NAME_REQUEST_LATENCY;
    if (metricConfig->isFamilyEnabled(familyName)) {
        auto family = registry->createFamily<MetricHistogram>(familyName,
            "Time difference between incoming request and output packet in mediapipe graph.");
        THROW_IF_NULL(family, "cannot create family");

        // KFS
        this->requestLatencyGrpcModelInferStream = family->addMetric({{"name", graphName},
                                                                         {"method", "ModelInferStream"}},
            this->buckets);
        THROW_IF_NULL(this->requestLatencyGrpcModelInferStream, "cannot create metric");
        // V3
        this->requestLatencyRestV3Stream = family->addMetric({{"name", graphName},
                                                                 {"method", "Stream"}},
            this->buckets);
        THROW_IF_NULL(this->requestLatencyRestV3Stream, "cannot create metric");
    }
}

}  // namespace ovms
