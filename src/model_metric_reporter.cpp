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

ServableMetricReporter::~ServableMetricReporter() {
    if (!this->registry) {
        return;
    }

    if (this->requestSuccessFamily) {
        this->registry->remove(this->requestSuccessFamily);
    }
    if (this->requestFailFamily) {
        this->registry->remove(this->requestFailFamily);
    }
    if (this->requestTimeFamily) {
        this->registry->remove(this->requestTimeFamily);
    }
}

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

    std::string familyName = "ovms_requests_success";
    this->requestSuccessFamily = registry->createFamily<MetricCounter>(familyName,
        "Number of successful requests to a model or a DAG.");
    THROW_IF_NULL(this->requestSuccessFamily, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        SPDLOG_INFO("Creating ovms_requests_success");
        // TFS
        this->requestSuccessGrpcPredict = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcPredict, "cannot create metric");

        this->requestSuccessGrpcGetModelMetadata = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcGetModelMetadata, "cannot create metric");

        this->requestSuccessGrpcGetModelStatus = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcGetModelStatus, "cannot create metric");

        this->requestSuccessRestPredict = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestPredict, "cannot create metric");

        this->requestSuccessRestGetModelMetadata = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestGetModelMetadata, "cannot create metric");

        this->requestSuccessRestGetModelStatus = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestGetModelStatus, "cannot create metric");

        // KFS
        this->requestSuccessGrpcModelInfer = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelInfer, "cannot create metric");

        this->requestSuccessGrpcModelMetadata = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelMetadata, "cannot create metric");

        this->requestSuccessGrpcModelReady = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestSuccessGrpcModelReady, "cannot create metric");

        this->requestSuccessRestModelInfer = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelInfer, "cannot create metric");

        this->requestSuccessRestModelMetadata = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelMetadata, "cannot create metric");

        this->requestSuccessRestModelReady = this->requestSuccessFamily->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestSuccessRestModelReady, "cannot create metric");
    }

    familyName = "ovms_requests_fail";
    this->requestFailFamily = registry->createFamily<MetricCounter>(familyName,
        "Number of failed requests to a model or a DAG.");
    THROW_IF_NULL(this->requestFailFamily, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        SPDLOG_INFO("Creating ovms_requests_fail");
        // TFS
        this->requestFailGrpcPredict = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcPredict, "cannot create metric");

        this->requestFailGrpcGetModelMetadata = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcGetModelMetadata, "cannot create metric");

        this->requestFailGrpcGetModelStatus = this->requestFailFamily->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcGetModelStatus, "cannot create metric");

        this->requestFailRestPredict = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestPredict, "cannot create metric");

        this->requestFailRestGetModelMetadata = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestGetModelMetadata, "cannot create metric");

        this->requestFailRestGetModelStatus = this->requestFailFamily->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestGetModelStatus, "cannot create metric");

        // KFS
        this->requestFailGrpcModelInfer = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelInfer, "cannot create metric");

        this->requestFailGrpcModelMetadata = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelMetadata, "cannot create metric");

        this->requestFailGrpcModelReady = this->requestFailFamily->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});
        THROW_IF_NULL(this->requestFailGrpcModelReady, "cannot create metric");

        this->requestFailRestModelInfer = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelInfer, "cannot create metric");

        this->requestFailRestModelMetadata = this->requestFailFamily->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelMetadata, "cannot create metric");

        this->requestFailRestModelReady = this->requestFailFamily->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
        THROW_IF_NULL(this->requestFailRestModelReady, "cannot create metric");
    }

    familyName = "ovms_request_time_us";
    this->requestTimeFamily = registry->createFamily<MetricHistogram>(familyName,
        "Processing time of requests to a model or a DAG.");
    THROW_IF_NULL(this->requestTimeFamily, "cannot create family");

    if (metricConfig->isFamilyEnabled(familyName)) {
        this->requestTimeGrpc = this->requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "gRPC"}},
            this->buckets);
        THROW_IF_NULL(this->requestTimeGrpc, "cannot create metric");

        this->requestTimeRest = this->requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "REST"}},
            this->buckets);
        THROW_IF_NULL(this->requestTimeRest, "cannot create metric");
    }
}

ModelMetricReporter::~ModelMetricReporter() {
    if (!this->registry) {
        return;
    }

    if (this->inferenceTimeFamily) {
        this->registry->remove(this->inferenceTimeFamily);
    }
    if (this->waitForInferReqTimeFamily) {
        this->registry->remove(this->waitForInferReqTimeFamily);
    }
    if (this->streamsFamily) {
        this->registry->remove(this->streamsFamily);
    }
    if (this->inferReqQueueSizeFamily) {
        this->registry->remove(this->inferReqQueueSizeFamily);
    }
    if (this->inferReqActiveFamily) {
        this->registry->remove(this->inferReqActiveFamily);
    }
    if (this->currentRequestsFamily) {
        this->registry->remove(this->currentRequestsFamily);
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

    std::string familyName = "ovms_inference_time_us";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->inferenceTimeFamily = registry->createFamily<MetricHistogram>(familyName,
            "Inference execution time in the OpenVINO backend.");
        THROW_IF_NULL(this->inferenceTimeFamily, "cannot create family");
        this->inferenceTime = this->inferenceTimeFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
            this->buckets);
        THROW_IF_NULL(this->inferenceTime, "cannot create metric");
    }

    familyName = "ovms_wait_for_infer_req_time_us";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->waitForInferReqTimeFamily = registry->createFamily<MetricHistogram>(familyName,
            "Request waiting time in the scheduling queue.");
        THROW_IF_NULL(this->waitForInferReqTimeFamily, "cannot create family");
        this->waitForInferReqTime = this->waitForInferReqTimeFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
            this->buckets);
        THROW_IF_NULL(this->waitForInferReqTime, "cannot create metric");
    }

    familyName = "ovms_streams";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->streamsFamily = registry->createFamily<MetricGauge>(familyName,
            "Number of OpenVINO execution streams.");
        THROW_IF_NULL(this->streamsFamily, "cannot create family");
        this->streams = this->streamsFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->streams, "cannot create metric");
    }

    familyName = "ovms_infer_req_queue_size";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->inferReqQueueSizeFamily = registry->createFamily<MetricGauge>(familyName,
            "Inference request queue size (nireq).");
        THROW_IF_NULL(this->inferReqQueueSizeFamily, "cannot create family");
        this->inferReqQueueSize = this->inferReqQueueSizeFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->inferReqQueueSize, "cannot create metric");
    }

    familyName = "ovms_infer_req_active";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->inferReqActiveFamily = registry->createFamily<MetricGauge>(familyName,
            "Number of currently consumed inference request from the processing queue.");
        THROW_IF_NULL(this->inferReqActiveFamily, "cannot create family");
        this->inferReqActive = this->inferReqActiveFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->inferReqActive, "cannot create metric");
    }

    familyName = "ovms_current_requests";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->currentRequestsFamily = registry->createFamily<MetricGauge>(familyName,
            "Number of inference requests currently in process.");
        THROW_IF_NULL(this->currentRequestsFamily, "cannot create family");
        this->currentRequests = this->currentRequestsFamily->addMetric(
            {{"name", modelName}, {"version", std::to_string(modelVersion)}});
        THROW_IF_NULL(this->currentRequests, "cannot create metric");
    }
}

}  // namespace ovms
