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

#include "execution_context.hpp"
#include "metric_config.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {

constexpr int NUMBER_OF_BUCKETS = 33;
constexpr double BUCKET_POWER_BASE = 1.8;
constexpr double BUCKET_MULTIPLIER = 10;

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
    auto family = registry->createFamily<MetricCounter>(familyName,
        "Number of successful requests to a model or a DAG.");

    if (metricConfig->isFamilyEnabled(familyName)) {
        // TFS
        this->requestSuccessGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});

        this->requestSuccessGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});

        this->requestSuccessGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});

        this->requestSuccessRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});

        this->requestSuccessRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});

        this->requestSuccessRestGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});
        // KFS
        this->requestSuccessGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});

        this->requestSuccessGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});

        this->requestSuccessGrpcModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});

        this->requestSuccessRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});

        this->requestSuccessRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});

        this->requestSuccessRestModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
    }

    familyName = "ovms_requests_fail";
    family = registry->createFamily<MetricCounter>(familyName,
        "Number of failed requests to a model or a DAG.");

    if (metricConfig->isFamilyEnabled(familyName)) {
        // TFS
        this->requestFailGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "gRPC"}});

        this->requestFailGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "gRPC"}});

        this->requestFailGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "gRPC"}});

        this->requestFailRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "Predict"},
            {"interface", "REST"}});

        this->requestFailRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "TensorFlowServing"},
            {"method", "GetModelMetadata"},
            {"interface", "REST"}});

        this->requestFailRestGetModelStatus = family->addMetric({{"name", modelName},
            {"api", "TensorFlowServing"},
            {"method", "GetModelStatus"},
            {"interface", "REST"}});

        // KFS
        this->requestFailGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "gRPC"}});

        this->requestFailGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "gRPC"}});

        this->requestFailGrpcModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "gRPC"}});

        this->requestFailRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelInfer"},
            {"interface", "REST"}});

        this->requestFailRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"api", "KServe"},
            {"method", "ModelMetadata"},
            {"interface", "REST"}});

        this->requestFailRestModelReady = family->addMetric({{"name", modelName},
            {"api", "KServe"},
            {"method", "ModelReady"},
            {"interface", "REST"}});
    }

    familyName = "ovms_request_time_us";
    auto requestTimeFamily = registry->createFamily<MetricHistogram>(familyName,
        "Processing time of requests to a model or a DAG.");

    if (metricConfig->isFamilyEnabled(familyName)) {
        this->requestTimeGrpc = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "gRPC"}},
            this->buckets);

        this->requestTimeRest = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "REST"}},
            this->buckets);
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
        this->inferenceTime = registry->createFamily<MetricHistogram>(familyName,
                                          "Inference execution time in the OpenVINO backend.")
                                  ->addMetric(
                                      {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                      this->buckets);
    }

    familyName = "ovms_wait_for_infer_req_time_us";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->waitForInferReqTime = registry->createFamily<MetricHistogram>(familyName,
                                                "Request waiting time in the scheduling queue.")
                                        ->addMetric(
                                            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                            this->buckets);
    }

    familyName = "ovms_streams";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->streams = registry->createFamily<MetricGauge>(familyName,
                                    "Number of OpenVINO execution streams.")
                            ->addMetric(
                                {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_infer_req_queue_size";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->inferReqQueueSize = registry->createFamily<MetricGauge>(familyName,
                                              "Inference request queue size (nireq).")
                                      ->addMetric(
                                          {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_infer_req_active";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->inferReqActive = registry->createFamily<MetricGauge>(familyName,
                                           "Number of currently consumed inference request from the processing queue.")
                                   ->addMetric(
                                       {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_current_requests";
    if (metricConfig->isFamilyEnabled(familyName)) {
        this->currentRequests = registry->createFamily<MetricGauge>(familyName,
                                            "Number of inference requests currently in process.")
                                    ->addMetric(
                                        {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }
}

}  // namespace ovms
