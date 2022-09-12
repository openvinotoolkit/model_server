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

    auto family = registry->createFamily<MetricCounter>("ovms_requests_success",
        "Number of successful requests to a model or a DAG.");
    // TFS
    if (metricConfig->requestSuccessGrpcPredict)
        this->requestSuccessGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessGrpcGetModelMetadata)
        this->requestSuccessGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessGrpcGetModelStatus)
        this->requestSuccessGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessRestPredict)
        this->requestSuccessRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "rest"}});
    if (metricConfig->requestSuccessRestGetModelMetadata)
        this->requestSuccessRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "rest"}});
    if (metricConfig->requestSuccessRestGetModelStatus)
        this->requestSuccessRestGetModelStatus = family->addMetric({{"name", modelName},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "rest"}});
    // KFS
    if (metricConfig->requestSuccessGrpcModelInfer)
        this->requestSuccessGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessGrpcModelMetadata)
        this->requestSuccessGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessGrpcModelReady)
        this->requestSuccessGrpcModelReady = family->addMetric({{"name", modelName},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "grpc"}});
    if (metricConfig->requestSuccessRestModelInfer)
        this->requestSuccessRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "rest"}});
    if (metricConfig->requestSuccessRestModelMetadata)
        this->requestSuccessRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "rest"}});
    if (metricConfig->requestSuccessRestModelReady)
        this->requestSuccessRestModelReady = family->addMetric({{"name", modelName},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "rest"}});

    family = registry->createFamily<MetricCounter>("ovms_requests_fail",
        "Number of failed requests to a model or a DAG.");

    // TFS
    if (metricConfig->requestFailGrpcPredict)
        this->requestFailGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailGrpcGetModelMetadata)
        this->requestFailGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailGrpcGetModelStatus)
        this->requestFailGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailRestPredict)
        this->requestFailRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "rest"}});
    if (metricConfig->requestFailRestGetModelMetadata)
        this->requestFailRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "rest"}});
    if (metricConfig->requestFailRestGetModelStatus)
        this->requestFailRestGetModelStatus = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "rest"}});
    // KFS
    if (metricConfig->requestFailGrpcModelInfer)
        this->requestFailGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailGrpcModelMetadata)
        this->requestFailGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailGrpcModelReady)
        this->requestFailGrpcModelReady = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "grpc"}});
    if (metricConfig->requestFailRestModelInfer)
        this->requestFailRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "rest"}});
    if (metricConfig->requestFailRestModelMetadata)
        this->requestFailRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "rest"}});
    if (metricConfig->requestFailRestModelReady)
        this->requestFailRestModelReady = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "rest"}});

    auto requestTimeFamily = registry->createFamily<MetricHistogram>("ovms_request_time_us",
        "Processing time of requests to a model or a DAG.");

    if (metricConfig->requestTimeGrpc)
        this->requestTimeGrpc = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "grpc"}},
            this->buckets);

    if (metricConfig->requestTimeRest)
        this->requestTimeRest = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "rest"}},
            this->buckets);
}

ModelMetricReporter::ModelMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion) :
    ServableMetricReporter(metricConfig, registry, modelName, modelVersion) {
    if (!registry) {
        return;
    }

    if (!metricConfig || !metricConfig->metricsEnabled) {
        return;
    }

    if (metricConfig->inferenceTime)
        this->inferenceTime = registry->createFamily<MetricHistogram>("ovms_inference_time_us",
                                          "Inference execution time in the OpenVINO backend.")
                                  ->addMetric(
                                      {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                      this->buckets);

    if (metricConfig->waitForInferReqTime)
        this->waitForInferReqTime = registry->createFamily<MetricHistogram>("ovms_wait_for_infer_req_time_us",
                                                "Request waiting time in the scheduling queue.")
                                        ->addMetric(
                                            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                            this->buckets);

    if (metricConfig->streams)
        this->streams = registry->createFamily<MetricGauge>("ovms_streams",
                                    "Number of OpenVINO execution streams.")
                            ->addMetric(
                                {{"name", modelName}, {"version", std::to_string(modelVersion)}});

    if (metricConfig->inferReqQueueSize)
        this->inferReqQueueSize = registry->createFamily<MetricGauge>("ovms_infer_req_queue_size",
                                              "Inference request queue size (nireq).")
                                      ->addMetric(
                                          {{"name", modelName}, {"version", std::to_string(modelVersion)}});

    if (metricConfig->inferReqActive)
        this->inferReqActive = registry->createFamily<MetricGauge>("ovms_infer_req_active",
                                           "Number of currently consumed inference request from the processing queue.")
                                   ->addMetric(
                                       {{"name", modelName}, {"version", std::to_string(modelVersion)}});

    if (metricConfig->currentRequests)
        this->currentRequests = registry->createFamily<MetricGauge>("ovms_current_requests",
                                            "Number of inference requests currently in process.")
                                    ->addMetric(
                                        {{"name", modelName}, {"version", std::to_string(modelVersion)}});
}

}  // namespace ovms
