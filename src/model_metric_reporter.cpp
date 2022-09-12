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

    auto familyName = "ovms_requests_success";
    auto family = registry->createFamily<MetricCounter>(familyName,
        "Number of successful requests to a model or a DAG.");
 
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        //TFS
        this->requestSuccessGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "grpc"}});

        this->requestSuccessGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "grpc"}});

        this->requestSuccessGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "grpc"}});

        this->requestSuccessRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "rest"}});

        this->requestSuccessRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "rest"}});

        this->requestSuccessRestGetModelStatus = family->addMetric({{"name", modelName},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "rest"}});
        // KFS
        this->requestSuccessGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "grpc"}});

        this->requestSuccessGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "grpc"}});

        this->requestSuccessGrpcModelReady = family->addMetric({{"name", modelName},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "grpc"}});

        this->requestSuccessRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "rest"}});

        this->requestSuccessRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "rest"}});

        this->requestSuccessRestModelReady = family->addMetric({{"name", modelName},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "rest"}});
    }

    familyName = "ovms_requests_fail";
    family = registry->createFamily<MetricCounter>(familyName,
        "Number of failed requests to a model or a DAG.");

    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        // TFS
        this->requestFailGrpcPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "grpc"}});

        this->requestFailGrpcGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "grpc"}});

        this->requestFailGrpcGetModelStatus = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "grpc"}});

        this->requestFailRestPredict = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "predict"},
            {"interface", "rest"}});

        this->requestFailRestGetModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelmetadata"},
            {"interface", "rest"}});

        this->requestFailRestGetModelStatus = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "tensorflowserving"},
            {"method", "getmodelstatus"},
            {"interface", "rest"}});
        
        // KFS
        this->requestFailGrpcModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "grpc"}});

        this->requestFailGrpcModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "grpc"}});

        this->requestFailGrpcModelReady = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "grpc"}});

        this->requestFailRestModelInfer = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelinfer"},
            {"interface", "rest"}});

        this->requestFailRestModelMetadata = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelmetadata"},
            {"interface", "rest"}});

        this->requestFailRestModelReady = family->addMetric({{"name", modelName},
            {"version", std::to_string(modelVersion)},
            {"protocol", "kserve"},
            {"method", "modelready"},
            {"interface", "rest"}});
    }

    familyName = "ovms_request_time_us";
    auto requestTimeFamily = registry->createFamily<MetricHistogram>(familyName,
        "Processing time of requests to a model or a DAG.");

    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->requestTimeGrpc = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "grpc"}},
            this->buckets);
    
        this->requestTimeRest = requestTimeFamily->addMetric({{"name", modelName},
                                                                 {"version", std::to_string(modelVersion)},
                                                                 {"interface", "rest"}},
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

    auto familyName = "ovms_inference_time_us";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->inferenceTime = registry->createFamily<MetricHistogram>(familyName,
                                          "Inference execution time in the OpenVINO backend.")
                                  ->addMetric(
                                      {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                      this->buckets);
    }

    familyName = "ovms_wait_for_infer_req_time_us";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->waitForInferReqTime = registry->createFamily<MetricHistogram>(familyName,
                                                "Request waiting time in the scheduling queue.")
                                        ->addMetric(
                                            {{"name", modelName}, {"version", std::to_string(modelVersion)}},
                                            this->buckets);
    }

    familyName = "ovms_streams";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->streams = registry->createFamily<MetricGauge>(familyName,
                                    "Number of OpenVINO execution streams.")
                            ->addMetric(
                                {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_infer_req_queue_size";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->inferReqQueueSize = registry->createFamily<MetricGauge>(familyName,
                                              "Inference request queue size (nireq).")
                                      ->addMetric(
                                          {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_infer_req_active";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->inferReqActive = registry->createFamily<MetricGauge>(familyName,
                                           "Number of currently consumed inference request from the processing queue.")
                                   ->addMetric(
                                       {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }

    familyName = "ovms_current_requests";
    if (metricConfig->enabledFamiliesList.find(familyName) != metricConfig->enabledFamiliesList.end())
    {
        this->currentRequests = registry->createFamily<MetricGauge>(familyName,
                                            "Number of inference requests currently in process.")
                                    ->addMetric(
                                        {{"name", modelName}, {"version", std::to_string(modelVersion)}});
    }
}

}  // namespace ovms
