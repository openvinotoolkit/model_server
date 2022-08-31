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

#include "execution_context.hpp"
#include "metric_config.hpp"
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {

ModelMetricReporter::ModelMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion) :
    registry(registry) {
    if (!registry) {
        return;
    }

    if (!metricConfig || !metricConfig->metricsEnabled) {
        return;
    }

    auto family = registry->createFamily<MetricCounter>("requests_success", "desc");
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
            {"method", "modelstatus"},
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
            {"method", "modelstatus"},
            {"interface", "rest"}});
    family = registry->createFamily<MetricCounter>("requests_fail", "desc");

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
            {"method", "modelstatus"},
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
            {"method", "modelstatus"},
            {"interface", "rest"}});
}

}  // namespace ovms
