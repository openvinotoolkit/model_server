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
#include "metric_family.hpp"
#include "metric_registry.hpp"

namespace ovms {

ModelMetricReporter::ModelMetricReporter(MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion) :
    registry(registry) {
    if (!registry) {
        return;
    }
    auto family = registry->createFamily<MetricCounter>("requests_success", "desc");
    // TFS
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
    this->requestSuccessGrpcModelStatus = family->addMetric({{"name", modelName},
        {"protocol", "kserve"},
        {"method", "modelstatus"},
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
    this->requestSuccessRestModelStatus = family->addMetric({{"name", modelName},
        {"protocol", "kserve"},
        {"method", "modelstatus"},
        {"interface", "rest"}});
    family = registry->createFamily<MetricCounter>("requests_fail", "desc");
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
    this->requestFailGrpcModelStatus = family->addMetric({{"name", modelName},
        {"version", std::to_string(modelVersion)},
        {"protocol", "kserve"},
        {"method", "modelstatus"},
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
    this->requestFailRestModelStatus = family->addMetric({{"name", modelName},
        {"version", std::to_string(modelVersion)},
        {"protocol", "kserve"},
        {"method", "modelstatus"},
        {"interface", "rest"}});
}

}  // namespace ovms
