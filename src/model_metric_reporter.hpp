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
#pragma once

#include <memory>
#include <string>

#include "modelversion.hpp"
#include "metric.hpp"

namespace ovms {

class MetricRegistry;
class MetricConfig;

class ModelMetricReporter {
    MetricRegistry* registry;

public:
    ModelMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion);

    // TFS
    std::shared_ptr<MetricCounter> requestSuccessGrpcPredict;
    std::shared_ptr<MetricCounter> requestSuccessGrpcGetModelMetadata;
    std::shared_ptr<MetricCounter> requestSuccessGrpcGetModelStatus;

    std::shared_ptr<MetricCounter> requestSuccessRestPredict;
    std::shared_ptr<MetricCounter> requestSuccessRestGetModelMetadata;
    std::shared_ptr<MetricCounter> requestSuccessRestGetModelStatus;

    std::shared_ptr<MetricCounter> requestFailGrpcPredict;
    std::shared_ptr<MetricCounter> requestFailGrpcGetModelMetadata;
    std::shared_ptr<MetricCounter> requestFailGrpcGetModelStatus;

    std::shared_ptr<MetricCounter> requestFailRestPredict;
    std::shared_ptr<MetricCounter> requestFailRestGetModelMetadata;
    std::shared_ptr<MetricCounter> requestFailRestGetModelStatus;

    // KFS
    std::shared_ptr<MetricCounter> requestSuccessGrpcModelInfer;
    std::shared_ptr<MetricCounter> requestSuccessGrpcModelMetadata;
    std::shared_ptr<MetricCounter> requestSuccessGrpcModelStatus;

    std::shared_ptr<MetricCounter> requestSuccessRestModelInfer;
    std::shared_ptr<MetricCounter> requestSuccessRestModelMetadata;
    std::shared_ptr<MetricCounter> requestSuccessRestModelStatus;

    std::shared_ptr<MetricCounter> requestFailGrpcModelInfer;
    std::shared_ptr<MetricCounter> requestFailGrpcModelMetadata;
    std::shared_ptr<MetricCounter> requestFailGrpcModelStatus;

    std::shared_ptr<MetricCounter> requestFailRestModelInfer;
    std::shared_ptr<MetricCounter> requestFailRestModelMetadata;
    std::shared_ptr<MetricCounter> requestFailRestModelStatus;
};

}  // namespace ovms
