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

#include "execution_context.hpp"
#include "metric.hpp"
#include "modelversion.hpp"

namespace ovms {

class MetricRegistry;

class ModelMetricReporter {
    MetricRegistry* registry;

public:
    ModelMetricReporter(MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion);

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

    inline std::shared_ptr<MetricCounter>& getGetModelStatusRequestSuccessMetric(ExecutionContext& context) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            return this->requestSuccessGrpcGetModelStatus;
        } else {
            return this->requestSuccessRestGetModelStatus;
        }
    }

    inline std::shared_ptr<MetricCounter>& getGetModelMetadataRequestMetric(ExecutionContext& context, bool success) {
        if (success) {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return this->requestSuccessGrpcGetModelMetadata;
            } else {
                return this->requestSuccessRestGetModelMetadata;
            }
        } else {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return this->requestFailGrpcGetModelMetadata;
            } else {
                return this->requestFailRestGetModelMetadata;
            }
        }
    }

    inline std::shared_ptr<MetricCounter>& getInferRequestMetric(ExecutionContext& context) {
        if (context.method == ExecutionContext::Method::Predict) {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return this->requestSuccessGrpcPredict;
            } else {
                return this->requestSuccessRestPredict;
            }
        } else if (context.method == ExecutionContext::Method::ModelInfer) {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return this->requestSuccessGrpcModelInfer;
            } else {
                return this->requestSuccessRestModelInfer;
            }
        } else {
            throw std::logic_error("wrong context method for inference");
        }
    }
};

}  // namespace ovms
