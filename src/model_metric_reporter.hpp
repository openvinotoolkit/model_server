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
#include <vector>

#include "execution_context.hpp"
#include "metric.hpp"
#include "modelversion.hpp"

namespace ovms {

class MetricRegistry;
class MetricConfig;

class ServableMetricReporter {
    MetricRegistry* registry;

protected:
    std::vector<double> buckets;

public:
    ServableMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion);
    virtual ~ServableMetricReporter();

    // TFS
    std::unique_ptr<MetricCounter> requestSuccessGrpcPredict;
    std::unique_ptr<MetricCounter> requestSuccessGrpcGetModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessGrpcGetModelStatus;

    std::unique_ptr<MetricCounter> requestSuccessRestPredict;
    std::unique_ptr<MetricCounter> requestSuccessRestGetModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessRestGetModelStatus;

    std::unique_ptr<MetricCounter> requestFailGrpcPredict;
    std::unique_ptr<MetricCounter> requestFailGrpcGetModelMetadata;
    std::unique_ptr<MetricCounter> requestFailGrpcGetModelStatus;

    std::unique_ptr<MetricCounter> requestFailRestPredict;
    std::unique_ptr<MetricCounter> requestFailRestGetModelMetadata;
    std::unique_ptr<MetricCounter> requestFailRestGetModelStatus;

    // KFS
    std::unique_ptr<MetricCounter> requestSuccessGrpcModelInfer;
    std::unique_ptr<MetricCounter> requestSuccessGrpcModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessGrpcModelReady;

    std::unique_ptr<MetricCounter> requestSuccessRestModelInfer;
    std::unique_ptr<MetricCounter> requestSuccessRestModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessRestModelReady;

    std::unique_ptr<MetricCounter> requestFailGrpcModelInfer;
    std::unique_ptr<MetricCounter> requestFailGrpcModelMetadata;
    std::unique_ptr<MetricCounter> requestFailGrpcModelReady;

    std::unique_ptr<MetricCounter> requestFailRestModelInfer;
    std::unique_ptr<MetricCounter> requestFailRestModelMetadata;
    std::unique_ptr<MetricCounter> requestFailRestModelReady;

    std::unique_ptr<MetricHistogram> requestTimeGrpc;
    std::unique_ptr<MetricHistogram> requestTimeRest;

    inline std::unique_ptr<MetricCounter>& getGetModelStatusRequestSuccessMetric(const ExecutionContext& context) {
        if (context.method != ExecutionContext::Method::GetModelStatus) {
            static std::unique_ptr<MetricCounter> empty = nullptr;
            return empty;  // In case something calls it from ConfigReload/ConfigStatus methods
        }
        if (context.interface == ExecutionContext::Interface::GRPC) {
            return this->requestSuccessGrpcGetModelStatus;
        } else {
            return this->requestSuccessRestGetModelStatus;
        }
    }

    inline std::unique_ptr<MetricCounter>& getGetModelMetadataRequestMetric(const ExecutionContext& context, bool success) {
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

    inline std::unique_ptr<MetricCounter>& getInferRequestMetric(const ExecutionContext& context, bool success = true) {
        if (context.method == ExecutionContext::Method::Predict) {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return success ? this->requestSuccessGrpcPredict : this->requestFailGrpcPredict;
            } else {
                return success ? this->requestSuccessRestPredict : this->requestFailRestPredict;
            }
        } else if (context.method == ExecutionContext::Method::ModelInfer) {
            if (context.interface == ExecutionContext::Interface::GRPC) {
                return success ? this->requestSuccessGrpcModelInfer : this->requestFailGrpcModelInfer;
            } else {
                return success ? this->requestSuccessRestModelInfer : this->requestFailRestModelInfer;
            }
        } else {
            throw std::logic_error("wrong context method for inference");
        }
    }

    inline std::unique_ptr<MetricCounter>& getModelMetadataMetric(const ExecutionContext& context, bool success = true) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            return success ? this->requestSuccessGrpcModelMetadata : this->requestFailGrpcModelMetadata;
        } else {
            return success ? this->requestSuccessRestModelMetadata : this->requestFailRestModelMetadata;
        }
    }

    inline std::unique_ptr<MetricCounter>& getModelReadyMetric(const ExecutionContext& context, bool success = true) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            return success ? this->requestSuccessGrpcModelReady : this->requestFailGrpcModelReady;
        } else {
            return success ? this->requestSuccessRestModelReady : this->requestFailRestModelReady;
        }
    }
};

class ModelMetricReporter : public ServableMetricReporter {
public:
    std::unique_ptr<MetricHistogram> inferenceTime;
    std::unique_ptr<MetricHistogram> waitForInferReqTime;

    std::unique_ptr<MetricGauge> streams;
    std::unique_ptr<MetricGauge> inferReqQueueSize;
    std::unique_ptr<MetricGauge> inferReqActive;
    std::unique_ptr<MetricGauge> currentRequests;

    ModelMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& modelName, model_version_t modelVersion);
};

}  // namespace ovms
