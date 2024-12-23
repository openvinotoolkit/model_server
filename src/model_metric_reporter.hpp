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

#include <iostream>
#include <memory>
#include <stdexcept>
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

class MediapipeServableMetricReporter {
    MetricRegistry* registry;

protected:
    std::vector<double> buckets;

public:
    std::unique_ptr<MetricGauge> currentGraphs;

    // KFS
    std::unique_ptr<MetricCounter> requestAcceptedGrpcModelInfer;
    std::unique_ptr<MetricCounter> requestAcceptedGrpcModelInferStream;

    std::unique_ptr<MetricCounter> requestAcceptedRestModelInfer;

    std::unique_ptr<MetricCounter> requestRejectedGrpcModelInfer;
    std::unique_ptr<MetricCounter> requestRejectedGrpcModelInferStream;

    std::unique_ptr<MetricCounter> requestRejectedRestModelInfer;

    // V3
    std::unique_ptr<MetricCounter> requestAcceptedRestV3Unary;
    std::unique_ptr<MetricCounter> requestAcceptedRestV3Stream;
    std::unique_ptr<MetricCounter> requestRejectedRestV3Unary;
    std::unique_ptr<MetricCounter> requestRejectedRestV3Stream;

    // --- responses -----
    // KFS
    std::unique_ptr<MetricCounter> responseGrpcModelInfer;
    std::unique_ptr<MetricCounter> responseGrpcModelInferStream;

    std::unique_ptr<MetricCounter> responseRestModelInfer;

    std::unique_ptr<MetricCounter> requestSuccessGrpcModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessGrpcModelReady;

    std::unique_ptr<MetricCounter> requestSuccessRestModelMetadata;
    std::unique_ptr<MetricCounter> requestSuccessRestModelReady;

    std::unique_ptr<MetricCounter> requestFailGrpcModelMetadata;
    std::unique_ptr<MetricCounter> requestFailGrpcModelReady;

    std::unique_ptr<MetricCounter> requestFailRestModelMetadata;
    std::unique_ptr<MetricCounter> requestFailRestModelReady;

    // V3
    std::unique_ptr<MetricCounter> responseRestV3Unary;
    std::unique_ptr<MetricCounter> responseRestV3Stream;

    std::unique_ptr<MetricCounter> requestErrorGrpcModelInfer;
    std::unique_ptr<MetricCounter> requestErrorGrpcModelInferStream;
    std::unique_ptr<MetricCounter> requestErrorRestModelInfer;
    std::unique_ptr<MetricCounter> requestErrorRestV3Unary;
    std::unique_ptr<MetricCounter> requestErrorRestV3Stream;

    std::unique_ptr<MetricHistogram> processingTimeGrpcModelInfer;
    std::unique_ptr<MetricHistogram> processingTimeGrpcModelInferStream;
    std::unique_ptr<MetricHistogram> processingTimeRestModelInfer;
    std::unique_ptr<MetricHistogram> processingTimeRestV3Unary;
    std::unique_ptr<MetricHistogram> processingTimeRestV3Stream;

    inline MetricHistogram* getProcessingTimeMetric(const ExecutionContext& context) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->processingTimeGrpcModelInfer.get();
            if (context.method == ExecutionContext::Method::ModelInferStream)
                return this->processingTimeGrpcModelInferStream.get();
            return nullptr;
        } else if (context.interface == ExecutionContext::Interface::REST) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->processingTimeRestModelInfer.get();
            if (context.method == ExecutionContext::Method::V3Unary)
                return this->processingTimeRestV3Unary.get();
            if (context.method == ExecutionContext::Method::V3Stream)
                return this->processingTimeRestV3Stream.get();
            return nullptr;
        } else {
            return nullptr;
        }
        return nullptr;
    }

    inline MetricCounter* getRequestsMetric(const ExecutionContext& context, bool success = true) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return success ? this->requestAcceptedGrpcModelInfer.get() : this->requestRejectedGrpcModelInfer.get();
            if (context.method == ExecutionContext::Method::ModelInferStream)
                return success ? this->requestAcceptedGrpcModelInferStream.get() : this->requestRejectedGrpcModelInferStream.get();
            return nullptr;
        } else if (context.interface == ExecutionContext::Interface::REST) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return success ? this->requestAcceptedRestModelInfer.get() : this->requestRejectedRestModelInfer.get();
            if (context.method == ExecutionContext::Method::V3Unary)
                return success ? this->requestAcceptedRestV3Unary.get() : this->requestRejectedRestV3Unary.get();
            if (context.method == ExecutionContext::Method::V3Stream)
                return success ? this->requestAcceptedRestV3Stream.get() : this->requestRejectedRestV3Stream.get();
            return nullptr;
        } else {
            return nullptr;
        }
        return nullptr;
    }

    inline MetricCounter* getGraphErrorMetric(const ExecutionContext& context) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->requestErrorGrpcModelInfer.get();
            if (context.method == ExecutionContext::Method::ModelInferStream)
                return this->requestErrorGrpcModelInferStream.get();
            return nullptr;
        } else if (context.interface == ExecutionContext::Interface::REST) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->requestErrorRestModelInfer.get();
            if (context.method == ExecutionContext::Method::V3Unary)
                return this->requestErrorRestV3Unary.get();
            if (context.method == ExecutionContext::Method::V3Stream)
                return this->requestErrorRestV3Stream.get();
            return nullptr;
        } else {
            return nullptr;
        }
        return nullptr;
    }

    inline MetricCounter* getResponsesMetric(const ExecutionContext& context) {
        if (context.interface == ExecutionContext::Interface::GRPC) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->responseGrpcModelInfer.get();
            if (context.method == ExecutionContext::Method::ModelInferStream)
                return this->responseGrpcModelInferStream.get();
            return nullptr;
        } else if (context.interface == ExecutionContext::Interface::REST) {
            if (context.method == ExecutionContext::Method::ModelInfer)
                return this->responseRestModelInfer.get();
            if (context.method == ExecutionContext::Method::V3Unary)
                return this->responseRestV3Unary.get();
            if (context.method == ExecutionContext::Method::V3Stream)
                return this->responseRestV3Stream.get();
            return nullptr;
        } else {
            return nullptr;
        }
        return nullptr;
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

    MediapipeServableMetricReporter(const MetricConfig* metricConfig, MetricRegistry* registry, const std::string& graphName);
};

}  // namespace ovms
