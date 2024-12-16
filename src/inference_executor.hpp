//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "modelinstance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "executingstreamidguard.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "outputkeeper.hpp"
#include "predict_request_validation_utils.hpp"
//#include "predict_request_validation_utils_impl.hpp"
#include "requestprocessor.hpp"
#include "statefulrequestprocessor.hpp"

#include "deserialization_common.hpp"
#include "serialization_common.hpp"
#include "status.hpp"
#include "timer.hpp"

namespace ovms {
enum : unsigned int {
    GET_INFER_REQUEST,
    PREPROCESS,
    DESERIALIZE,
    PREDICTION,
    SERIALIZE,
    POSTPROCESS,
    TIMER_END
};
template <typename RequestType, typename ResponseType>
Status modelInferAsync(ModelInstance& instance, const RequestType* request,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;
    // we don't have response yet
    // auto requestProcessor = createRequestProcessor(request, responseProto);  // request, response passed only to deduce type
    // auto status = requestProcessor->extractRequestParameters(request);
    // if (!status.ok())
    //    return status;
    auto status = request_validation_utils::validate(
        *request,
        instance.getInputsInfo(),
        instance.getOutputsInfo(),
        instance.getName(),
        instance.getVersion(),
        instance.getOptionalInputNames(),
        instance.getModelConfig().getBatchingMode(),
        instance.getModelConfig().getShapes());
    if (status.batchSizeChangeRequired() || status.reshapeRequired()) {
        // We are ensured that request shape is valid and convertible to model shape (non negative, non zero)
        // We can use it to perform reshape via shape=auto
        auto requestBatchSize = getRequestBatchSize(request, instance.getBatchSizeIndex());
        auto requestShapes = getRequestShapes(request);
        status = instance.reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    }
    if (!status.ok())
        return status;
    /* status = requestProcessor->prepare();
    if (!status.ok())
        return status;
*/
    timer.start(GET_INFER_REQUEST);
    OVMS_PROFILE_SYNC_BEGIN("getInferRequest");
    auto executingStreamIdGuard = std::make_shared<ExecutingStreamIdGuard>(instance.getInferRequestsQueue(), instance.getMetricReporter());
    //int executingInferId = executingStreamIdGuard->getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard->getInferRequest();
    OVMS_PROFILE_SYNC_END("getInferRequest");
    timer.stop(GET_INFER_REQUEST);
    /*
    double getInferRequestTime = timer.elapsed<(std::chrono::microseconds)>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(instance.getMetricReporter().waitForInferReqTime, getInferRequestTime);
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, getInferRequestTime / 1000);
        */

    /*
    timer.start(PREPROCESS);
    status = requestProcessor->preInferenceProcessing(inferRequest);
    timer.stop(PREPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<std::chrono::microseconds>(PREPROCESS) / 1000);
*/
    timer.start(DESERIALIZE);
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    std::shared_ptr<OutputKeeper> outKeeper;
    if (instance.doesSupportOutputReset()) {
        outKeeper = std::make_shared<OutputKeeper>(executingStreamIdGuard->getInferRequest(), instance.getOutputsInfo());
    }
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator, InputSink<ov::InferRequest&>>(*request, instance.getInputsInfo(), instance.getOutputsInfo(), inputSink, isPipeline, instance.getTensorFactories());
    timer.stop(DESERIALIZE);
    if (!status.ok()) {
        SPDLOG_DEBUG("Deserialization of outputs failed for model {}, version {}", instance.getName(), instance.getVersion());
        return status;
    }
    /*
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<std::chrono::microseconds>(DESERIALIZE) / 1000);
    */
    // set callback
    // TODO check if there is callback in async
    //OVMS_InferenceRequestCompletionCallback_t userCallback = request->getResponseCompleteCallback();
    OVMS_InferenceRequestCompletionCallback_t userCallback = getCallback(request);

    if (userCallback == nullptr) {
        SPDLOG_DEBUG("User callback not set for async inference.");
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    }

    void* userCallbackData = request->getResponseCompleteCallbackData();
    // here pass by copy into callback
    {
        // order is important here - destructors are called in order from right to left
        inferRequest.set_callback(
            [&instance, request, &inferRequest, userCallback, userCallbackData, modelUnloadGuardPtrMoved = std::shared_ptr<ModelInstanceUnloadGuard>(std::move(modelUnloadGuardPtr)), streamIdGuardMoved = std::move(executingStreamIdGuard), movedOutputKeeper = std::move(outKeeper)](std::exception_ptr exception) mutable {
                struct CallbackGuard {
                    OVMS_InferenceRequestCompletionCallback_t userCallback{nullptr};
                    void* userCallbackData{nullptr};
                    bool success{false};
                    ov::InferRequest& request;
                    OVMS_InferenceResponse* response{nullptr};
                    CallbackGuard(OVMS_InferenceRequestCompletionCallback_t userCallback, void* userCallbackData, ov::InferRequest& request) :
                        userCallback(userCallback),
                        userCallbackData(userCallbackData),
                        request(request) {}
                    ~CallbackGuard() {
                        SPDLOG_DEBUG("Calling user provided callback with success: {}", success);
                        if (!success) {
                            userCallback(nullptr, 1, userCallbackData);
                        } else {
                            userCallback(response, 0, userCallbackData);
                        }
                        SPDLOG_DEBUG("Called user provided callback");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
                        try {
                            OV_LOGGER("ov::InferRequest: {} set_callback() with empty lambda", (void*)&request);
                            request.set_callback([](std::exception_ptr exception_ptr) {});
                        } catch (std::exception& e) {
                            SPDLOG_ERROR("Caught critical exception from OpenVINO InferRequest", e.what());
                            throw e;
                        } catch (...) {
                            SPDLOG_ERROR("Caught critical exception from OpenVINO InferRequest");
                            throw;
                        }
#pragma GCC diagnostic pop
                    }
                };
                SPDLOG_DEBUG("Entry of ov::InferRequest callback call");
                CallbackGuard callbackGuard(userCallback, userCallbackData, inferRequest);
                if (exception) {
                    try {
                        SPDLOG_DEBUG("rethrow_exception");
                        std::rethrow_exception(exception);
                    } catch (const std::exception& e) {
                        SPDLOG_DEBUG("got exception in ov::InferRequest callback: {}", e.what());
                    } catch (...) {
                        SPDLOG_DEBUG("got exception in ov::InferRequest callback");
                        return;
                    }
                }
                std::unique_ptr<ResponseType> res(new ResponseType(instance.getName(), instance.getVersion()));
                OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
                try {
                    // TODO created filter based on what is in request, then perform casual serialization for what was NOT in request, and rewrite tensors from request to response for those that were
                    auto status = serializePredictResponse(outputGetter, instance.getName(), instance.getVersion(), instance.getOutputsInfo(), request, res.get(), getTensorInfoName, useSharedOutputContentFn(request));
                    if (!status.ok()) {
                        SPDLOG_DEBUG("Encountered issue during response serialization:{}", status.string());
                        return;
                    }
                } catch (std::exception& e) {
                    SPDLOG_DEBUG("caught exception in ov::InferRequest callback: {}", e.what());
                } catch (...) {
                    SPDLOG_DEBUG("caught exception in ov::InferRequest callback");
                }
                callbackGuard.response = reinterpret_cast<OVMS_InferenceResponse*>(res.release());
                callbackGuard.success = true;
            });
    }

    try {
        SPDLOG_DEBUG("ov::InferRequest: {}, inferRequest.start_async()", reinterpret_cast<void*>(&inferRequest));
        inferRequest.start_async();
    } catch (std::exception& e) {
        SPDLOG_DEBUG("caught exception in ov::InferRequest.start_async: {}", e.what());
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    } catch (...) {
        SPDLOG_DEBUG("caught exception in ov::InferRequest.start_async");
        return StatusCode::OV_INTERNAL_INFERENCE_ERROR;
    }
    return StatusCode::OK;
}
template <typename RequestType, typename ResponseType>
Status infer(ModelInstance& instance, const RequestType* requestProto,
    ResponseType* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;
    auto processorManager = instance.getSequenceManager();
    std::unique_ptr<RequestProcessor<RequestType, ResponseType>> requestProcessor;
    if (processorManager) {
        requestProcessor = std::make_unique<StatefulRequestProcessor<RequestType, ResponseType>>(*processorManager);

    } else {
        requestProcessor = std::make_unique<RequestProcessor<RequestType, ResponseType>>();

    }
    auto status = requestProcessor->extractRequestParameters(requestProto);
    if (!status.ok())
        return status;
    status = request_validation_utils::validate(
        *requestProto,
        instance.getInputsInfo(),
        instance.getOutputsInfo(),
        instance.getName(),
        instance.getVersion(),
        instance.getOptionalInputNames(),
        instance.getModelConfig().getBatchingMode(),
        instance.getModelConfig().getShapes());
    if (status.batchSizeChangeRequired() || status.reshapeRequired()) {
        // We are ensured that request shape is valid and convertible to model shape (non negative, non zero)
        // We can use it to perform reshape via shape=auto
        auto requestBatchSize = getRequestBatchSize(requestProto, instance.getBatchSizeIndex());
        auto requestShapes = getRequestShapes(requestProto);
        status = instance.reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    }
    if (!status.ok())
        return status;
    status = requestProcessor->prepare();
    if (!status.ok())
        return status;

    timer.start(GET_INFER_REQUEST);
    OVMS_PROFILE_SYNC_BEGIN("getInferRequest");
    ExecutingStreamIdGuard executingStreamIdGuard(instance.getInferRequestsQueue(), instance.getMetricReporter());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    OVMS_PROFILE_SYNC_END("getInferRequest");
    timer.stop(GET_INFER_REQUEST);
    double getInferRequestTime = timer.elapsed<microseconds>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(instance.getMetricReporter().waitForInferReqTime, getInferRequestTime);
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, getInferRequestTime / 1000);

    timer.start(PREPROCESS);
    status = requestProcessor->preInferenceProcessing(inferRequest);
    timer.stop(PREPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(PREPROCESS) / 1000);

    timer.start(DESERIALIZE);
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;

    std::unique_ptr<OutputKeeper> outKeeper;
    if (instance.doesSupportOutputReset()) {
        outKeeper = std::make_unique<OutputKeeper>(executingStreamIdGuard.getInferRequest(), instance.getOutputsInfo());
    }
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator, InputSink<ov::InferRequest&>>(*requestProto, instance.getInputsInfo(), instance.getOutputsInfo(), inputSink, isPipeline, instance.getTensorFactories());
    timer.stop(DESERIALIZE);
    if (!status.ok()) {
        SPDLOG_DEBUG("Deserialization of outputs failed for model {}, version {}", instance.getName(), instance.getVersion());
        return status;
    }
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(DESERIALIZE) / 1000);

    timer.start(PREDICTION);
    status = instance.performInference(inferRequest);
    timer.stop(PREDICTION);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(PREDICTION) / 1000);

    timer.start(SERIALIZE);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    status = serializePredictResponse(outputGetter, instance.getName(), instance.getVersion(), instance.getOutputsInfo(), requestProto, responseProto, getTensorInfoName, useSharedOutputContentFn(requestProto));
    timer.stop(SERIALIZE);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(SERIALIZE) / 1000);

    timer.start(POSTPROCESS);
    status = requestProcessor->postInferenceProcessing(responseProto, inferRequest);
    timer.stop(POSTPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Postprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(POSTPROCESS) / 1000);
    /*if (instance.getTargetDevice() == "AUTO") // TODO FIXME @atobisze perf for auto drop?
        for (std::string device : compiledModel->get_property(ov::execution_devices))
            SPDLOG_DEBUG("Used device: {}", device);
    */
    status = requestProcessor->release();
    // handleCallback(requestProto, responseProto); to be enabled when callbacks are implemented in network API's
    return status;
}
}  // namespace ovms
