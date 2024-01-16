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
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../config.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../prediction_service.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

// This checks for counter to be present with exact value and other remaining metrics of the family to be 0.
static void checkRequestsCounter(const std::string& collectedMetricData, const std::string& metricName, const std::string& endpointName, std::optional<model_version_t> endpointVersion, const std::string& interfaceName, const std::string& method, const std::string& api, int value) {
    for (std::string _interface : std::set<std::string>{"gRPC", "REST"}) {
        for (std::string _api : std::set<std::string>{"TensorFlowServing", "KServe"}) {
            if (_api == "KServe") {
                for (std::string _method : std::set<std::string>{"ModelInfer", "ModelMetadata", "ModelReady"}) {
                    std::stringstream ss;
                    ss << metricName << "{api=\"" << _api << "\",interface=\"" << _interface << "\",method=\"" << _method << "\",name=\"" << endpointName << "\"";
                    if (_method != "ModelReady") {
                        ss << ",version=\"" << endpointVersion.value() << "\"";
                    }
                    ss << "}";
                    int expectedValue = interfaceName == _interface && method == _method && api == _api ? value : 0;
                    ss << " " << expectedValue << "\n";
                    ASSERT_THAT(collectedMetricData, HasSubstr(ss.str()));
                }
            } else {
                for (std::string _method : std::set<std::string>{"Predict", "GetModelMetadata", "GetModelStatus"}) {
                    std::stringstream ss;
                    ss << metricName << "{api=\"" << _api << "\",interface=\"" << _interface << "\",method=\"" << _method << "\",name=\"" << endpointName << "\"";
                    if (_method != "GetModelStatus") {
                        ss << ",version=\"" << endpointVersion.value() << "\"";
                    }
                    ss << "}";
                    int expectedValue = interfaceName == _interface && method == _method && api == _api ? value : 0;
                    ss << " " << expectedValue << "\n";
                    ASSERT_THAT(collectedMetricData, HasSubstr(ss.str()));
                }
            }
        }
    }
}

class ServableManagerModuleWithMockedManager : public ServableManagerModule {
    ConstructorEnabledModelManager& mockedManager;

public:
    ServableManagerModuleWithMockedManager(ovms::Server& ovmsServer, ConstructorEnabledModelManager& manager) :
        ServableManagerModule(ovmsServer),
        mockedManager(manager) {}

    ModelManager& getServableManager() const override { return this->mockedManager; }
};

class ServerWithMockedManagerModule : public Server {
    ConstructorEnabledModelManager manager;

public:
    ServerWithMockedManagerModule() {
        auto module = this->createModule(METRICS_MODULE_NAME);
        this->modules.emplace(METRICS_MODULE_NAME, std::move(module));
        module = std::make_unique<ServableManagerModuleWithMockedManager>(*this, this->manager);
        this->modules.emplace(SERVABLE_MANAGER_MODULE_NAME, std::move(module));
        module = this->createModule(GRPC_SERVER_MODULE_NAME);
        this->modules.emplace(GRPC_SERVER_MODULE_NAME, std::move(module));
    }

    ConstructorEnabledModelManager& getManager() {
        return this->manager;
    }

    std::string collect() {
        return this->getManager().getMetricRegistry()->collect();
    }
};

class MetricFlowTest : public TestWithTempDir {
protected:
    ServerWithMockedManagerModule server;

    const int numberOfSuccessRequests = 5;
    const int numberOfFailedRequests = 7;
    const int64_t dynamicBatch = 3;

    const Precision correctPrecision = Precision::FP32;
    const Precision wrongPrecision = Precision::I32;

    const std::string modelName = "dummy";
    const std::string dagName = "dummy_demux";

    std::optional<int64_t> modelVersion = std::nullopt;
    std::optional<std::string_view> modelVersionLabel{std::nullopt};

    std::string prepareConfigContent();

    void SetUp() override {
        TestWithTempDir::SetUp();
        char* n_argv[] = {(char*)"ovms", (char*)"--config_path", (char*)"/unused", (char*)"--rest_port", (char*)"8080"};  // Workaround to have rest_port parsed in order to enable metrics
        int arg_count = 5;
        ovms::Config::instance().parse(arg_count, n_argv);
        std::string fileToReload = this->directoryPath + "/config.json";
        createConfigFileWithContent(this->prepareConfigContent(), fileToReload);
        ASSERT_EQ(server.getManager().loadConfig(fileToReload), StatusCode::OK);
    }
};

TEST_F(MetricFlowTest, GrpcPredict) {
    PredictionServiceImpl impl(server);
    tensorflow::serving::PredictRequest request;
    tensorflow::serving::PredictResponse response;

    // Successful single model calls
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(modelName);
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {DUMMY_MODEL_SHAPE, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        ASSERT_EQ(impl.Predict(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }
    // Failed single model calls
    for (int i = 0; i < numberOfFailedRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(modelName);
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {DUMMY_MODEL_SHAPE, wrongPrecision}}};
        preparePredictRequest(request, inputsMeta);
        ASSERT_EQ(impl.Predict(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }

    // Successful DAG calls
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(dagName);
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {ovms::signed_shape_t{dynamicBatch, 1, DUMMY_MODEL_INPUT_SIZE}, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        ASSERT_EQ(impl.Predict(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    // Failed DAG calls
    for (int i = 0; i < numberOfFailedRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(dagName);
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {ovms::signed_shape_t{dynamicBatch, 1, DUMMY_MODEL_INPUT_SIZE}, wrongPrecision}}};
        preparePredictRequest(request, inputsMeta);
        ASSERT_EQ(impl.Predict(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }

    // ovms_requests_success
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "Predict", "TensorFlowServing", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "Predict", "TensorFlowServing", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "gRPC", "Predict", "TensorFlowServing", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "gRPC", "Predict", "TensorFlowServing", numberOfFailedRequests);    // ran by real request

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(0)));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(4)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(2)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));
}

TEST_F(MetricFlowTest, GrpcGetModelMetadata) {
    PredictionServiceImpl impl(server);
    tensorflow::serving::GetModelMetadataRequest request;
    tensorflow::serving::GetModelMetadataResponse response;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(modelName);
        request.add_metadata_field("signature_def");
        ASSERT_EQ(impl.GetModelMetadata(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(dagName);
        request.add_metadata_field("signature_def");
        ASSERT_EQ(impl.GetModelMetadata(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "GetModelMetadata", "TensorFlowServing", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "GetModelMetadata", "TensorFlowServing", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, GrpcGetModelStatus) {
    ModelServiceImpl impl(server);
    tensorflow::serving::GetModelStatusRequest request;
    tensorflow::serving::GetModelStatusResponse response;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(modelName);
        ASSERT_EQ(impl.GetModelStatus(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_model_spec()->mutable_name()->assign(dagName);
        ASSERT_EQ(impl.GetModelStatus(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "GetModelStatus", "TensorFlowServing", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "GetModelStatus", "TensorFlowServing", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, GrpcModelInfer) {
    KFSInferenceServiceImpl impl(server);
    ::KFSRequest request;
    ::KFSResponse response;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {DUMMY_MODEL_SHAPE, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(modelName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {DUMMY_MODEL_SHAPE, wrongPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(modelName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {ovms::signed_shape_t{dynamicBatch, 1, DUMMY_MODEL_INPUT_SIZE}, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(dagName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{DUMMY_MODEL_INPUT_NAME, {ovms::signed_shape_t{dynamicBatch, 1, DUMMY_MODEL_INPUT_SIZE}, wrongPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(dagName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelInfer", "KServe", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelInfer", "KServe", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "gRPC", "ModelInfer", "KServe", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "gRPC", "ModelInfer", "KServe", numberOfFailedRequests);    // ran by real request

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(0)));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(4)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(2)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));
}

TEST_F(MetricFlowTest, GrpcModelMetadata) {
    KFSInferenceServiceImpl impl(server);
    ::KFSModelMetadataRequest request;
    KFSModelMetadataResponse response;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(modelName);
        ASSERT_EQ(impl.ModelMetadata(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(dagName);
        ASSERT_EQ(impl.ModelMetadata(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelMetadata", "KServe", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, GrpcModelReady) {
    KFSInferenceServiceImpl impl(server);
    ::KFSGetModelStatusRequest request;
    ::KFSGetModelStatusResponse response;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(modelName);
        ASSERT_EQ(impl.ModelReady(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(dagName);
        ASSERT_EQ(impl.ModelReady(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelReady", "KServe", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, RestPredict) {
    HttpRestApiHandler handler(server, 0);

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string request = R"({"signature_name": "serving_default", "instances": [[1,2,3,4,5,6,7,8,9,10]]})";
        std::string response;
        ASSERT_EQ(handler.processPredictRequest(modelName, modelVersion, modelVersionLabel, request, &response), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        std::string request = R"({"signature_name": "serving_default", "instances": [[1,2,3,4,5,6,7,8,9]]})";
        std::string response;
        ASSERT_EQ(handler.processPredictRequest(modelName, modelVersion, modelVersionLabel, request, &response), ovms::StatusCode::INVALID_SHAPE);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string request = R"({"signature_name": "serving_default", "instances": [[[1,2,3,4,5,6,7,8,9,10]],[[1,2,3,4,5,6,7,8,9,10]],[[1,2,3,4,5,6,7,8,9,10]]]})";
        std::string response;
        ASSERT_EQ(handler.processPredictRequest(dagName, modelVersion, modelVersionLabel, request, &response), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        std::string request = R"({"signature_name": "serving_default", "instances": [[[1,2,3,4,5,6,7,8,9,10]],[[1,2,3,4,5,6,7,8,9,10]],[[1,2,3,4,5,6,7,8,9]]]})";
        std::string response;
        ASSERT_EQ(handler.processPredictRequest(dagName, modelVersion, modelVersionLabel, request, &response), ovms::StatusCode::REST_COULD_NOT_PARSE_INSTANCE);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "Predict", "TensorFlowServing", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "Predict", "TensorFlowServing", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "REST", "Predict", "TensorFlowServing", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "REST", "Predict", "TensorFlowServing", numberOfFailedRequests);    // ran by real request

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(4)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(2)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));
}

TEST_F(MetricFlowTest, RestGetModelMetadata) {
    HttpRestApiHandler handler(server, 0);

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string response;
        ASSERT_EQ(handler.processModelMetadataRequest(modelName, modelVersion, modelVersionLabel, &response), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string response;
        ASSERT_EQ(handler.processModelMetadataRequest(dagName, modelVersion, modelVersionLabel, &response), ovms::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "GetModelMetadata", "TensorFlowServing", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "GetModelMetadata", "TensorFlowServing", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, RestGetModelStatus) {
    HttpRestApiHandler handler(server, 0);

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string response;
        ASSERT_EQ(handler.processModelStatusRequest(modelName, modelVersion, modelVersionLabel, &response), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        std::string response;
        ASSERT_EQ(handler.processModelStatusRequest(dagName, modelVersion, modelVersionLabel, &response), ovms::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "GetModelStatus", "TensorFlowServing", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "GetModelStatus", "TensorFlowServing", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, RestModelInfer) {
    HttpRestApiHandler handler(server, 0);
    HttpRequestComponents components;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = modelName;
        std::string request = R"({"inputs":[{"name":"b","shape":[1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        components.model_name = modelName;
        std::string request = R"({{"inputs":[{"name":"b","shape":[1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::JSON_INVALID);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = dagName;
        std::string request = R"({"inputs":[{"name":"b","shape":[3,1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfFailedRequests; i++) {
        components.model_name = dagName;
        std::string request = R"({{"inputs":[{"name":"b","shape":[3,1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::JSON_INVALID);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelInfer", "KServe", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelInfer", "KServe", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "REST", "ModelInfer", "KServe", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "REST", "ModelInfer", "KServe", numberOfFailedRequests);    // ran by real request

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"gRPC\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(0)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_REQUEST_TIME + std::string{"_count{interface=\"REST\",name=\""} + dagName + std::string{"\",version=\"1\"} "} + std::to_string(numberOfSuccessRequests)));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFERENCE_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_WAIT_FOR_INFER_REQ_TIME + std::string{"_count{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(4)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_STREAMS + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + modelName + std::string{"\",version=\"1\"} "} + std::to_string(2)));
    EXPECT_THAT(server.collect(), Not(HasSubstr(METRIC_NAME_INFER_REQ_QUEUE_SIZE + std::string{"{name=\""} + dagName + std::string{"\",version=\"1\"} "})));
}

TEST_F(MetricFlowTest, RestModelMetadata) {
    HttpRestApiHandler handler(server, 0);
    HttpRequestComponents components;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = modelName;
        std::string request, response;
        ASSERT_EQ(handler.processModelMetadataKFSRequest(components, response, request), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = dagName;
        std::string request, response;
        ASSERT_EQ(handler.processModelMetadataKFSRequest(components, response, request), ovms::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelMetadata", "KServe", numberOfSuccessRequests);    // ran by real request
}

TEST_F(MetricFlowTest, ModelReady) {
    HttpRestApiHandler handler(server, 0);
    HttpRequestComponents components;

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = modelName;
        std::string request, response;
        ASSERT_EQ(handler.processModelReadyKFSRequest(components, response, request), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = dagName;
        std::string request, response;
        ASSERT_EQ(handler.processModelReadyKFSRequest(components, response, request), ovms::StatusCode::OK);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelReady", "KServe", numberOfSuccessRequests);    // ran by real request
}

std::string MetricFlowTest::prepareConfigContent() {
    return std::string{R"({
        "monitoring": {
            "metrics": {
                "enable": true,
                "metrics_list": [)"} +
           R"(")" + METRIC_NAME_INFER_REQ_QUEUE_SIZE +
           R"(",")" + METRIC_NAME_INFER_REQ_ACTIVE +
           R"(",")" + METRIC_NAME_CURRENT_REQUESTS +
           R"(",")" + METRIC_NAME_REQUESTS_SUCCESS +
           R"(",")" + METRIC_NAME_REQUESTS_FAIL +
           R"(",")" + METRIC_NAME_REQUEST_TIME +
           R"(",")" + METRIC_NAME_STREAMS +
           R"(",")" + METRIC_NAME_INFERENCE_TIME +
           R"(",")" + METRIC_NAME_WAIT_FOR_INFER_REQ_TIME +
           R"("]
            }
        },
        "model_config_list": [
            {"config": {
                    "name": "dummy",
                    "nireq": 2,
                    "plugin_config": {"CPU_THROUGHPUT_STREAMS": 4},
                    "base_path": "/ovms/src/test/dummy"}}
        ],
        "pipeline_config_list": [
            {
                "name": "dummy_demux",
                "inputs": [
                    "b"
                ],
                "demultiply_count": 0,
                "nodes": [
                    {
                        "name": "dummy-node",
                        "model_name": "dummy",
                        "type": "DL model",
                        "inputs": [
                            {"b": {
                                    "node_name": "request",
                                    "data_item": "b"}}],
                        "outputs": [
                            {"data_item": "a",
                                "alias": "a"}]
                    }
                ],
                "outputs": [
                    {"a": {
                            "node_name": "dummy-node",
                            "data_item": "a"}}
                ]
            }
        ]
    }
    )";
}
