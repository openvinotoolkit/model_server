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
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>

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
#include "test_http_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

// This is for Single Model and DAG.
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

#if (MEDIAPIPE_DISABLE == 0)
// This is for MediaPipe.
// This checks for counter to be present with exact value and other remaining metrics of the family to be 0.
static void checkMediapipeRequestsCounter(const std::string& collectedMetricData, const std::string& metricName, const std::string& endpointName, const std::string& interfaceName, const std::string& method, const std::string& api, int value) {
    for (std::string _interface : std::set<std::string>{"gRPC", "REST"}) {
        for (std::string _api : std::set<std::string>{"KServe", "V3"}) {
            if (_api == "KServe") {
                for (std::string _method : std::set<std::string>{"ModelInfer", "ModelInferStream"}) {
                    if (_interface == "REST")
                        continue;
                    std::stringstream ss;
                    ss << metricName << "{api=\"" << _api << "\",interface=\"" << _interface << "\",method=\"" << _method << "\",name=\"" << endpointName << "\"";
                    ss << "}";
                    int expectedValue = interfaceName == _interface && method == _method && api == _api ? value : 0;
                    ss << " " << expectedValue << "\n";
                    ASSERT_THAT(collectedMetricData, HasSubstr(ss.str()));
                }
            } else if (_interface == "REST") {  // V3 - only REST
                for (std::string _method : std::set<std::string>{"Unary", "Stream"}) {
                    std::stringstream ss;
                    ss << metricName << "{api=\"" << _api << "\",interface=\"" << _interface << "\",method=\"" << _method << "\",name=\"" << endpointName << "\"";
                    ss << "}";
                    int expectedValue = interfaceName == _interface && method == _method && api == _api ? value : 0;
                    ss << " " << expectedValue << "\n";
                    ASSERT_THAT(collectedMetricData, HasSubstr(ss.str()));
                }
            }
        }
    }
}

static void checkMediapipeRequestsCounterMetadataReady(const std::string& collectedMetricData, const std::string& metricName, const std::string& endpointName, const std::string& interfaceName, const std::string& method, const std::string& api, int value) {
    for (std::string _interface : std::set<std::string>{"gRPC", "REST"}) {
        for (std::string _method : std::set<std::string>{"ModelMetadata", "ModelReady"}) {
            std::stringstream ss;
            ss << metricName << "{api=\""
               << "KServe"
               << "\",interface=\"" << _interface << "\",method=\"" << _method << "\",name=\"" << endpointName << "\"";
            ss << "}";
            int expectedValue = interfaceName == _interface && method == _method ? value : 0;
            ss << " " << expectedValue << "\n";
            ASSERT_THAT(collectedMetricData, HasSubstr(ss.str()));
        }
    }
}
#endif

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
    const int numberOfAcceptedRequests = 11;
    const int numberOfRejectedRequests = 13;
    const int64_t dynamicBatch = 3;

    const Precision correctPrecision = Precision::FP32;
    const Precision wrongPrecision = Precision::I32;

    const std::string modelName = "dummy";
    const std::string dagName = "dummy_demux";
    const std::string mpName = "dummy_mp";
    const std::string negativeName = "negative";

    std::optional<int64_t> modelVersion = std::nullopt;
    std::optional<std::string_view> modelVersionLabel{std::nullopt};

    std::string prepareConfigContent();

    void unloadAllModels() {
        std::string content = R"(
            {
                "model_config_list": [],
                "pipeline_config_list": []
            }
        )";
        std::string fileToReload = this->directoryPath + "/config.json";
        createConfigFileWithContent(content, this->directoryPath + "/config.json");
        ASSERT_EQ(server.getManager().loadConfig(fileToReload), StatusCode::OK);
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        char* n_argv[] = {(char*)"ovms", (char*)"--config_path", (char*)"/unused", (char*)"--rest_port", (char*)"8080"};  // Workaround to have rest_port parsed in order to enable metrics
        int arg_count = 5;
        ovms::Config::instance().parse(arg_count, n_argv);
        std::string fileToReload = this->directoryPath + "/config.json";
        ASSERT_TRUE(createConfigFileWithContent(this->prepareConfigContent(), fileToReload));
        ASSERT_EQ(server.getManager().loadConfig(fileToReload), StatusCode::OK);
    }
};

TEST_F(MetricFlowTest, GrpcPredict) {
#ifdef _WIN32
    GTEST_SKIP() << "Test disabled on windows [SPORADIC] pipeline_config_list";
#endif
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

#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfAcceptedRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(mpName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    for (int i = 0; i < numberOfRejectedRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{"wrong_name", {DUMMY_MODEL_SHAPE, wrongPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(mpName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }
#endif

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelInfer", "KServe", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelInfer", "KServe", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "gRPC", "ModelInfer", "KServe", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "gRPC", "ModelInfer", "KServe", numberOfFailedRequests);    // ran by real request

#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_ACCEPTED, mpName, "gRPC", "ModelInfer", "KServe", numberOfAcceptedRequests);
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_REJECTED, mpName, "gRPC", "ModelInfer", "KServe", numberOfRejectedRequests);

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_RESPONSES, mpName, "gRPC", "ModelInfer", "KServe", numberOfAcceptedRequests);

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_PROCESSING_TIME + std::string{"_count{method=\"ModelInfer\",name=\""} + mpName + std::string{"\"} "} + std::to_string(numberOfAcceptedRequests)));
#endif

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

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, GrpcPredictGraphError) {
    KFSInferenceServiceImpl impl(server);
    ::KFSRequest request;
    ::KFSResponse response;
    size_t numberOfRequests = 3;
    for (size_t i = 0; i < numberOfRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(negativeName);
        ASSERT_NE(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_GRAPH_ERROR, negativeName, "gRPC", "ModelInfer", "KServe", numberOfRequests);

    for (size_t i = 0; i < numberOfRequests; i++) {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, correctPrecision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign(negativeName);
        ASSERT_NE(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_GRAPH_ERROR, negativeName, "gRPC", "ModelInfer", "KServe", 2 * numberOfRequests);
}
#endif

template <class W, class R>
class MockedServerReaderWriter final : public ::grpc::ServerReaderWriterInterface<W, R> {
public:
    MOCK_METHOD(void, SendInitialMetadata, (), (override));
    MOCK_METHOD(bool, NextMessageSize, (uint32_t * sz), (override));
    MOCK_METHOD(bool, Read, (R * msg), (override));
    MOCK_METHOD(bool, Write, (const W& msg, ::grpc::WriteOptions options), (override));
};

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, GrpcModelInferStream) {
    KFSInferenceServiceImpl impl(server);
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;

    using ::testing::_;
    using ::testing::Return;

    int counter = 0;
    inputs_info_t correctInputsMeta{{"in", {DUMMY_MODEL_SHAPE, correctPrecision}}};
    EXPECT_CALL(stream, Read(_))
        .WillRepeatedly([this, correctInputsMeta, &counter](::inference::ModelInferRequest* req) {
            if (counter >= this->numberOfAcceptedRequests)
                return false;
            preparePredictRequest(*req, correctInputsMeta);
            req->mutable_model_name()->assign(this->mpName);
            counter++;
            return true;
        });
    ON_CALL(stream, Write(_, _)).WillByDefault(Return(1));
    ASSERT_EQ(impl.ModelStreamInferImpl(nullptr, &stream), ovms::StatusCode::OK);

    counter = 0;
    inputs_info_t wrongInputsMeta{{"wrong_name", {DUMMY_MODEL_SHAPE, correctPrecision}}};
    EXPECT_CALL(stream, Read(_))
        .WillRepeatedly([this, wrongInputsMeta, &counter](::inference::ModelInferRequest* req) {
            if (counter >= this->numberOfRejectedRequests)
                return false;
            preparePredictRequest(*req, wrongInputsMeta);
            req->mutable_model_name()->assign(this->mpName);
            counter++;
            return true;
        });
    ON_CALL(stream, Write(_, _)).WillByDefault(Return(1));
    ASSERT_EQ(impl.ModelStreamInferImpl(nullptr, &stream), ovms::StatusCode::OK);

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_ACCEPTED, mpName, "gRPC", "ModelInferStream", "KServe", numberOfAcceptedRequests);
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_REJECTED, mpName, "gRPC", "ModelInferStream", "KServe", numberOfRejectedRequests);
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_RESPONSES, mpName, "gRPC", "ModelInferStream", "KServe", numberOfAcceptedRequests);
}
#endif

TEST_F(MetricFlowTest, GrpcModelMetadata) {
    KFSInferenceServiceImpl impl(server);
    ::KFSModelMetadataRequest request;
    KFSModelMetadataResponse response;
    KFSModelExtraMetadata extraMetadata;

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
#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(mpName);
        ASSERT_EQ(impl.ModelMetadata(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }
#endif
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelMetadata", "KServe", numberOfSuccessRequests);    // ran by real request
#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounterMetadataReady(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, mpName, "gRPC", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
#endif
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

#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        request.Clear();
        response.Clear();
        request.mutable_name()->assign(mpName);
        ASSERT_EQ(impl.ModelReady(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    }
#endif
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "gRPC", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "gRPC", "ModelReady", "KServe", numberOfSuccessRequests);    // ran by real request
#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounterMetadataReady(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, mpName, "gRPC", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
#endif
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

#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfAcceptedRequests; i++) {
        components.model_name = mpName;
        std::string request = R"({"inputs":[{"name":"in","shape":[3,1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::OK);
    }

    for (int i = 0; i < numberOfRejectedRequests; i++) {
        components.model_name = mpName;
        std::string request = R"({"inputs":[{"name":"wrong_name","shape":[3,1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::INVALID_UNEXPECTED_INPUT);
    }
#endif

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelInfer", "KServe", dynamicBatch * numberOfSuccessRequests + numberOfSuccessRequests);  // ran by demultiplexer + real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelInfer", "KServe", numberOfSuccessRequests);                                             // ran by real request

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "REST", "ModelInfer", "KServe", numberOfFailedRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, dagName, 1, "REST", "ModelInfer", "KServe", numberOfFailedRequests);    // ran by real request

#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_ACCEPTED, mpName, "REST", "ModelInfer", "KServe", numberOfAcceptedRequests);
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_REJECTED, mpName, "REST", "ModelInfer", "KServe", numberOfRejectedRequests);

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_RESPONSES, mpName, "REST", "ModelInfer", "KServe", numberOfAcceptedRequests);

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_PROCESSING_TIME + std::string{"_count{method=\"ModelInfer\",name=\""} + mpName + std::string{"\"} "} + std::to_string(numberOfAcceptedRequests)));
#endif

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

TEST_F(MetricFlowTest, RestModelInferOnUnloadedModel) {
    this->unloadAllModels();

    HttpRestApiHandler handler(server, 0);
    HttpRequestComponents components;

    const int numberOfRequests = 5;

    for (int i = 0; i < numberOfRequests; i++) {
        components.model_name = modelName;
        components.model_version = 1;  // This is required to ensure we request specific version which is unloaded
        std::string request = R"({"inputs":[{"name":"b","shape":[1,10],"datatype":"FP32","data":[1,2,3,4,5,6,7,8,9,10]}], "parameters":{"binary_data_output":true}})";
        std::string response;
        std::optional<int> inferenceHeaderContentLength;
        ASSERT_EQ(handler.processInferKFSRequest(components, response, request, inferenceHeaderContentLength), ovms::StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE);
    }

    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelInfer", "KServe", 0);
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_FAIL, modelName, 1, "REST", "ModelInfer", "KServe", numberOfRequests);
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
#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = mpName;
        std::string request, response;
        ASSERT_EQ(handler.processModelMetadataKFSRequest(components, response, request), ovms::StatusCode::OK);
    }
#endif
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelMetadata", "KServe", numberOfSuccessRequests);    // ran by real request
#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounterMetadataReady(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, mpName, "REST", "ModelMetadata", "KServe", numberOfSuccessRequests);  // ran by real request
#endif
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
#if (MEDIAPIPE_DISABLE == 0)
    for (int i = 0; i < numberOfSuccessRequests; i++) {
        components.model_name = mpName;
        std::string request, response;
        ASSERT_EQ(handler.processModelReadyKFSRequest(components, response, request), ovms::StatusCode::OK);
    }
#endif
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, modelName, 1, "REST", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
    checkRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, dagName, 1, "REST", "ModelReady", "KServe", numberOfSuccessRequests);    // ran by real request
#if (MEDIAPIPE_DISABLE == 0)
    checkMediapipeRequestsCounterMetadataReady(server.collect(), METRIC_NAME_REQUESTS_SUCCESS, mpName, "REST", "ModelReady", "KServe", numberOfSuccessRequests);  // ran by real request
#endif
}

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, RestV3Unary) {
    HttpRestApiHandler handler(server, 0);
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();

    EXPECT_CALL(*stream, IsDisconnected())
        .WillRepeatedly(::testing::Return(false));

    for (int i = 0; i < numberOfAcceptedRequests; i++) {
        std::string request = R"({"model": "dummy_gpt", "prompt": "Hello World"})";
        std::string response;
        HttpRequestComponents comps;
        auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
        auto status = handler.processV3("/v3/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
        status = handler.processV3("/v3/v1/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::OK) << status.string();
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_ACCEPTED, "dummy_gpt", "REST", "Unary", "V3", numberOfAcceptedRequests * 2);
    // checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_REJECTED, "dummy_gpt", "REST", "Unary", "V3", numberOfRejectedRequests);
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_RESPONSES, "dummy_gpt", "REST", "Unary", "V3", numberOfAcceptedRequests * 2);

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_PROCESSING_TIME + std::string{"_count{method=\"Unary\",name=\""} + "dummy_gpt" + std::string{"\"} "} + std::to_string(numberOfAcceptedRequests * 2)));
}
#endif

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, RestV3UnaryError) {
    HttpRestApiHandler handler(server, 0);
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);

    EXPECT_CALL(*stream, IsDisconnected())
        .WillRepeatedly(::testing::Return(false));

    size_t numberOfRequests = 3;

    for (size_t i = 0; i < numberOfRequests; i++) {
        std::string request = R"({"model": "dummy_gpt", "prompt":"ReturnError"})";
        std::string response;
        HttpRequestComponents comps;
        auto status = handler.processV3("/v3/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
        status = handler.processV3("/v3/v1/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_GRAPH_ERROR, "dummy_gpt", "REST", "Unary", "V3", numberOfRequests * 2);
}
#endif

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, RestV3Stream) {
    HttpRestApiHandler handler(server, 0);
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    ON_CALL(*stream, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential

    EXPECT_CALL(*stream, IsDisconnected())
        .WillRepeatedly(::testing::Return(false));

    for (int i = 0; i < numberOfAcceptedRequests; i++) {
        std::string request = R"({"model": "dummy_gpt", "stream": true, "prompt": "Hello World"})";
        std::string response;
        HttpRequestComponents comps;
        auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);
        auto status = handler.processV3("/v3/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::PARTIAL_END) << status.string();
        status = handler.processV3("/v3/v1/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::PARTIAL_END) << status.string();
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_ACCEPTED, "dummy_gpt", "REST", "Stream", "V3", numberOfAcceptedRequests * 2);
    // checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_REQUESTS_REJECTED, "dummy_gpt", "REST", "Stream", "V3", numberOfRejectedRequests);
    const int numberOfMockedChunksPerRequest = 9;  // Defined in openai_chat_completions_mock_calculator.cpp
    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_RESPONSES, "dummy_gpt", "REST", "Stream", "V3", numberOfAcceptedRequests * numberOfMockedChunksPerRequest * 2);

    EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_PROCESSING_TIME + std::string{"_count{method=\"Stream\",name=\""} + "dummy_gpt" + std::string{"\"} "} + std::to_string(numberOfAcceptedRequests * 2)));

    SPDLOG_ERROR(server.collect());
}
#endif

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, RestV3StreamError) {
    HttpRestApiHandler handler(server, 0);
    std::shared_ptr<MockedServerRequestInterface> stream = std::make_shared<MockedServerRequestInterface>();
    auto streamPtr = std::static_pointer_cast<ovms::HttpAsyncWriter>(stream);

    ON_CALL(*stream, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
    EXPECT_CALL(*stream, IsDisconnected())
        .WillRepeatedly(::testing::Return(false));

    size_t numberOfRequests = 3;

    for (size_t i = 0; i < numberOfRequests; i++) {
        std::string request = R"({"model": "dummy_gpt", "stream": true, "prompt": "ReturnError"})";
        std::string response;
        HttpRequestComponents comps;
        auto status = handler.processV3("/v3/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::PARTIAL_END) << status.string();
        status = handler.processV3("/v3/v1/completions", comps, response, request, streamPtr);
        ASSERT_EQ(status, ovms::StatusCode::PARTIAL_END) << status.string();
    }

    checkMediapipeRequestsCounter(server.collect(), METRIC_NAME_GRAPH_ERROR, "dummy_gpt", "REST", "Stream", "V3", numberOfRequests * 2);
    SPDLOG_ERROR(server.collect());
}
#endif

#if (MEDIAPIPE_DISABLE == 0)
TEST_F(MetricFlowTest, CurrentGraphs) {
    using ::testing::_;
    using ::testing::Return;

    KFSInferenceServiceImpl impl(server);
    const size_t numberOfWorkloads = 5;
    std::atomic<size_t> numberOfCurrentlyFinishedWorkloads{0};
    std::vector<std::thread> threads;
    std::condition_variable cv;
    std::mutex mtx;

    for (size_t i = 0; i < numberOfWorkloads; i++) {
        threads.emplace_back(std::thread([this, &impl, numberOfWorkloads, &numberOfCurrentlyFinishedWorkloads, &cv, &mtx]() -> void {
            MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
            int counter = 0;
            inputs_info_t correctInputsMeta{{"in1", {DUMMY_MODEL_SHAPE, this->correctPrecision}}};
            EXPECT_CALL(stream, Read(_))
                .WillRepeatedly([this, correctInputsMeta, &counter, numberOfWorkloads, &numberOfCurrentlyFinishedWorkloads, &cv, &mtx](::inference::ModelInferRequest* req) {
                    if (counter >= this->numberOfAcceptedRequests) {
                        if (++numberOfCurrentlyFinishedWorkloads >= numberOfWorkloads) {
                            // Check the metric. The graph requires 2 inputs in order to start processing and we deliver only 1.
                            // This way we ensure that X graphs are created (wait for second input)
                            // Before we disconnect (return false) we can check if the metric is equal to number of graphs (X)
                            // X=numberOfWorkloads
                            EXPECT_THAT(server.collect(), HasSubstr(METRIC_NAME_CURRENT_GRAPHS + std::string{"{name=\"multi_input_synchronized_graph\"} "} + std::to_string(numberOfWorkloads)));
                            cv.notify_all();
                            return false;  // disconnect
                        }

                        // Wait for finished workloads to be =numberOfWorkloads
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [&numberOfCurrentlyFinishedWorkloads, numberOfWorkloads]() {
                            return numberOfCurrentlyFinishedWorkloads >= numberOfWorkloads;
                        });
                        return false;  // disconnect
                    }
                    preparePredictRequest(*req, correctInputsMeta);
                    req->mutable_model_name()->assign("multi_input_synchronized_graph");
                    counter++;
                    return true;
                });
            ON_CALL(stream, Write(_, _)).WillByDefault(Return(1));
            ASSERT_EQ(impl.ModelStreamInferImpl(nullptr, &stream), ovms::StatusCode::OK);
        }));
    }

    for (size_t i = 0; i < numberOfWorkloads; i++) {
        threads[i].join();
    }
}
#endif

// Test MP metrics when mediapipe is enabled at build time
#if (MEDIAPIPE_DISABLE == 0)
std::string MetricFlowTest::prepareConfigContent() {
    auto configContent = std::string{R"({
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
                         R"(",")" + METRIC_NAME_CURRENT_GRAPHS +
                         R"(",")" + METRIC_NAME_REQUESTS_ACCEPTED +
                         R"(",")" + METRIC_NAME_REQUESTS_REJECTED +
                         R"(",")" + METRIC_NAME_RESPONSES +
                         R"(",")" + METRIC_NAME_GRAPH_ERROR +
                         R"(",")" + METRIC_NAME_PROCESSING_TIME +
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
        ],
        "mediapipe_config_list": [
            {
                "name":"dummy_mp",
                "graph_path":"/ovms/src/test/mediapipe/graphkfspass.pbtxt"
            },
            {
                "name": "dummy_gpt",
                "graph_path": "/ovms/src/test/mediapipe/graph_gpt.pbtxt"
            },
            {
                "name": "multi_input_synchronized_graph",
                "graph_path": "/ovms/src/test/mediapipe/two_input_graph.pbtxt"
            },
            {
                "name": "negative",
                "graph_path": "/ovms/src/test/mediapipe/negative/graph_error.pbtxt"
            }
        ]
    }
    )";
    adjustConfigForTargetPlatform(configContent);
    return configContent;
}
#else
// Do not test MP metrics when mediapipe is disabled at build time
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
           R"(",")" + METRIC_NAME_CURRENT_GRAPHS +
           R"(",")" + METRIC_NAME_REQUESTS_ACCEPTED +
           R"(",")" + METRIC_NAME_REQUESTS_REJECTED +
           R"(",")" + METRIC_NAME_RESPONSES +
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
#endif
