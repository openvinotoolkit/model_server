//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <openvino/runtime/tensor.hpp>
#include <sys/stat.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/calculators/ovms/modelapiovmsadapter.hpp"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_graph_executor_impl.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../ovms_exit_codes.hpp"
#include "../precision.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

#if (PYTHON_DISABLE == 0)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

#include "../python/pythonnoderesources.hpp"
namespace py = pybind11;
using namespace py::literals;
#endif

using namespace ovms;

using testing::HasSubstr;
using testing::Not;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

class MediapipeCliFlowTest : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";

    void SetUpServer(const char* graphPath, const char* graphName) {
        ::SetUpServer(this->t, this->server, this->port, getGenericFullPathForSrcTest(graphPath).c_str(), graphName);
    }

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, getGenericFullPathForSrcTest(configPath).c_str());
    }

    void SetUp() override {
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class MediapipeCliFlowTestNegative : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";
};

class MediapipeCliFlowTestDummy : public MediapipeCliFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/cli", "graphkfspass");
    }
};

class MediapipeCliFlowTestDummyModelMesh : public MediapipeCliFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/model_mesh/cli", "graphkfspass");
    }
};

class MediapipeConfigFlowTestDummyModelMesh : public MediapipeCliFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/model_mesh/config.json");
    }
};

class MediapipeConfigFlowTestDummyModelMeshNegative : public MediapipeCliFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/model_mesh/Nonexisting/config.json");
    }
};

class MediapipeCliFlowTestDummyRelative : public MediapipeCliFlowTest {
public:
    std::filesystem::path originalCwd;
    void SetUp() {
        // Workaround for bazel test execution from /root/ or bazel-out directory
        originalCwd = std::filesystem::current_path();
#ifdef __linux__
        std::filesystem::path newCwd = "/ovms";
        std::filesystem::current_path(newCwd);
#endif
        SetUpServer("src/test/mediapipe/cli", "graphkfspass");
        std::filesystem::current_path(originalCwd);
    }
};

TEST_F(MediapipeCliFlowTestNegative, UnsupportedCliParamBatchSize) {
    server.setShutdownRequest(0);
    randomizePort(this->port);
    char* argv[] = {(char*)"ovms",
        (char*)"--model_name",
        (char*)"graphkfspass",
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/cli").c_str(),
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--batch_size",
        (char*)"10"};
    int argc = 9;
    t.reset(new std::thread([&argc, &argv, this]() {
        EXPECT_EQ(OVMS_EX_USAGE, server.start(argc, argv));
    }));

    server.setShutdownRequest(1);
    t->join();
}

void Infer(ovms::Server& server) {
    const Precision precision = Precision::FP32;
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphkfspass";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    std::vector<float> requestData2{0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // Checking that KFSPASS calculator copies requestData1 to the response so that we expect requestData1 on output
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}

TEST_F(MediapipeCliFlowTestDummy, Infer) {
    Infer(server);
}

TEST_F(MediapipeCliFlowTestDummyModelMesh, Infer) {
    Infer(server);
}

TEST_F(MediapipeConfigFlowTestDummyModelMesh, Infer) {
    Infer(server);
}

TEST_F(MediapipeCliFlowTestDummyRelative, Infer) {
    Infer(server);
}

class MediapipeFlowTest : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, getGenericFullPathForSrcTest(configPath).c_str());
    }

    void SetUp() override {
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class MediapipeFlowAddTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    }
};

class MediapipeFlowKfsTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_kfs.json");
    }
};

class MediapipeTFTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mp_tf_passthrough.json");
    }
};

class MediapipeTensorTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/config_mp_passthrough.json");
    }
};

#if (PYTHON_DISABLE == 0)
class MediapipePyTensorOvTensorConverterTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_pytensor_ovtensor_converter.json");
    }
};
class MediapipeOvTensorPyTensorConverterTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_ovtensor_pytensor_converter.json");
    }
};
#endif

class MediapipeTfLiteTensorTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/config_tflite_passthrough.json");
    }
};

class MediapipeEmbeddingsTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/embeddings/config_embeddings.json");
    }
};

template <class W, class R>
class MockedServerReaderWriter final : public ::grpc::ServerReaderWriterInterface<W, R> {
public:
    MOCK_METHOD(void, SendInitialMetadata, (), (override));
    MOCK_METHOD(bool, NextMessageSize, (uint32_t * sz), (override));
    MOCK_METHOD(bool, Read, (R * msg), (override));
    MOCK_METHOD(bool, Write, (const W& msg, ::grpc::WriteOptions options), (override));
};

TEST_F(MediapipeEmbeddingsTest, startup) {
    EnsureServerStartedWithTimeout(server, 5);
    const ovms::Module* servableModule = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    ASSERT_TRUE(servableModule != nullptr);
    ModelManager* manager = &dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();
    auto mediapipeGraphDefinition = manager->getMediapipeFactory().findDefinitionByName("embeddings");
    ASSERT_TRUE(mediapipeGraphDefinition != nullptr);
    ASSERT_TRUE(mediapipeGraphDefinition->getStatus().isAvailable());
}

TEST_F(MediapipeEmbeddingsTest, grpcInference) {
    EnsureServerStartedWithTimeout(server, 5);
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "embeddings";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"input", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    preparePredictRequest(request, inputsMeta, requestData1);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(MediapipeFlowKfsTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeDummyKFS";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    std::vector<float> requestData2{0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // Checking that KFSPASS calculator copies requestData1 to the response so that we expect requestData1 on output
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}

#if (PYTHON_DISABLE == 0)
TEST_F(MediapipePyTensorOvTensorConverterTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipePyTensorOvTensorConverter";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    ASSERT_EQ(response.model_name(), "mediapipePyTensorOvTensorConverter");
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), "out") << "Did not find:"
                                                         << "out";
    const auto& output = *response.outputs().begin();
    const std::string& content = response.raw_output_contents(0);

    ASSERT_EQ(content.size(), 10 * sizeof(float));
    ASSERT_EQ(output.shape_size(), 2);
    ASSERT_EQ(output.shape(0), 1);
    ASSERT_EQ(output.shape(1), 10);
    ASSERT_EQ(output.datatype(), "FP32");

    const float* actualOutput = (const float*)content.data();
    float* expectedOutput = requestData1.data();
    const int dataLengthToCheck = 10 * sizeof(float);
    EXPECT_EQ(actualOutput[0], expectedOutput[0]);
    EXPECT_EQ(0, std::memcmp(actualOutput, expectedOutput, dataLengthToCheck))
        << readableError(expectedOutput, actualOutput, dataLengthToCheck / sizeof(float));
}

TEST_F(MediapipeOvTensorPyTensorConverterTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeOvTensorPyTensorConverter";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    ASSERT_EQ(response.model_name(), "mediapipeOvTensorPyTensorConverter");
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), "out") << "Did not find:"
                                                         << "out";
    const auto& output = *response.outputs().begin();
    const std::string& content = response.raw_output_contents(0);

    ASSERT_EQ(content.size(), 10 * sizeof(float));
    ASSERT_EQ(output.shape_size(), 2);
    ASSERT_EQ(output.shape(0), 1);
    ASSERT_EQ(output.shape(1), 10);

    const float* actualOutput = (const float*)content.data();
    float* expectedOutput = requestData1.data();
    const int dataLengthToCheck = 10 * sizeof(float);
    EXPECT_EQ(actualOutput[0], expectedOutput[0]);
    EXPECT_EQ(0, std::memcmp(actualOutput, expectedOutput, dataLengthToCheck))
        << readableError(expectedOutput, actualOutput, dataLengthToCheck / sizeof(float));
}
#endif

TEST_F(MediapipeTFTest, Passthrough) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName{"mpTfsPassthrough"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    size_t dummysInTheGraph = 0;
    checkDummyResponse("out", requestData, request, response, dummysInTheGraph, 1, modelName);
}

TEST_F(MediapipeTFTest, DummyInfer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName{"mpTFDummy"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    size_t dummysInTheGraph = 1;
    checkDummyResponse("out", requestData, request, response, dummysInTheGraph, 1, modelName);
}

TEST_F(MediapipeTensorTest, DummyInfer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"mpTensorDummy"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    size_t dummysInTheGraph = 1;
    checkDummyResponse("out", requestData, request, response, dummysInTheGraph, 1, modelName);
}

TEST_F(MediapipeTfLiteTensorTest, DummyInfer) {
    GTEST_SKIP() << "OVMS calculator doesn't handle TfLite on output. Only vector of TfLite"
                 << "OVMS deserialization & serialization of TfLiteTensors is not finished as well";
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"mpTfLiteTensorDummy"};
    request.Clear();
    response.Clear();
    // TfLite tensors don't hold batch size dimension so we send shape [10] instead of default dummy's [1, 10]
    inputs_info_t inputsMeta{{"in", {{10}, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    EXPECT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    size_t dummysInTheGraph = 1;
    checkDummyResponse("out", requestData, request, response, dummysInTheGraph, 1, modelName);
}

// Incorrect KServe proto to TFTensor conversion
TEST_F(MediapipeTFTest, SendDummyInferMoreDataThanExpected) {
    const std::string modelName{"mpTFDummy"};
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    request.Clear();
    response.Clear();
    const size_t numElements = 50000;
    inputs_info_t inputsMeta{{"in", {{1, numElements}, precision}}};
    std::vector<float> requestData(numElements);
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    request.mutable_inputs(0)->set_shape(1, 1);  // change only shape [1,numElements] to [1,1], keep data
    ASSERT_NE(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
}

// Scalar in KServe proto to TFTensor conversion
TEST_F(MediapipeTFTest, DummyInferScalar) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"mpTFScalar"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {{1}, precision}}};
    std::vector<float> requestData{7.1f};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_inputs(0)->clear_shape();  // imitate scalar
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), "out") << "Did not find:out";
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(content->size(), sizeof(float));
    ASSERT_EQ(output_proto.shape_size(), 0);
}

// 0-data KServe proto to TFTensor conversion
TEST_F(MediapipeTFTest, DummyInferZeroData) {
    const std::string modelName{"mpTFDummy"};
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {{1, 0}, precision}}};
    std::vector<float> requestData;
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), "out") << "Did not find:out";
    const auto& output_proto = *response.outputs().begin();
    std::string* content = response.mutable_raw_output_contents(0);

    ASSERT_EQ(content->size(), 0);
    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), 1);
    ASSERT_EQ(output_proto.shape(1), 0);
}

class MediapipeFlowDummyTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full.json");
    }
};
class MediapipeFlowDummyNegativeTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_nonexistent_calculator.json");
    }
};
class MediapipeFlowScalarTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_scalar.json");
    }
};

class MediapipeFlowDynamicZeroDimTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_dynamic.json");
    }
};
class MediapipeFlowDummyPathsRelativeToBasePathTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_relative_to_base_path.json");
    }
};

class MediapipeFlowDummyNoGraphPathTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_no_graph_path.json");
    }
};

class MediapipeFlowDummyOnlyGraphNameSpecified : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/graph_only_name/config_mediapipe_dummy_adapter_full_only_name_specified.json");
    }
};

class MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfig : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/graph_only_name/config_mediapipe_dummy_adapter_full_only_name_specified_in_model_config.json");
    }
};

class MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfigNoBase : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/graph_only_name/config_mediapipe_dummy_adapter_full_only_name_specified_in_model_config_no_base.json");
    }
};

class MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfigNoBaseMeshCase : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/graph_mesh_case/config_mediapipe_dummy_adapter_full_only_name_specified_in_model_config_no_base.json");
    }
};

class MediapipeFlowDummySubconfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_subconfig.json");
    }
};

class MediapipeFlowDummyDefaultSubconfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_default_subconfig.json");
    }
};

static size_t convertKFSDataTypeToMatFormat(const KFSDataType& datatype) {
    static std::unordered_map<KFSDataType, size_t> datatypeFormatMap{
        {"UINT8", CV_8U},
        {"UINT16", CV_16U},
        {"INT8", CV_8U},
        {"INT16", CV_16U},
        {"INT32", CV_16U},
        {"FP32", CV_32F}};
    // CV_16F and CV_64F are not supported in Mediapipe::ImageFrame
    auto it = datatypeFormatMap.find(datatype);
    if (it == datatypeFormatMap.end()) {
        SPDLOG_DEBUG("Converting KFS datatype to mat format failed. Mat format will be set to default - CV_8U");
        return CV_8U;
    }
    return it->second;
}

class MediapipeFlowImageInput : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_image_input.json");
    }

    void PerformTestWithGivenDatatype(KFSDataType datatype) {
        const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
        KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
        ::KFSRequest request;
        ::KFSResponse response;
        const std::string modelName = "mediapipeImageInput";
        request.Clear();
        response.Clear();
        cv::Mat imageRaw = cv::imread(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), cv::IMREAD_UNCHANGED);
        ASSERT_TRUE(!imageRaw.empty());
        cv::Mat image;
        size_t matFormat = convertKFSDataTypeToMatFormat(datatype);
        imageRaw.convertTo(image, matFormat);

        KFSTensorInputProto* input = request.add_inputs();
        input->set_name("in");
        input->set_datatype(datatype);
        input->mutable_shape()->Clear();
        input->add_shape(image.rows);
        input->add_shape(image.cols);
        input->add_shape(image.channels());

        std::string* content = request.add_raw_input_contents();
        size_t elementSize = image.elemSize1();
        content->resize(image.cols * image.rows * image.channels() * elementSize);
        std::memcpy(content->data(), image.data, image.cols * image.rows * image.channels() * elementSize);
        request.mutable_model_name()->assign(modelName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
        ASSERT_EQ(response.model_name(), modelName);
        ASSERT_EQ(response.outputs_size(), 1);
        ASSERT_EQ(response.outputs()[0].shape().size(), 3);
        ASSERT_EQ(response.outputs()[0].shape()[0], image.cols);
        ASSERT_EQ(response.outputs()[0].shape()[1], image.rows);
        ASSERT_EQ(response.outputs()[0].shape()[2], image.channels());
        ASSERT_EQ(response.raw_output_contents_size(), 1);
        ASSERT_EQ(response.raw_output_contents()[0].size(), image.cols * image.rows * image.channels() * elementSize);
        ASSERT_EQ(0, memcmp(response.raw_output_contents()[0].data(), image.data, image.cols * image.rows * image.channels() * elementSize));
    }

    void PerformTestWithGivenDatatypeOneChannel(KFSDataType datatype) {
        const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
        KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
        ::KFSRequest request;
        ::KFSResponse response;
        const std::string modelName = "mediapipeImageInput";
        request.Clear();
        response.Clear();
        cv::Mat imageRaw = cv::imread(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/grayscale.jpg"), cv::IMREAD_UNCHANGED);
        ASSERT_TRUE(!imageRaw.empty());
        cv::Mat grayscaled;
        size_t matFormat = convertKFSDataTypeToMatFormat(datatype);
        imageRaw.convertTo(grayscaled, matFormat);

        KFSTensorInputProto* input = request.add_inputs();
        input->set_name("in");
        input->set_datatype(datatype);
        input->mutable_shape()->Clear();
        input->add_shape(grayscaled.rows);
        input->add_shape(grayscaled.cols);
        input->add_shape(grayscaled.channels());

        std::string* content = request.add_raw_input_contents();
        size_t elementSize = grayscaled.elemSize1();
        content->resize(grayscaled.cols * grayscaled.rows * grayscaled.channels() * elementSize);
        std::memcpy(content->data(), grayscaled.data, grayscaled.cols * grayscaled.rows * grayscaled.channels() * elementSize);
        request.mutable_model_name()->assign(modelName);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
        ASSERT_EQ(response.model_name(), modelName);
        ASSERT_EQ(response.outputs_size(), 1);
        ASSERT_EQ(response.outputs()[0].shape()[0], grayscaled.cols);
        ASSERT_EQ(response.outputs()[0].shape()[1], grayscaled.rows);
        ASSERT_EQ(response.outputs()[0].shape()[2], grayscaled.channels());
        ASSERT_EQ(response.raw_output_contents_size(), 1);
        ASSERT_EQ(response.raw_output_contents()[0].size(), grayscaled.cols * grayscaled.rows * grayscaled.channels() * elementSize);
        ASSERT_EQ(0, memcmp(response.raw_output_contents()[0].data(), grayscaled.data, grayscaled.cols * grayscaled.rows * grayscaled.channels() * elementSize));
    }
};

TEST_F(MediapipeFlowImageInput, InvalidInputName) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    request.Clear();
    response.Clear();

    request.mutable_model_name()->assign(modelName);
    KFSTensorInputProto* input = request.add_inputs();
    input->set_name("invalid");
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeFlowImageInput, NoInputs) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    request.Clear();
    response.Clear();

    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeFlowImageInput, InvalidShape) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    request.Clear();
    response.Clear();
    cv::Mat imageRaw = cv::imread(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), cv::IMREAD_UNCHANGED);
    ASSERT_TRUE(!imageRaw.empty());
    cv::Mat image;
    size_t matFormat = convertKFSDataTypeToMatFormat("UINT8");
    imageRaw.convertTo(image, matFormat);
    std::string* content = request.add_raw_input_contents();
    size_t elementSize = image.elemSize1();
    content->resize(image.cols * image.rows * image.channels() * elementSize);
    std::memcpy(content->data(), image.data, image.cols * image.rows * image.channels() * elementSize);

    KFSTensorInputProto* input = request.add_inputs();
    input->set_name("in");
    input->set_datatype("UINT8");
    input->mutable_shape()->Clear();
    input->add_shape(2);

    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeFlowImageInput, InvalidShapes) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    response.Clear();
    request.mutable_model_name()->assign(modelName);

    KFSTensorInputProto* input = request.add_inputs();
    request.add_raw_input_contents();
    input->set_name("in");
    input->set_datatype("UINT8");
    input->mutable_shape()->Clear();
    input->add_shape(3);                                     // cols
    input->add_shape(3);                                     // rows
    input->add_shape(3);                                     // channels
    for (auto dimIndex : std::vector<size_t>{0, 1, 2}) {     // h/w/c
        for (auto dimValue : std::vector<int64_t>{0, -5}) {  // zero and negative
            input->set_shape(dimIndex, dimValue);
            EXPECT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT) << " for dim index: " << dimIndex;
        }
    }
}

TEST_F(MediapipeFlowImageInput, InvalidDatatype) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    request.Clear();
    response.Clear();
    cv::Mat imageRaw = cv::imread(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), cv::IMREAD_UNCHANGED);
    ASSERT_TRUE(!imageRaw.empty());
    cv::Mat image;
    size_t matFormat = convertKFSDataTypeToMatFormat("INT64");
    imageRaw.convertTo(image, matFormat);
    std::string* content = request.add_raw_input_contents();
    size_t elementSize = image.elemSize1();
    content->resize(image.cols * image.rows * image.channels() * elementSize);
    std::memcpy(content->data(), image.data, image.cols * image.rows * image.channels() * elementSize);

    KFSTensorInputProto* input = request.add_inputs();
    input->set_name("in");
    input->set_datatype("INT64");
    input->mutable_shape()->Clear();
    input->add_shape(image.cols);
    input->add_shape(image.rows);
    input->add_shape(3);

    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeFlowImageInput, Float32_4Channels) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeImageInput";
    request.Clear();
    response.Clear();
    cv::Mat imageRaw = cv::imread(getGenericFullPathForSrcTest("/ovms/src/test/binaryutils/rgb4x4.jpg"), cv::IMREAD_UNCHANGED);
    ASSERT_TRUE(!imageRaw.empty());
    cv::Mat imageFP32;
    imageRaw.convertTo(imageFP32, CV_32F);
    cv::Mat image;
    cv::cvtColor(imageFP32, image, cv::COLOR_BGR2BGRA);

    KFSTensorInputProto* input = request.add_inputs();
    input->set_name("in");
    input->set_datatype("FP32");
    input->mutable_shape()->Clear();
    input->add_shape(image.rows);
    input->add_shape(image.cols);
    input->add_shape(image.channels());

    std::string* content = request.add_raw_input_contents();
    size_t elementSize = image.elemSize1();
    content->resize(image.cols * image.rows * image.channels() * elementSize);
    std::memcpy(content->data(), image.data, image.cols * image.rows * image.channels() * elementSize);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.outputs()[0].shape().size(), 3);
    ASSERT_EQ(response.outputs()[0].shape()[0], image.cols);
    ASSERT_EQ(response.outputs()[0].shape()[1], image.rows);
    ASSERT_EQ(response.outputs()[0].shape()[2], image.channels());
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.raw_output_contents()[0].size(), image.cols * image.rows * image.channels() * elementSize);
    ASSERT_EQ(0, memcmp(response.raw_output_contents()[0].data(), image.data, image.cols * image.rows * image.channels() * elementSize));
}

class MediapipeFlowImageInputThreeChannels : public MediapipeFlowImageInput {};

TEST_P(MediapipeFlowImageInputThreeChannels, Infer) {
    std::string datatype = GetParam();
    if (datatype == "FP32") {
        GTEST_SKIP_("Unsupported precision?");
    }
    PerformTestWithGivenDatatype(datatype);
}

static const std::vector<std::string> PRECISIONS{
    // "FP64",
    "FP32",
    // "FP16",
    "UINT8",
    "UINT16",
    "INT8",
    "INT16",
    // "INT32",
};

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    MediapipeFlowImageInputThreeChannels,
    ::testing::ValuesIn(PRECISIONS),
    [](const ::testing::TestParamInfo<MediapipeFlowImageInputThreeChannels::ParamType>& info) {
        return info.param;
    });

class MediapipeFlowImageInputOneChannel : public MediapipeFlowImageInput {};

TEST_P(MediapipeFlowImageInputOneChannel, Infer) {
    std::string datatype = GetParam();
    PerformTestWithGivenDatatypeOneChannel(datatype);
}

INSTANTIATE_TEST_SUITE_P(
    TestDeserialize,
    MediapipeFlowImageInputOneChannel,
    ::testing::ValuesIn(PRECISIONS),
    [](const ::testing::TestParamInfo<MediapipeFlowImageInputOneChannel::ParamType>& info) {
        return info.param;
    });

class MediapipeFlowDummyEmptySubconfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_empty_subconfig.json");
    }
};

static void performMediapipeInfer(const ovms::Server& server, ::KFSRequest& request, ::KFSResponse& response, const Precision& precision, const std::string& modelName) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
}

TEST_F(MediapipeFlowDummyOnlyGraphNameSpecified, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphdummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfig, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphdummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfigNoBase, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphdummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}
TEST_F(MediapipeFlowDummyOnlyGraphNameSpecifiedInModelConfigNoBaseMeshCase, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphdummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyDefaultSubconfigTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyEmptySubconfigTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyPathsRelativeToBasePathTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummySubconfigTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

class MediapipeFlowTwoOutputsTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_two_outputs.json");
    }
};

TEST_F(MediapipeFlowTwoOutputsTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeDummyTwoOutputs";
    performMediapipeInfer(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 2);
    ASSERT_EQ(response.raw_output_contents_size(), 2);
    const auto& output_proto_1 = response.outputs().Get(0);
    std::string* content = response.mutable_raw_output_contents(0);
    ASSERT_EQ(content->size(), DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto_1.shape_size(), 2);
    ASSERT_EQ(output_proto_1.shape(0), 1);
    ASSERT_EQ(output_proto_1.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    const int seriesLength = 1;
    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

    float* actual_output = (float*)content->data();
    float* expected_output = responseData.data();
    int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));

    const auto& output_proto_2 = response.outputs().Get(1);
    content = response.mutable_raw_output_contents(1);

    ASSERT_EQ(content->size(), DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto_2.shape_size(), 2);
    ASSERT_EQ(output_proto_2.shape(0), 1);
    ASSERT_EQ(output_proto_2.shape(1), DUMMY_MODEL_OUTPUT_SIZE);

    ASSERT_TRUE((output_proto_1.name() == "out_1" && output_proto_2.name() == "out_2") ||
                (output_proto_1.name() == "out_2" && output_proto_2.name() == "out_1"));

    actual_output = (float*)content->data();
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

class MediapipeFlowTwoOutputsDagTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_two_outputs_dag.json");
    }
};

TEST_F(MediapipeFlowTwoOutputsDagTest, Infer) {
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> factors{1, 3, 2, 2};

    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeTwoOutputsDag";
    request.mutable_inputs()->Clear();
    request.mutable_raw_input_contents()->Clear();
    prepareKFSInferInputTensor(request, "in_1", {{1, 10}, ovms::Precision::FP32}, input, false);
    prepareKFSInferInputTensor(request, "in_2", {{1, 4}, ovms::Precision::FP32}, factors, false);
    ASSERT_EQ(request.inputs_size(), 2);
    request.mutable_model_name()->assign(modelName);
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);

    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 2);
    ASSERT_EQ(response.raw_output_contents_size(), 2);

    ASSERT_TRUE((response.outputs().Get(0).name() == "out_1" && response.outputs().Get(1).name() == "out_2") ||
                (response.outputs().Get(0).name() == "out_2" && response.outputs().Get(1).name() == "out_1"));

    std::string* content1;
    std::string* content2;
    KFSTensorOutputProto outputProto1, outputProto2;
    if (response.outputs().Get(0).name() == "out_1") {
        outputProto1 = response.outputs().Get(0);
        content1 = response.mutable_raw_output_contents(0);
        outputProto2 = response.outputs().Get(1);
        content2 = response.mutable_raw_output_contents(1);
    } else {
        outputProto1 = response.outputs().Get(1);
        content1 = response.mutable_raw_output_contents(1);
        outputProto2 = response.outputs().Get(0);
        content2 = response.mutable_raw_output_contents(0);
    }

    int out1DataSize = 40;
    int out2DataSize = 16;
    std::vector<float> out1Data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
    std::vector<float> out2Data{1, 3, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2};

    ASSERT_EQ(content1->size(), out1DataSize * sizeof(float));
    ASSERT_EQ(outputProto1.shape_size(), 3);
    ASSERT_EQ(outputProto1.shape(0), 4);
    ASSERT_EQ(outputProto1.shape(1), 1);
    ASSERT_EQ(outputProto1.shape(2), 10);

    float* actual_output = (float*)content1->data();
    float* expected_output = out1Data.data();
    int dataLengthToCheck = out1DataSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));

    ASSERT_EQ(content2->size(), out2DataSize * sizeof(float));
    ASSERT_EQ(outputProto2.shape_size(), 3);
    ASSERT_EQ(outputProto2.shape(0), 4);
    ASSERT_EQ(outputProto2.shape(1), 1);
    ASSERT_EQ(outputProto2.shape(2), 4);

    actual_output = (float*)content2->data();
    expected_output = out2Data.data();
    dataLengthToCheck = out2DataSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

class MediapipeFlowDummyDummyInSubconfigAndConfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_dummy_in_both_config_and_subconfig.json");
    }
};

TEST_F(MediapipeFlowDummyDummyInSubconfigAndConfigTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaDummy";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {{1, 12}, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 12);
}

TEST_F(MediapipeFlowDummyNoGraphPathTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "graphdummy";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_P(MediapipeFlowDummyTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyNegativeTest, NegativeShouldNotReachInferDueToNonexistentCalculator) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaDummyNonexistentCaclulator";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
}

TEST_F(MediapipeFlowDummyNegativeTest, NegativeShouldNotReachInferStreamDueToNonexistentCalculator) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaDummyNonexistentCaclulator";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([request](::inference::ModelInferRequest* req) {
            *req = request;
            return true;  // sending 1st request with wrong endpoint name
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_)).Times(0);
    ASSERT_EQ(impl.ModelStreamInferImpl(nullptr, &stream), StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(MediapipeFlowScalarTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaScalar";
    request.Clear();
    response.Clear();
    // Empty shape is used in the test framework to generate default shape (usually dummy 2d (1,10))
    // Here we generate (1,1) tensor which has the same data size as scalar and just reshape to scalar () below.
    inputs_info_t inputsMeta{{"in", {{1, 1}, precision}}};
    preparePredictRequest(request, inputsMeta);
    auto* content = request.mutable_raw_input_contents()->Mutable(0);
    ASSERT_EQ(content->size(), sizeof(float));
    *((float*)content->data()) = 3.8f;
    ASSERT_EQ(request.inputs_size(), 1);
    (*request.mutable_inputs())[0].clear_shape();  // scalar
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);

    const std::string outputName = "out";
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* outContent = response.mutable_raw_output_contents(0);

    ASSERT_EQ(output_proto.shape_size(), 0);

    ASSERT_EQ(outContent->size(), sizeof(float));
    EXPECT_EQ(*(float*)outContent->data(), 3.8f);
}

// KServe proto to OVTensor conversion
TEST_F(MediapipeFlowDynamicZeroDimTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaDummy";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {{2, 0}, precision}}};
    preparePredictRequest(request, inputsMeta);
    auto* content = request.mutable_raw_input_contents()->Mutable(0);
    ASSERT_EQ(content->size(), 0);
    ASSERT_EQ(request.inputs_size(), 1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);

    const std::string outputName = "out";
    ASSERT_EQ(response.model_name(), modelName);
    ASSERT_EQ(response.outputs_size(), 1);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.outputs().begin()->name(), outputName) << "Did not find:" << outputName;
    const auto& output_proto = *response.outputs().begin();
    std::string* outContent = response.mutable_raw_output_contents(0);

    ASSERT_EQ(output_proto.shape_size(), 2);
    ASSERT_EQ(output_proto.shape(0), 2);
    ASSERT_EQ(output_proto.shape(1), 0);

    ASSERT_EQ(outContent->size(), 0);
}

TEST_P(MediapipeFlowAddTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in1", {DUMMY_MODEL_SHAPE, precision}},
        {"in2", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
    std::vector<float> requestData2{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.set_id("my_id");
    request.mutable_model_name()->assign(modelName);
    auto status = impl.ModelInfer(nullptr, &request, &response);
    ASSERT_EQ(status.error_code(), grpc::StatusCode::OK) << status.error_message();
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
    ASSERT_EQ(response.id(), "my_id");
}

const std::vector<std::string> mediaGraphsDummy{"mediaDummy",
    "mediaDummyADAPTFULL"};
const std::vector<std::string> mediaGraphsAdd{"mediapipeAdd",
    "mediapipeAddADAPTFULL"};

class MediapipeStreamFlowAddTest : public MediapipeFlowAddTest {
protected:
    constexpr static const size_t NUM_REQUESTS{3};
    const std::string modelName = mediaGraphsAdd[1];
    ::KFSRequest request[NUM_REQUESTS];
    ::KFSResponse response[NUM_REQUESTS];
    std::vector<float> requestData1[NUM_REQUESTS];

    void SetUp() override {
        MediapipeFlowAddTest::SetUp();
        for (size_t i = 0; i < NUM_REQUESTS; i++) {
            this->request[i].Clear();
            this->response[i].Clear();
        }
        inputs_info_t inputsMeta{
            {"in1", {DUMMY_MODEL_SHAPE, precision}},
            {"in2", {DUMMY_MODEL_SHAPE, precision}}};
        this->requestData1[0] = {3., 7., 1., 6., 4., 2., 0, 5., 9., 8.};
        this->requestData1[1] = {6., 1., 4., 2., 0., 1., 9, 8., 9., 2.};
        this->requestData1[2] = {4., 2., 0., 1., 9., 8., 5, 1., 4., 6.};
        for (size_t i = 0; i < NUM_REQUESTS; i++) {
            preparePredictRequest(this->request[i], inputsMeta, this->requestData1[i]);
            this->request[i].mutable_model_name()->assign(modelName);
        }
    }

    MediapipeGraphDefinition* getMPDefinitionByName(const std::string& name) {
        const ServableManagerModule* smm = dynamic_cast<const ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
        ModelManager& modelManager = smm->getServableManager();
        const MediapipeFactory& factory = modelManager.getMediapipeFactory();
        return factory.findDefinitionByName(name);
    }
};

// Smoke test - send multiple requests with ov::Tensor, receive multiple responses
// Gets the executor from model manager
TEST_F(MediapipeStreamFlowAddTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // correct sending 1st request
        })
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[1];
            return true;  // correct sending 2nd request
        })
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[2];
            return true;  // correct sending 3rd request
        })
        .WillOnce([](::inference::ModelInferRequest* req) {
            return false;  // disconnection
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_))
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect first response
            checkAddResponse("out", this->requestData1[0], this->requestData1[0], this->request[0], msg.infer_response(), 1, 1, this->modelName);
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect second response
            checkAddResponse("out", this->requestData1[1], this->requestData1[1], this->request[1], msg.infer_response(), 1, 1, this->modelName);
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect third and no more responses
            checkAddResponse("out", this->requestData1[2], this->requestData1[2], this->request[2], msg.infer_response(), 1, 1, this->modelName);
            return true;
        });
    auto status = impl.ModelStreamInferImpl(nullptr, &stream);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
}

// Inference on unloaded mediapipe graph
// Expect old stream to continue responding until closure
// Expect new stream to be rejected
TEST_F(MediapipeStreamFlowAddTest, InferOnUnloadedGraph) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    const ServableManagerModule* smm = dynamic_cast<const ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
    ModelManager& modelManager = smm->getServableManager();

    auto* definition = this->getMPDefinitionByName(this->modelName);
    ASSERT_NE(definition, nullptr);

    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    std::promise<void> startUnloading;
    std::promise<void> finishedUnloading;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // correct sending 1st request
        })
        .WillOnce([this, &finishedUnloading](::inference::ModelInferRequest* req) {
            *req = this->request[1];
            // Second Read() operation will wait, until graph unloading is finished
            finishedUnloading.get_future().get();
            return true;  // correct sending 2nd request
        })
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[2];
            return true;  // correct sending 3rd request
        })
        .WillOnce([](::inference::ModelInferRequest* req) {
            return false;  // disconnection
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_))
        .WillOnce([this, &startUnloading](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect first response
            checkAddResponse("out", this->requestData1[0], this->requestData1[0], this->request[0], msg.infer_response(), 1, 1, this->modelName);
            // notify that we should start unloading (first request is processed and response is sent)
            startUnloading.set_value();
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect second response
            checkAddResponse("out", this->requestData1[1], this->requestData1[1], this->request[1], msg.infer_response(), 1, 1, this->modelName);
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect third and no more responses
            checkAddResponse("out", this->requestData1[2], this->requestData1[2], this->request[2], msg.infer_response(), 1, 1, this->modelName);
            return true;
        });
    std::thread unloader([&startUnloading, &finishedUnloading, &definition, &modelManager]() {
        // Wait till first response notifies that we should start unloading
        startUnloading.get_future().get();
        definition->retire(modelManager);
        // Notify second request to arrive because we unloaded the graph
        finishedUnloading.set_value();
    });
    auto status = impl.ModelStreamInferImpl(nullptr, &stream);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    unloader.join();

    // Opening new stream, expect graph to be unavailable
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> newStream;
    EXPECT_CALL(newStream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // sending 1st request which should fail creating new graph
        });
    EXPECT_CALL(newStream, Write(::testing::_, ::testing::_)).Times(0);
    status = impl.ModelStreamInferImpl(nullptr, &newStream);
    ASSERT_EQ(status, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE) << status.string();
}

// Inference on reloaded mediapipe graph, completely different pipeline
// Expects old stream to still use old configuration
// Expect new stream to use new configuration XXXXXX
TEST_F(MediapipeStreamFlowAddTest, InferOnReloadedGraph) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    const ServableManagerModule* smm = dynamic_cast<const ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
    ModelManager& modelManager = smm->getServableManager();

    auto* definition = this->getMPDefinitionByName(this->modelName);
    ASSERT_NE(definition, nullptr);

    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    std::promise<void> startReloading;
    std::promise<void> finishedReloading;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // correct sending 1st request
        })
        .WillOnce([this, &finishedReloading](::inference::ModelInferRequest* req) {
            *req = this->request[1];
            // Second Read() operation will wait, until graph reloading is finished
            finishedReloading.get_future().get();
            return true;  // correct sending 2nd request
        })
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[2];
            return true;  // correct sending 3rd request
        })
        .WillOnce([](::inference::ModelInferRequest* req) {
            return false;  // disconnection
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_))
        .WillOnce([this, &startReloading](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect first response
            checkAddResponse("out", this->requestData1[0], this->requestData1[0], this->request[0], msg.infer_response(), 1, 1, this->modelName);
            // notify that we should start reloading (first request is processed and response is sent)
            startReloading.set_value();
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect second response
            checkAddResponse("out", this->requestData1[1], this->requestData1[1], this->request[1], msg.infer_response(), 1, 1, this->modelName);
            return true;
        })
        .WillOnce([this](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            // expect third and no more responses
            checkAddResponse("out", this->requestData1[2], this->requestData1[2], this->request[2], msg.infer_response(), 1, 1, this->modelName);
            return true;
        });
    std::thread reloader([&startReloading, &finishedReloading, &definition, &modelManager, this]() {
        // Wait till first response notifies that we should start reloading
        startReloading.get_future().get();
        MediapipeGraphConfig mgc{
            this->modelName,
            "",                                                                             // default base path
            getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/graphscalar_tf.pbtxt"),  // graphPath - valid but includes missing models, will fail for new streams
            "",                                                                             // default subconfig path
            ""                                                                              // dummy md5
        };
        auto status = definition->reload(modelManager, mgc);
        ASSERT_EQ(status, StatusCode::OK) << status.string();
        // Notify second request to arrive because we unloaded the graph
        finishedReloading.set_value();
    });
    auto status = impl.ModelStreamInferImpl(nullptr, &stream);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    reloader.join();

    // Opening new stream, expect new graph to be available but errors in processing
    std::promise<void> canDisconnect;
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> newStream;
    EXPECT_CALL(newStream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // sending 1st request which should fail creating new graph
        })
        .WillOnce([&canDisconnect](::inference::ModelInferRequest* req) {
            canDisconnect.get_future().get();
            return false;
        });
    EXPECT_CALL(newStream, Write(::testing::_, ::testing::_))
        .WillOnce([&canDisconnect](const ::inference::ModelStreamInferResponse& msg, ::grpc::WriteOptions options) {
            [&msg]() {
                const auto& outputs = msg.infer_response().outputs();
                ASSERT_EQ(outputs.size(), 0);
                ASSERT_EQ(msg.error_message(), Status(StatusCode::INVALID_UNEXPECTED_INPUT).string() + " - in1 is unexpected; partial deserialization of first request");
            }();
            canDisconnect.set_value();
            return true;
        });
    status = impl.ModelStreamInferImpl(nullptr, &newStream);
    ASSERT_EQ(status, StatusCode::MEDIAPIPE_PRECONDITION_FAILED) << status.string();
}

TEST_F(MediapipeStreamFlowAddTest, NegativeShouldNotReachInferDueToRetiredGraph) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    const ServableManagerModule* smm = dynamic_cast<const ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME));
    ModelManager& modelManager = smm->getServableManager();
    auto* definition = this->getMPDefinitionByName(this->modelName);
    ASSERT_NE(definition, nullptr);
    definition->retire(modelManager);

    // Opening new stream, expect graph to be unavailable
    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([this](::inference::ModelInferRequest* req) {
            *req = this->request[0];
            return true;  // sending 1st request which should fail creating new graph
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_)).Times(0);
    auto status = impl.ModelStreamInferImpl(nullptr, &stream);
    ASSERT_EQ(status, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE) << status.string();
}

TEST_P(MediapipeFlowAddTest, InferStreamDisconnectionBeforeFirstRequest) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();

    MockedServerReaderWriter<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest> stream;
    EXPECT_CALL(stream, Read(::testing::_))
        .WillOnce([](::inference::ModelInferRequest* req) {
            return false;  // immediate disconnection
        });
    EXPECT_CALL(stream, Write(::testing::_, ::testing::_)).Times(0);
    auto status = impl.ModelStreamInferImpl(nullptr, &stream);
    ASSERT_EQ(status, StatusCode::MEDIAPIPE_UNINITIALIZED_STREAM_CLOSURE) << status.string();
}

TEST_F(MediapipeFlowTest, InferWithParams) {
    GTEST_SKIP() << "Not possible with graph queue";
    return;
    SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_graph_with_side_packets.json");
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaWithParams";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in_not_used", {{1, 1}, ovms::Precision::I32}}};
    std::vector<float> requestData{0.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    // here add params
    const std::string stringParamValue = "abecadlo";
    const bool boolParamValue = true;
    const int64_t int64ParamValue = 42;
    request.mutable_parameters()->operator[]("string_param").set_string_param(stringParamValue);
    request.mutable_parameters()->operator[]("bool_param").set_bool_param(boolParamValue);
    request.mutable_parameters()->operator[]("int64_param").set_int64_param(int64ParamValue);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    // here check outputs
    ASSERT_EQ(outputs.size(), 3);
    // 1st string
    auto it = response.outputs().begin();
    size_t outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_string") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "UINT8");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), stringParamValue.size());
        const std::string& content = response.raw_output_contents(outputId);
        SPDLOG_ERROR("Received output size:{} content:{}", content.size(), content);
        EXPECT_EQ(content, stringParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
    it = response.outputs().begin();
    outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_bool") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "BOOL");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), 1);
        const std::string& content = response.raw_output_contents(outputId);
        ASSERT_EQ(content.size(), sizeof(bool));
        const bool castContent = *((bool*)content.data());
        SPDLOG_ERROR("Received output size:{} content:{}; castContent:{}", content.size(), content, castContent);
        EXPECT_EQ(castContent, boolParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
    it = response.outputs().begin();
    outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_int64") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "INT64");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), 1);
        const std::string& content = response.raw_output_contents(outputId);
        ASSERT_EQ(content.size(), sizeof(int64_t));
        const int64_t castContent = *((int64_t*)content.data());
        SPDLOG_ERROR("Received output size:{} content:{}; castContent:{}", content.size(), content, castContent);
        EXPECT_EQ(castContent, int64ParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
}

TEST_F(MediapipeFlowTest, InferWithRestrictedParamName) {
    SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_graph_with_side_packets.json");
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    for (auto restrictedParamName : std::vector<std::string>{"py"}) {
        ::KFSRequest request;
        ::KFSResponse response;
        const std::string modelName = "mediaWithParams";
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{
            {"in_not_used", {{1, 1}, ovms::Precision::I32}}};
        std::vector<float> requestData{0.};
        preparePredictRequest(request, inputsMeta, requestData);
        request.mutable_model_name()->assign(modelName);
        // here add params
        const std::string stringParamValue = "abecadlo";
        const bool boolParamValue = true;
        const int64_t int64ParamValue = 42;
        request.mutable_parameters()->operator[]("string_param").set_string_param(stringParamValue);
        request.mutable_parameters()->operator[]("bool_param").set_bool_param(boolParamValue);
        request.mutable_parameters()->operator[]("int64_param").set_int64_param(int64ParamValue);
        request.mutable_parameters()->operator[](restrictedParamName).set_int64_param(int64ParamValue);
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    }
}

using testing::ElementsAre;

TEST_F(MediapipeFlowAddTest, AdapterMetadata) {
    mediapipe::ovms::OVMSInferenceAdapter adapter("add");
    const std::shared_ptr<const ov::Model> model;
    ov::Core unusedCore;
    ov::AnyMap notUsedAnyMap;
    adapter.loadModel(model, unusedCore, "NOT_USED", notUsedAnyMap);
    EXPECT_THAT(adapter.getInputNames(), ElementsAre(SUM_MODEL_INPUT_NAME_1, SUM_MODEL_INPUT_NAME_2));
    EXPECT_THAT(adapter.getOutputNames(), ElementsAre(SUM_MODEL_OUTPUT_NAME));
    EXPECT_EQ(adapter.getInputShape(SUM_MODEL_INPUT_NAME_1), ov::Shape({1, 10}));
    EXPECT_EQ(adapter.getInputShape(SUM_MODEL_INPUT_NAME_2), ov::Shape({1, 10}));
}

TEST_F(MediapipeFlowTest, AdapterMetadataDynamicShape) {
    SetUpServer("/ovms/src/test/configs/config_dummy_dynamic_shape.json");
    mediapipe::ovms::OVMSInferenceAdapter adapter("dummy");
    const std::shared_ptr<const ov::Model> model;
    ov::Core unusedCore;
    ov::AnyMap notUsedAnyMap;
    adapter.loadModel(model, unusedCore, "NOT_USED", notUsedAnyMap);
    EXPECT_THAT(adapter.getInputNames(), ElementsAre(DUMMY_MODEL_INPUT_NAME));
    EXPECT_THAT(adapter.getOutputNames(), ElementsAre(DUMMY_MODEL_OUTPUT_NAME));
    EXPECT_EQ(adapter.getInputShape(DUMMY_MODEL_INPUT_NAME), ov::PartialShape({1, {1, 10}}));
}

namespace {
class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {
    }
    ov::AnyMap getRTInfo() override {
        std::vector<std::string> mockLabels;
        for (size_t i = 0; i < 5; i++) {
            mockLabels.emplace_back(std::to_string(i));
        }
        ov::AnyMap modelInfo = {
            {"layout", "data:HWCN"},
            {"resize_type", "unnatural"},
            {"labels", mockLabels}};
        ov::AnyMap rtInfo = {{"model_info", modelInfo}};
        return rtInfo;
    }
};

class MockModel : public ovms::Model {
public:
    MockModel(const std::string& name) :
        Model(name, false /*stateful*/, nullptr) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t, ov::Core& ieCore, ovms::MetricRegistry* registry = nullptr, const ovms::MetricConfig* metricConfig = nullptr) override {
        return std::make_shared<MockModelInstance>(ieCore);
    }
};

class MockModelManager : public ovms::ModelManager {
    ovms::MetricRegistry registry;

public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return std::make_shared<MockModel>(name);
    }

public:
    MockModelManager(const std::string& modelCacheDirectory = "") :
        ovms::ModelManager(modelCacheDirectory, &registry) {
    }
    ~MockModelManager() {
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        join();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        models.clear();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    }
};

class MockedServableManagerModule : public ovms::ServableManagerModule {
    mutable MockModelManager mockModelManager;

public:
    MockedServableManagerModule(ovms::Server& ovmsServer) :
        ovms::ServableManagerModule(ovmsServer) {
    }
    ModelManager& getServableManager() const override {
        return mockModelManager;
    }
};

class MockedServer : public Server {
public:
    MockedServer() = default;
    std::unique_ptr<Module> createModule(const std::string& name) override {
        if (name != ovms::SERVABLE_MANAGER_MODULE_NAME)
            return Server::createModule(name);
        return std::make_unique<MockedServableManagerModule>(*this);
    };
    Module* getModule(const std::string& name) {
        return const_cast<Module*>(Server::getModule(name));
    }
};

}  // namespace

TEST(Mediapipe, AdapterRTInfo) {
    MockedServer server;
    OVMS_Server* cserver = reinterpret_cast<OVMS_Server*>(&server);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    std::string port{"5555"};
    randomizePort(port);
    uint32_t portNum = ovms::stou32(port).value();
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, portNum));
    // we will use dummy model that will have mocked rt_info
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config.json").c_str()));

    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    const std::string mockedModelName = "dummy";
    uint32_t servableVersion = 1;
    mediapipe::ovms::OVMSInferenceAdapter adapter(mockedModelName, servableVersion, cserver);
    const std::shared_ptr<const ov::Model> model;
    /*ov::AnyMap configuration = {
        {"layout", "data:HWCN"},
        {"resize_type", "unnatural"},
        {"labels", mockLabels}*/
    ov::Core unusedCore;
    ov::AnyMap notUsedAnyMap;
    adapter.loadModel(model, unusedCore, "NOT_USED", notUsedAnyMap);
    ov::AnyMap modelConfig = adapter.getModelConfig();

    auto checkModelInfo = [](const ov::AnyMap& modelConfig) {
        std::cout << "Model config size: " << modelConfig.size() << std::endl;
        ASSERT_EQ(modelConfig.size(), 3);
        auto it = modelConfig.find("resize_type");
        ASSERT_NE(modelConfig.end(), it);
        EXPECT_EQ(std::string("unnatural"), it->second.as<std::string>());
        it = modelConfig.find("layout");
        ASSERT_NE(modelConfig.end(), it);
        ASSERT_EQ(std::string("data:HWCN"), it->second.as<std::string>());
        it = modelConfig.find("labels");
        ASSERT_NE(modelConfig.end(), it);
        const std::vector<std::string>& resultLabels = it->second.as<std::vector<std::string>>();
        EXPECT_THAT(resultLabels, ElementsAre("0", "1", "2", "3", "4"));
    };
    checkModelInfo(modelConfig);

    OVMS_ServableMetadata* servableMetadata = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, mockedModelName.c_str(), servableVersion, &servableMetadata));

    const ov::AnyMap* servableMetadataRtInfo;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInfo(servableMetadata, reinterpret_cast<const void**>(&servableMetadataRtInfo)));
    ASSERT_NE(nullptr, servableMetadataRtInfo);
    checkModelInfo((*servableMetadataRtInfo).at("model_info").as<ov::AnyMap>());
    OVMS_ServableMetadataDelete(servableMetadata);
}

TEST(Mediapipe, MetadataDummy) {
    ConstructorEnabledModelManager manager;
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/graphdummy.pbtxt").c_str()};
    ovms::MediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc);
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    tensor_map_t inputs = mediapipeDummy.getInputsInfo();
    tensor_map_t outputs = mediapipeDummy.getOutputsInfo();
    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("in"), inputs.end());
    ASSERT_NE(outputs.find("out"), outputs.end());
    const auto& input = inputs.at("in");
    EXPECT_EQ(input->getShape(), Shape({}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::UNDEFINED);
    const auto& output = outputs.at("out");
    EXPECT_EQ(output->getShape(), Shape({}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::UNDEFINED);
}

TEST(Mediapipe, MetadataDummyInputTypes) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "TEST:in"
    input_stream: "TEST33:in2"
    output_stream: "TEST0:out"
    output_stream: "TEST1:out2"
    output_stream: "TEST3:out3"
        node {
            calculator: "OVMSOVCalculator"
            input_stream: "B:in"
            output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
        node {
            calculator: "OVMSOVCalculator"
            input_stream: "B:in2"
            output_stream: "A:out2"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
        node {
            calculator: "OVMSOVCalculator"
            input_stream: "B:in2"
            output_stream: "A:out3"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    tensor_map_t inputs = mediapipeDummy.getInputsInfo();
    tensor_map_t outputs = mediapipeDummy.getOutputsInfo();
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(outputs.size(), 3);
    ASSERT_NE(inputs.find("in"), inputs.end());
    ASSERT_NE(outputs.find("out"), outputs.end());
    const auto& input = inputs.at("in");
    EXPECT_EQ(input->getShape(), Shape({}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::UNDEFINED);
    const auto& output = outputs.at("out");
    EXPECT_EQ(output->getShape(), Shape({}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::UNDEFINED);
}

TEST(Mediapipe, MetadataExistingInputNames) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "TEST:in"
    input_stream: "TEST33:in"
    output_stream: "TEST0:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME);
}

TEST(Mediapipe, MetadataExistingOutputNames) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "TEST:in"
    output_stream: "TEST0:out"
    output_stream: "TEST1:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME);
}

TEST(Mediapipe, MetadataMissingResponseOutputTypes) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "TEST3:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_OUTPUT_RESPONSE_TAG);
}

TEST(Mediapipe, MetadataMissingRequestInputTypes) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "TEST:in"
    output_stream: "RESPONSE:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_INPUT_REQUEST_TAG);
}

TEST(Mediapipe, MetadataNegativeWrongInputTypes) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "wrong:REQUEST:in"
    output_stream: "number:test3:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST(Mediapipe, MetadataNegativeWrongOutputTypes) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "TEST:in"
    output_stream: "TEST:TEST:out"
        node {
        calculator: "OVMSOVCalculator"
        input_stream: "B:in"
        output_stream: "A:out"
            node_options: {
                [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                  servable_name: "dummyUpper"
                  servable_version: "1"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST(Mediapipe, MetadataEmptyConfig) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = "";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

const std::vector<std::string> mediaGraphsKfs{"mediapipeDummyKFS"};

class MediapipeNoTagMapping : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";
    void SetUpServer(const char* configPath) {
        server.setShutdownRequest(0);
        randomizePort(this->port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str(),
            (char*)"--log_level",
            (char*)"DEBUG"};
        int argc = 7;
        t.reset(new std::thread([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        EnsureServerStartedWithTimeout(server, 5);
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        TestWithTempDir::TearDown();
    }
};

TEST_F(MediapipeNoTagMapping, DummyUppercase) {
    // Here we use dummy with uppercase input/output
    // and we shouldn't need tag mapping
    ConstructorEnabledModelManager manager;
    // create config file
    std::string configJson = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummyUpper",
                "base_path": "/ovms/src/test/dummyUppercase"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeDummyUppercase",
        "graph_path":"PATH_TO_REPLACE"
    }
    ]
})";
    const std::string pathToReplace = "PATH_TO_REPLACE";
    auto it = configJson.find(pathToReplace);
    ASSERT_NE(it, std::string::npos);
    const std::string graphPbtxt = R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "dummyUpper"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "B:in"
  output_stream: "A:out"
})";
    const std::string pbtxtPath = this->directoryPath + "/graphDummyUppercase.pbtxt";
    createConfigFileWithContent(graphPbtxt, pbtxtPath);
    configJson.replace(it, pathToReplace.size(), pbtxtPath);

    const std::string configJsonPath = this->directoryPath + "/subconfig.json";
    adjustConfigForTargetPlatform(configJson);
    createConfigFileWithContent(configJson, configJsonPath);
    this->SetUpServer(configJsonPath.c_str());
    // INFER
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeDummyUppercase";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

class MediapipeConfig : public MediapipeFlowTest {
public:
    void TearDown() override {}
};

const std::string NAME = "Name";
TEST_P(MediapipeConfig, MediapipeGraphDefinitionNonExistentFile) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    MediapipeGraphConfig mgc{"noname", basePath + "NONEXISTENT_FILE"};
    MediapipeGraphDefinition mgd(NAME, mgc);
    EXPECT_EQ(mgd.validate(manager), StatusCode::FILE_INVALID);
}

TEST_P(MediapipeConfig, MediapipeAdd) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + "test/mediapipe/config_mediapipe_add_adapter_full.json";
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsAdd) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        ASSERT_NE(graphDefinition, nullptr);
        EXPECT_EQ(graphDefinition->getStatus().isAvailable(), true);
    }

    manager.join();
}

TEST_P(MediapipeConfig, MediapipeDummyWithDag) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + "test/mediapipe/config_mediapipe_dummy_adapter_full_dag.json";
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsDummy) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        ASSERT_NE(graphDefinition, nullptr);
        EXPECT_EQ(graphDefinition->getStatus().isAvailable(), true);
    }

    auto pipelineDefinition = manager.getPipelineFactory().findDefinitionByName("dummyDAG");
    EXPECT_NE(pipelineDefinition, nullptr);
    EXPECT_EQ(pipelineDefinition->getStatus().getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);

    auto model = manager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_P(MediapipeConfig, MediapipeFullRelativePaths) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + "test/mediapipe/relative_paths/config_relative_dummy.json";
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("graph1");
    ASSERT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("graph2");
    ASSERT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);

    manager.join();
}

TEST_P(MediapipeConfig, MediapipeFullRelativePathsSubconfig) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + "test/mediapipe/relative_paths/config_relative_add_subconfig.json";
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("graph1");
    ASSERT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("graph2");
    ASSERT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_P(MediapipeConfig, MediapipeFullRelativePathsSubconfigBasePath) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + "test/mediapipe/relative_paths/config_relative_dummy_subconfig_base_path.json";
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("graphaddadapterfull");
    ASSERT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("graphadd");
    ASSERT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_P(MediapipeConfig, MediapipeFullRelativePathsNegative) {
    ConstructorEnabledModelManager manager;
    std::string basePath = GetParam();
    std::replace(basePath.begin(), basePath.end(), 'X', '/');
    basePath = basePath + std::string("test/mediapipe/relative_paths/config_relative_dummy_negative.json");
    auto status = manager.startFromFile(getGenericFullPathForSrcTest(basePath));
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    ASSERT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), false);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    ASSERT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), false);

    manager.join();
}

// Run with config file provided in absolute and relative path
// X is changed to / after in a test to work around the fact that / is rejected in parameter
const std::vector<std::string> basePaths{"XovmsXsrcX", "srcX"};
INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeConfig,
    ::testing::ValuesIn(basePaths),
    [](const ::testing::TestParamInfo<MediapipeConfig::ParamType>& info) {
        return info.param;
    });

class MediapipeConfigChanges : public TestWithTempDir {
    void SetUp() override {
        TestWithTempDir::SetUp();
    }

public:
    static const std::string mgdName;
    static const std::string configFileWithGraphPathToReplace;
    static const std::string configFileWithGraphPathToReplaceWithoutModel;
    static const std::string configFileWithGraphPathToReplaceAndSubconfig;
    static const std::string configFileWithEmptyBasePath;
    static const std::string configFileWithNoBasePath;
    static const std::string configFileWithoutGraph;
    static const std::string pbtxtContent;
    static const std::string pbtxtContentNonexistentCalc;
    template <typename Request, typename Response>
    static void checkStatus(ModelManager& manager, ovms::StatusCode code) {
        std::shared_ptr<MediapipeGraphExecutor> executor;
        Request request;
        Response response;
        auto status = manager.createPipeline(executor, mgdName);
        EXPECT_EQ(status, code) << status.string();
    }
};
const std::string MediapipeConfigChanges::mgdName{"mediapipeGraph"};
const std::string MediapipeConfigChanges::configFileWithGraphPathToReplace = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithEmptyBasePath = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "base_path":""
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithNoBasePath = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithGraphPathToReplaceAndSubconfig = R"(
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ",
        "subconfig":"SUBCONFIG_PATH"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithGraphPathToReplaceWithoutModel = R"(
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithoutGraph = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ]
}
)";

const std::string MediapipeConfigChanges::pbtxtContent = R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "B:in"
  output_stream: "A:out"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
      tag_to_input_tensor_names {
        key: "B"
        value: "b"
      }
      tag_to_output_tensor_names {
        key: "A"
        value: "a"
      }
    }
  }
}
)";

const std::string MediapipeConfigChanges::pbtxtContentNonexistentCalc = R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "OpenVINOModelServerSessionCalculatorNONEXISTENT"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "B:in"
  output_stream: "A:out"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
      tag_to_input_tensor_names {
        key: "B"
        value: "b"
      }
      tag_to_output_tensor_names {
        key: "A"
        value: "a"
      }
    }
  }
}
)";
TEST_F(MediapipeConfigChanges, AddProperGraphThenChangeInputNameInDefinition) {
    std::string graphPbtxtFileContent = pbtxtContent;
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";

    const std::string inputName{"in\""};
    const std::string newInputName{"in2\""};

    // Start with initial input name
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);

    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(graphPbtxtFileContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    auto model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    EXPECT_EQ(definition->getInputsInfo().count("in"), 1);
    EXPECT_EQ(definition->getInputsInfo().count("in2"), 0);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);

    // now change the input name in graph.pbtxt and trigger config reload
    graphPbtxtFileContent.replace(graphPbtxtFileContent.find(inputName), inputName.size(), newInputName);
    graphPbtxtFileContent.replace(graphPbtxtFileContent.find(inputName), inputName.size(), newInputName);
    createConfigFileWithContent(graphPbtxtFileContent, graphFilePath);

    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    EXPECT_EQ(definition->getInputsInfo().count("in"), 0);
    EXPECT_EQ(definition->getInputsInfo().count("in2"), 1);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

TEST_F(MediapipeConfigChanges, ConfigWithEmptyBasePath) {
    std::string graphPbtxtFileContent = pbtxtContent;
    std::string configFileContent = configFileWithEmptyBasePath;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphName = "mediapipeGraph";
    std::string graphFilePath = directoryPath + "/" + graphName + "/graph.pbtxt";

    const std::string inputName{"in\""};
    const std::string newInputName{"in2\""};

    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    std::string defaultGraphDirectoryPath = directoryPath + "/" + graphName;
    std::filesystem::create_directories(defaultGraphDirectoryPath);
    createConfigFileWithContent(graphPbtxtFileContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    auto model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    EXPECT_EQ(definition->getInputsInfo().count("in"), 1);
    EXPECT_EQ(definition->getInputsInfo().count("in2"), 0);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

class MediapipeSerialization : public ::testing::Test {
    class MockedMediapipeGraphExecutor : public ovms::MediapipeGraphExecutor {
    public:
        MockedMediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
            stream_types_mapping_t inputTypes,
            stream_types_mapping_t outputTypes,
            std::vector<std::string> inputNames, std::vector<std::string> outputNames,
            const std::shared_ptr<PythonNodeResourcesMap>& pythonNodeResourcesMap,
            const std::shared_ptr<GenAiServableMap>& gasm,
            MediapipeServableMetricReporter* mediapipeServableMetricReporter, GraphIdGuard&& guard) :
            MediapipeGraphExecutor(name, version, config, inputTypes, outputTypes, inputNames, outputNames, pythonNodeResourcesMap, gasm, nullptr, mediapipeServableMetricReporter, std::move(guard)) {}
    };

protected:
    std::unique_ptr<MediapipeServableMetricReporter> reporter;
    std::shared_ptr<GraphQueue> queue;
    std::unique_ptr<MockedMediapipeGraphExecutor> executor;
    ::inference::ModelInferResponse mp_response;
    void SetUp() {
        ovms::stream_types_mapping_t mapping;
        mapping["kfs_response"] = mediapipe_packet_type_enum::KFS_RESPONSE;
        mapping["tf_response"] = mediapipe_packet_type_enum::TFTENSOR;
        mapping["ov_response"] = mediapipe_packet_type_enum::OVTENSOR;
        mapping["mp_response"] = mediapipe_packet_type_enum::MPTENSOR;
        mapping["mp_img_response"] = mediapipe_packet_type_enum::MEDIAPIPE_IMAGE;
        const std::vector<std::string> inputNames;
        const std::vector<std::string> outputNames;
        const ::mediapipe::CalculatorGraphConfig config;
        this->reporter = std::make_unique<MediapipeServableMetricReporter>(nullptr, nullptr, "");  // disabled reporter
        std::shared_ptr<GenAiServableMap> gasm = std::make_shared<GenAiServableMap>();
        std::shared_ptr<PythonNodeResourcesMap> pnsm = std::make_shared<PythonNodeResourcesMap>();
        std::shared_ptr<GraphQueue> queue = std::make_shared<GraphQueue>(config, pnsm, gasm, 1);
        GraphIdGuard guard(queue);
        executor = std::make_unique<MockedMediapipeGraphExecutor>("", "", config, mapping, mapping, inputNames, outputNames, pnsm, gasm, this->reporter.get(), std::move(guard));
        SPDLOG_ERROR("Exit SetUp");
    }
};

TEST_F(MediapipeSerialization, KFSResponse) {
    KFSResponse response;
    response.set_id("1");
    auto output = response.add_outputs();
    output->add_shape(1);
    output->set_datatype("FP32");
    std::vector<float> data = {1.0f};
    response.add_raw_output_contents()->assign(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    ::mediapipe::Packet packet = ::mediapipe::MakePacket<KFSResponse>(response);
    ASSERT_EQ(onPacketReadySerializeImpl("1", "name", "1", "name", mediapipe_packet_type_enum::KFS_RESPONSE, packet, mp_response), StatusCode::OK);
    ASSERT_EQ(mp_response.id(), "1");
    ASSERT_EQ(mp_response.outputs_size(), 1);
    auto mp_output = mp_response.outputs(0);
    ASSERT_EQ(mp_output.datatype(), "FP32");
    ASSERT_EQ(mp_output.shape_size(), 1);
    ASSERT_EQ(mp_output.shape(0), 1);
    ASSERT_EQ(mp_response.raw_output_contents_size(), 1);
    ASSERT_EQ(mp_response.raw_output_contents().at(0).size(), 4);
    ASSERT_EQ(reinterpret_cast<const float*>(mp_response.raw_output_contents().at(0).data())[0], 1.0f);
}

TEST_F(MediapipeSerialization, TFTensor) {
    tensorflow::Tensor response(TFSDataType::DT_FLOAT, {1});
    response.flat<float>()(0) = 1.0f;
    ::mediapipe::Packet packet = ::mediapipe::MakePacket<tensorflow::Tensor>(response);
    ASSERT_EQ(onPacketReadySerializeImpl("1", "tf_response", "1", "tf_response", mediapipe_packet_type_enum::TFTENSOR, packet, mp_response), StatusCode::OK);
    ASSERT_EQ(mp_response.id(), "1");
    ASSERT_EQ(mp_response.outputs(0).datatype(), "FP32");
    ASSERT_EQ(mp_response.outputs_size(), 1);
    auto mp_output = mp_response.outputs(0);
    ASSERT_EQ(mp_output.shape_size(), 1);
    ASSERT_EQ(mp_output.shape(0), 1);
    ASSERT_EQ(mp_response.raw_output_contents_size(), 1);
    ASSERT_EQ(mp_response.raw_output_contents().at(0).size(), 4);
    ASSERT_EQ(reinterpret_cast<const float*>(mp_response.raw_output_contents().at(0).data())[0], 1.0f);
}

TEST_F(MediapipeSerialization, OVTensor) {
    std::vector<float> data = {1.0f};
    ov::element::Type type(ov::element::Type_t::f32);
    ov::Tensor response(type, {1}, data.data());
    ::mediapipe::Packet packet = ::mediapipe::MakePacket<ov::Tensor>(response);
    ASSERT_EQ(onPacketReadySerializeImpl("1", "ov_response", "1", "ov_response", mediapipe_packet_type_enum::OVTENSOR, packet, mp_response), StatusCode::OK);
    ASSERT_EQ(mp_response.id(), "1");
    ASSERT_EQ(mp_response.outputs(0).datatype(), "FP32");
    ASSERT_EQ(mp_response.outputs_size(), 1);
    auto mp_output = mp_response.outputs(0);
    ASSERT_EQ(mp_output.shape_size(), 1);
    ASSERT_EQ(mp_output.shape(0), 1);
    ASSERT_EQ(mp_response.raw_output_contents_size(), 1);
    ASSERT_EQ(mp_response.raw_output_contents().at(0).size(), 4);
    ASSERT_EQ(reinterpret_cast<const float*>(mp_response.raw_output_contents().at(0).data())[0], 1.0f);
}

TEST_F(MediapipeSerialization, MPTensor) {
    mediapipe::Tensor response(mediapipe::Tensor::ElementType::kFloat32, {1});
    response.GetCpuWriteView().buffer<float>()[0] = 1.0f;
    ::mediapipe::Packet packet = ::mediapipe::MakePacket<mediapipe::Tensor>(std::move(response));
    ASSERT_EQ(onPacketReadySerializeImpl("1", "mp_response", "1", "mp_response", mediapipe_packet_type_enum::MPTENSOR, packet, mp_response), StatusCode::OK);
    ASSERT_EQ(mp_response.id(), "1");
    ASSERT_EQ(mp_response.outputs(0).datatype(), "FP32");
    ASSERT_EQ(mp_response.outputs_size(), 1);
    auto mp_output = mp_response.outputs(0);
    ASSERT_EQ(mp_output.shape_size(), 1);
    ASSERT_EQ(mp_output.shape(0), 1);
    ASSERT_EQ(mp_response.raw_output_contents_size(), 1);
    ASSERT_EQ(mp_response.raw_output_contents().at(0).size(), 4);
    ASSERT_EQ(reinterpret_cast<const float*>(mp_response.raw_output_contents().at(0).data())[0], 1.0f);
}

TEST_F(MediapipeSerialization, MPImageTensor) {
    mediapipe::ImageFrame response(static_cast<mediapipe::ImageFormat::Format>(1), 1, 1);
    response.MutablePixelData()[0] = (char)1;
    response.MutablePixelData()[1] = (char)1;
    response.MutablePixelData()[2] = (char)1;
    ::mediapipe::Packet packet = ::mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(response));
    ASSERT_EQ(onPacketReadySerializeImpl("1", "mp_img_response", "1", "mp_img_response", mediapipe_packet_type_enum::MEDIAPIPE_IMAGE, packet, mp_response), StatusCode::OK);
    ASSERT_EQ(mp_response.id(), "1");
    ASSERT_EQ(mp_response.outputs(0).datatype(), "UINT8");
    ASSERT_EQ(mp_response.outputs_size(), 1);
    auto mp_output = mp_response.outputs(0);
    ASSERT_EQ(mp_output.shape_size(), 3);
    ASSERT_EQ(mp_output.shape(0), 1);
    ASSERT_EQ(mp_response.raw_output_contents_size(), 1);
    ASSERT_EQ(mp_response.raw_output_contents().at(0).size(), 3);
    EXPECT_EQ(mp_response.raw_output_contents().at(0).data()[0], 1);
    EXPECT_EQ(mp_response.raw_output_contents().at(0).data()[1], 1);
    EXPECT_EQ(mp_response.raw_output_contents().at(0).data()[2], 1);
}

TEST_F(MediapipeConfigChanges, ConfigWithNoBasePath) {
    std::string graphPbtxtFileContent = pbtxtContent;
    std::string configFileContent = configFileWithNoBasePath;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphName = "mediapipeGraph";
    std::string graphFilePath = directoryPath + "/" + graphName + "/graph.pbtxt";

    const std::string inputName{"in\""};
    const std::string newInputName{"in2\""};

    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    std::string defaultGraphDirectoryPath = directoryPath + "/" + graphName;
    std::filesystem::create_directories(defaultGraphDirectoryPath);
    createConfigFileWithContent(graphPbtxtFileContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    auto model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    EXPECT_EQ(definition->getInputsInfo().count("in"), 1);
    EXPECT_EQ(definition->getInputsInfo().count("in2"), 0);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

TEST_F(MediapipeConfigChanges, AddProperGraphThenRetireThenAddAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we retire
    configFileContent = configFileWithoutGraph;
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::RETIRED);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE);
    // now we add again
    configFileContent = configFileWithGraphPathToReplace;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

TEST_F(MediapipeConfigChanges, AddImproperGraphThenFixWithReloadThenBreakAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    ovms::Status status;
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
    // TODO check for tfs as well - now not supported
    // checkStatus<TFSPredictRequest, TFSPredictResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
    // now we fix the config
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we break
    configFileContent = configFileWithGraphPathToReplace;
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(MediapipeConfigChanges, GraphWithNonexistentCalcShouldBeInNotLoadedYet) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/subconfig.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContentNonexistentCalc, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    ovms::Status status;
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
    // TODO check for tfs as well - now not supported
    // checkStatus<TFSPredictRequest, TFSPredictResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(MediapipeConfigChanges, AddModelToConfigThenUnloadThenAddToSubconfig) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we retire the model
    configFileContent = configFileWithGraphPathToReplaceWithoutModel;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    model = modelManager.findModelByName("dummy");
    ASSERT_EQ(nullptr, model->getDefaultModelInstance());
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we add model to subconfig
    std::string subconfigFilePath = directoryPath + "/subconfig.json";
    SPDLOG_ERROR("{}", subconfigFilePath);
    configFileContent = configFileWithoutGraph;
    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, subconfigFilePath);
    configFileContent = configFileWithGraphPathToReplaceAndSubconfig;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    const std::string subconfigPathToReplace{"SUBCONFIG_PATH"};
    configFileContent.replace(configFileContent.find(subconfigPathToReplace), subconfigPathToReplace.size(), subconfigFilePath);
    adjustConfigForTargetPlatform(configFileContent);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

TEST(MediapipeStreamTypes, Recognition) {
    using ovms::mediapipe_packet_type_enum;
    using ovms::MediapipeGraphDefinition;
    using streamNameTypePair_t = std::pair<std::string, mediapipe_packet_type_enum>;
    // basic tag name matching
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), ovms::getStreamNamePair("TENSOR:out", MediaPipeStreamType::OUTPUT));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::TFTENSOR), ovms::getStreamNamePair("TFTENSOR:out", MediaPipeStreamType::OUTPUT));
    EXPECT_EQ(streamNameTypePair_t("input", mediapipe_packet_type_enum::OVTENSOR), ovms::getStreamNamePair("OVTENSOR:input", MediaPipeStreamType::INPUT));
    EXPECT_EQ(streamNameTypePair_t("input", mediapipe_packet_type_enum::KFS_REQUEST), ovms::getStreamNamePair("REQUEST:input", MediaPipeStreamType::INPUT));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::KFS_RESPONSE), ovms::getStreamNamePair("RESPONSE:out", MediaPipeStreamType::OUTPUT));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MEDIAPIPE_IMAGE), ovms::getStreamNamePair("IMAGE:out", MediaPipeStreamType::OUTPUT));
    // string after suffix doesn't matter
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), ovms::getStreamNamePair("TENSOR1:out", MediaPipeStreamType::OUTPUT));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), ovms::getStreamNamePair("TENSOR_1:out", MediaPipeStreamType::OUTPUT));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::KFS_RESPONSE), ovms::getStreamNamePair("RESPONSE_COSTAM:out", MediaPipeStreamType::OUTPUT));
    // number as additional part doesn't affect recognized type
    EXPECT_EQ(streamNameTypePair_t("in", mediapipe_packet_type_enum::MPTENSOR), ovms::getStreamNamePair("TENSOR:1:in", MediaPipeStreamType::INPUT));
    // negative
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::UNKNOWN), ovms::getStreamNamePair("TENSO:out", MediaPipeStreamType::OUTPUT));             // negative - non-matching tag
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::UNKNOWN), ovms::getStreamNamePair("SOME_STRANGE_TAG:out", MediaPipeStreamType::OUTPUT));  // negative - non-matching tag
    EXPECT_EQ(streamNameTypePair_t("in", mediapipe_packet_type_enum::UNKNOWN), ovms::getStreamNamePair("in", MediaPipeStreamType::INPUT));
}

// TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfigNegative) {
//     ConstructorEnabledModelManager manager;
//     auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_add_subconfig_negative.json");
//     EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
//     manager.join();
// }
//

std::promise<void> unblockLoading2ndGraph;
namespace mediapipe {
class LongLoadingCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        auto signal = unblockLoading2ndGraph.get_future();
        signal.get();
        for (const std::string& tag : cc->Inputs().GetTags()) {
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        }
        return absl::OkStatus();
    }
    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Process(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
};
REGISTER_CALCULATOR(LongLoadingCalculator);
}  // namespace mediapipe

static void stopServer() {
    OVMS_Server* cserver;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ovms::Server& server = ovms::Server::instance();
    server.setShutdownRequest(1);
}
static bool isMpReady(const std::string name) {
    ovms::Server& server = ovms::Server::instance();
    SPDLOG_TRACE("serverReady:{}", server.isReady());
    const ovms::Module* servableModule = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    if (!servableModule) {
        return false;
    }
    ModelManager* manager = &dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();
    auto mediapipeGraphDefinition = manager->getMediapipeFactory().findDefinitionByName(name);
    if (!mediapipeGraphDefinition) {
        return false;
    }
    return mediapipeGraphDefinition->getStatus().isAvailable();
}

class MediapipeFlowStartTest : public TestWithTempDir {
protected:
    bool isMpReady(const std::string name) {
        return ::isMpReady(name);
    }
    void stopServer() {
        ::stopServer();
    }
    void TearDown() {
        OVMS_Server* cserver = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        bool serverLive = false;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerLive(cserver, &serverLive));
        if (serverLive) {
            stopServer();
        }
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(0);
        std::promise<void>().swap(unblockLoading2ndGraph);
    }

    // 1st thread starts to load OVMS with C-API but we make it stuck on 2nd graph
    // 2nd thread as soon as sees that 1st MP graph is ready executest inference
    void executeFlow(std::string& configContent, const std::string& waitForServable = "mediapipeDummy") {
        std::string configFilePath = directoryPath + "/config.json";
        adjustConfigForTargetPlatform(configContent);
        createConfigFileWithContent(configContent, configFilePath);
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(0);
        std::string port{"9000"};
        randomizePort(port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configFilePath.c_str(),
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 5;
        std::thread t([&argc, &argv, &server]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        });

        ::KFSRequest request;
        ::KFSResponse response;
        const std::string servableName{"mediapipeDummy"};
        request.Clear();
        response.Clear();
        const Precision precision = Precision::FP32;
        inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
        std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
        preparePredictRequest(request, inputsMeta, requestData);
        request.mutable_model_name()->assign(servableName);

        auto start = std::chrono::high_resolution_clock::now();
        while (!isMpReady(waitForServable) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS)) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
        const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
        if (!grpcModule) {
            this->stopServer();
            t.join();
            throw 42;
        }
        KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
        ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
        try {
            unblockLoading2ndGraph.set_value();
        } catch (const std::future_error& e) {
            // Case where we already set the value before execute.
            ASSERT_EQ(e.code(), std::future_errc::promise_already_satisfied);
        }

        size_t dummysInTheGraph = 1;
        checkDummyResponse("out", requestData, request, response, dummysInTheGraph, 1, servableName);
        this->stopServer();
        t.join();
    }
};

TEST_F(MediapipeFlowStartTest, AsSoonAsMediaPipeGraphDefinitionReadyInferShouldPass) {
    std::string configContent = R"(
{
    "model_config_list": [
        {"config": {
            "name": "dummy",
            "base_path": "/ovms/src/test/dummy"
            }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeDummy",
        "graph_path": "/ovms/src/test/mediapipe/graphdummyadapterfull.pbtxt"
    },
    {
        "name": "mediapipeLongLoading",
        "graph_path": "/ovms/src/test/mediapipe/negative/graph_long_loading.pbtxt"
    }
    ]
}
)";

    executeFlow(configContent);
}

TEST_F(MediapipeFlowStartTest, AsSoonAsMediaPipeGraphDefinitionReadyInferShouldPassGraphInModelConfig) {
    std::string configContent = R"(
{
    "model_config_list": [
        {"config": {
            "name": "dummy",
            "base_path": "/ovms/src/test/dummy"
            }
        },
        {"config": {
            "name":"mediapipeDummy",
            "base_path":"/ovms/src/test/mediapipe/",
            "graph_path": "graphdummyadapterfull.pbtxt"
            }
        },
        {"config": {
            "name": "mediapipeLongLoading",
            "base_path":"/ovms/src/test/mediapipe/negative",
            "graph_path": "graph_long_loading.pbtxt"
            }
        }
    ]
}
)";

    executeFlow(configContent);
}

TEST_F(MediapipeFlowStartTest, AsSoonAsMediaPipeGraphDefinitionReadyInferShouldPassGraphInModelConfigFastLoading) {
    std::string configContent = R"(
{
    "model_config_list": [
        {"config": {
            "name": "dummy",
            "base_path": "/ovms/src/test/dummy"
            }
        },
        {"config": {
            "name":"mediapipeDummy",
            "base_path":"/ovms/src/test/mediapipe/",
            "graph_path": "graphdummyadapterfull.pbtxt"
            }
        }
    ],
    "mediapipe_config_list": [
    {
        "name": "mediapipeLongLoading",
        "base_path":"/ovms/src/test/mediapipe/negative",
        "graph_path": "graph_long_loading.pbtxt"
    }
    ]
}
)";
    // Set value here to avoid deadlock when long loading is loaded first
    unblockLoading2ndGraph.set_value();

    executeFlow(configContent);
}

TEST_F(MediapipeFlowStartTest, AsSoonAsMediaPipeGraphDefinitionReadyInferShouldPassGraphInModelConfigLongLoading) {
    std::string configContent = R"(
{
    "model_config_list": [
        {"config": {
            "name": "dummy",
            "base_path": "/ovms/src/test/dummy"
            }
        },
        {"config": {
            "name": "mediapipeLongLoading",
            "base_path":"/ovms/src/test/mediapipe/negative",
            "graph_path": "graph_long_loading.pbtxt"
            }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeDummy",
        "base_path":"/ovms/src/test/mediapipe/",
        "graph_path": "graphdummyadapterfull.pbtxt"
    }
    ]
}
)";

    executeFlow(configContent);
}

std::unordered_map<std::type_index, ovms::Precision> TYPE_TO_OVMS_PRECISION{
    {typeid(float), ovms::Precision::FP32},
    {typeid(uint64_t), ovms::Precision::U64},
    {typeid(uint32_t), ovms::Precision::U32},
    {typeid(uint16_t), ovms::Precision::U16},
    {typeid(uint8_t), ovms::Precision::U8},
    {typeid(int64_t), ovms::Precision::I64},
    {typeid(int32_t), ovms::Precision::I32},
    {typeid(int16_t), ovms::Precision::I16},
    {typeid(int8_t), ovms::Precision::I8},
    {typeid(bool), ovms::Precision::BOOL},
    {typeid(double), ovms::Precision::FP64},
    {typeid(void), ovms::Precision::BIN}};

template <typename T>
std::vector<T> prepareData(size_t elemCount, T value = std::numeric_limits<T>::max()) {
    return std::vector<T>(elemCount, value);
}

template <class T>
class KFSGRPCContentFieldsSupportTest : public TestWithTempDir {
protected:
    std::string configFilePath = "config.json";
    std::string configContent = R"(
{
    "model_config_list": [
        {"config": {
            "name": "dummy",
            "base_path": "/ovms/src/test/dummy"
            }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeDummy",
        "graph_path": "XYZ"
    }
    ]
}
)";
    const std::string modelPathToReplace{"XYZ"};
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;
    std::string port = "9000";
    const std::string servableName{"mediapipeDummy"};
    bool maxValue = false;
    bool putDataInInputContents = true;
    size_t elemCount = 10;
    KFSRequest request;
    KFSResponse response;

    void CreateConfigAndPbtxt(std::string pbtxtContent) {
        std::string graphFilePath = this->directoryPath + "/graph.pbtxt";
        this->configContent.replace(this->configContent.find(this->modelPathToReplace), this->modelPathToReplace.size(), graphFilePath);
        this->configFilePath = this->directoryPath + this->configFilePath;
        createConfigFileWithContent(this->configContent, this->configFilePath);
        createConfigFileWithContent(pbtxtContent, graphFilePath);
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        randomizePort(port);
        request.Clear();
        request.mutable_model_name()->assign(servableName);
    }

    void TearDown() override {
        TestWithTempDir::TearDown();
        stopServer();
        t->join();
    }

    void performInference(ovms::StatusCode expectedStatus) {
        response.Clear();
        auto start = std::chrono::high_resolution_clock::now();
        while (!isMpReady(servableName) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        const ovms::Module* grpcModule = this->server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
        if (!grpcModule) {
            throw 42;
        }
        const ServableManagerModule* smm = dynamic_cast<const ServableManagerModule*>(this->server.getModule(SERVABLE_MANAGER_MODULE_NAME));
        ModelManager& modelManager = smm->getServableManager();
        std::shared_ptr<MediapipeGraphExecutor> executor;
        ASSERT_EQ(modelManager.createPipeline(executor, this->request.model_name()), ovms::StatusCode::OK);
        using ovms::ExecutionContext;
        ExecutionContext executionContext{ExecutionContext::Interface::GRPC, ExecutionContext::Method::ModelInfer};
        auto status = executor->infer(&this->request, &response, executionContext);
        EXPECT_EQ(status, expectedStatus) << status.string();
        if (expectedStatus == ovms::StatusCode::OK) {
            ASSERT_EQ(response.outputs_size(), 1);
            ASSERT_EQ(response.raw_output_contents_size(), 1);
            ASSERT_EQ(response.raw_output_contents()[0].size(), 10 * ovms::KFSDataTypeSize(request.inputs()[0].datatype()));
        }
    }

    void performInvalidContentSizeTest(const std::string& pbtxtContentOVTensor, ovms::StatusCode expectedStatus) {
        this->CreateConfigAndPbtxt(pbtxtContentOVTensor);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)this->configFilePath.c_str(),
            (char*)"--port",
            (char*)this->port.c_str()};
        int argc = 5;
        this->server.setShutdownRequest(0);
        this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
        });
        // prepare data
        T value = 1.0;
        std::vector<T> data = prepareData<T>(this->elemCount, value);
        preparePredictRequest(this->request,
            {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION[typeid(T)]}}},
            data, this->putDataInInputContents);
        auto tensor = this->request.mutable_inputs()->begin();
        switch (TYPE_TO_OVMS_PRECISION[typeid(T)]) {
        case ovms::Precision::FP64: {
            auto ptr = tensor->mutable_contents()->mutable_fp64_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::FP32: {
            auto ptr = tensor->mutable_contents()->mutable_fp32_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::U64: {
            auto ptr = tensor->mutable_contents()->mutable_uint64_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::U8:
        case ovms::Precision::U16:
        case ovms::Precision::U32: {
            auto ptr = tensor->mutable_contents()->mutable_uint_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::I64: {
            auto ptr = tensor->mutable_contents()->mutable_int64_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::BOOL: {
            auto ptr = tensor->mutable_contents()->mutable_bool_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::I8:
        case ovms::Precision::I16:
        case ovms::Precision::I32: {
            auto ptr = tensor->mutable_contents()->mutable_int_contents()->Add();
            *ptr = 0;
            break;
        }
        case ovms::Precision::FP16:
        case ovms::Precision::U1:
        case ovms::Precision::CUSTOM:
        case ovms::Precision::UNDEFINED:
        case ovms::Precision::DYNAMIC:
        case ovms::Precision::MIXED:
        case ovms::Precision::Q78:
        case ovms::Precision::BIN:
        default: {
        }
        }
        const std::string servableName{"mediapipeDummy"};
        this->request.mutable_model_name()->assign(servableName);
        this->performInference(expectedStatus);
    }
};

std::unordered_map<std::type_index, std::pair<ovms::Precision, ovms::StatusCode>> TYPE_TO_OVMS_PRECISION_TO_STATUS_OV_TENSOR{
    {typeid(float), {ovms::Precision::FP32, ovms::StatusCode::OK}},
    {typeid(uint64_t), {ovms::Precision::U64, ovms::StatusCode::OK}},
    {typeid(uint32_t), {ovms::Precision::U32, ovms::StatusCode::OK}},
    {typeid(uint16_t), {ovms::Precision::U16, ovms::StatusCode::OK}},
    {typeid(uint8_t), {ovms::Precision::U8, ovms::StatusCode::OK}},
    {typeid(int64_t), {ovms::Precision::I64, ovms::StatusCode::OK}},
    {typeid(int32_t), {ovms::Precision::I32, ovms::StatusCode::OK}},
    {typeid(int16_t), {ovms::Precision::I16, ovms::StatusCode::OK}},
    {typeid(int8_t), {ovms::Precision::I8, ovms::StatusCode::OK}},
    {typeid(bool), {ovms::Precision::BOOL, ovms::StatusCode::OK}},
    {typeid(double), {ovms::Precision::FP64, ovms::StatusCode::OK}},
    {typeid(void), {ovms::Precision::BIN, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR}}};

typedef testing::Types<float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t, uint8_t, bool> InferInputTensorContentsTypesToTest;
TYPED_TEST_SUITE(KFSGRPCContentFieldsSupportTest, InferInputTensorContentsTypesToTest);
TYPED_TEST(KFSGRPCContentFieldsSupportTest, OVTensorCheckExpectedStatusCode) {
    const std::string pbtxtContentOVTensor = R"(
        input_stream: "OVTENSOR:in"
        output_stream: "OVTENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "OVTENSOR:in"
        output_stream: "OVTENSOR:out"
        }
    )";
    this->CreateConfigAndPbtxt(pbtxtContentOVTensor);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)this->configFilePath.c_str(),
        (char*)"--port",
        (char*)this->port.c_str()};
    int argc = 5;
    this->server.setShutdownRequest(0);
    this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
        EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
    });
    // prepare data
    std::vector<TypeParam> data = prepareData<TypeParam>(this->elemCount);
    preparePredictRequest(this->request,
        {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION_TO_STATUS_OV_TENSOR[typeid(TypeParam)].first}}},
        data, this->putDataInInputContents);
    const std::string servableName{"mediapipeDummy"};
    this->request.mutable_model_name()->assign(servableName);
    this->performInference(TYPE_TO_OVMS_PRECISION_TO_STATUS_OV_TENSOR[typeid(TypeParam)].second);
}

#if (PYTHON_DISABLE == 0)
TYPED_TEST(KFSGRPCContentFieldsSupportTest, PyTensorCheckExpectedStatusCode) {
    const std::string pbtxtContentPytensor = R"(
        input_stream: "OVMS_PY_TENSOR:in"
        output_stream: "OVMS_PY_TENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "OVMS_PY_TENSOR:in"
        output_stream: "OVMS_PY_TENSOR:out"
        }
    )";
    this->CreateConfigAndPbtxt(pbtxtContentPytensor);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)this->configFilePath.c_str(),
        (char*)"--port",
        (char*)this->port.c_str()};
    int argc = 5;
    this->server.setShutdownRequest(0);
    this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
        EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
    });
    // prepare data
    std::vector<TypeParam> data = prepareData<TypeParam>(this->elemCount);
    preparePredictRequest(this->request,
        {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION_TO_STATUS_OV_TENSOR[typeid(TypeParam)].first}}},
        data, this->putDataInInputContents);
    const std::string servableName{"mediapipeDummy"};
    this->request.mutable_model_name()->assign(servableName);
    this->performInference(TYPE_TO_OVMS_PRECISION_TO_STATUS_OV_TENSOR[typeid(TypeParam)].second);
}

TYPED_TEST(KFSGRPCContentFieldsSupportTest, PyTensorInvalidContentSize) {
    const std::string pbtxtContentPyTensor = R"(
        input_stream: "OVMS_PY_TENSOR:in"
        output_stream: "OVMS_PY_TENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "OVMS_PY_TENSOR:in"
        output_stream: "OVMS_PY_TENSOR:out"
        }
    )";
    this->performInvalidContentSizeTest(pbtxtContentPyTensor, ovms::StatusCode::INVALID_VALUE_COUNT);
}
#endif

std::unordered_map<std::type_index, std::pair<ovms::Precision, ovms::StatusCode>> TYPE_TO_OVMS_PRECISION_TO_STATUS_TF_TENSOR{
    {typeid(float), {ovms::Precision::FP32, ovms::StatusCode::OK}},
    {typeid(uint64_t), {ovms::Precision::U64, ovms::StatusCode::OK}},
    {typeid(uint32_t), {ovms::Precision::U32, ovms::StatusCode::OK}},
    {typeid(uint16_t), {ovms::Precision::U16, ovms::StatusCode::OK}},
    {typeid(uint8_t), {ovms::Precision::U8, ovms::StatusCode::OK}},
    {typeid(int64_t), {ovms::Precision::I64, ovms::StatusCode::OK}},
    {typeid(int32_t), {ovms::Precision::I32, ovms::StatusCode::OK}},
    {typeid(int16_t), {ovms::Precision::I16, ovms::StatusCode::OK}},
    {typeid(int8_t), {ovms::Precision::I8, ovms::StatusCode::OK}},
    {typeid(bool), {ovms::Precision::BOOL, ovms::StatusCode::OK}},
    {typeid(double), {ovms::Precision::FP64, ovms::StatusCode::OK}},
    {typeid(void), {ovms::Precision::BIN, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR}}};

TYPED_TEST(KFSGRPCContentFieldsSupportTest, TFTensorCheckExpectedStatusCode) {
    const std::string pbtxtContentTFtensor = R"(
        input_stream: "TFTENSOR:in"
        output_stream: "TFTENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "TFTENSOR:in"
        output_stream: "TFTENSOR:out"
        }
    )";
    this->CreateConfigAndPbtxt(pbtxtContentTFtensor);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)this->configFilePath.c_str(),
        (char*)"--port",
        (char*)this->port.c_str()};
    int argc = 5;
    this->server.setShutdownRequest(0);
    this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
        EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
    });
    // prepare data
    std::vector<TypeParam> data = prepareData<TypeParam>(this->elemCount);
    preparePredictRequest(this->request,
        {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION_TO_STATUS_TF_TENSOR[typeid(TypeParam)].first}}},
        data, this->putDataInInputContents);
    const std::string servableName{"mediapipeDummy"};
    this->request.mutable_model_name()->assign(servableName);
    this->performInference(TYPE_TO_OVMS_PRECISION_TO_STATUS_TF_TENSOR[typeid(TypeParam)].second);
}

std::unordered_map<std::type_index, std::pair<ovms::Precision, ovms::StatusCode>> TYPE_TO_OVMS_PRECISION_TO_STATUS_MP_TENSOR{
    {typeid(float), {ovms::Precision::FP32, ovms::StatusCode::OK}},
    {typeid(uint64_t), {ovms::Precision::U64, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(uint32_t), {ovms::Precision::U32, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(uint16_t), {ovms::Precision::U16, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(uint8_t), {ovms::Precision::U8, ovms::StatusCode::OK}},
    {typeid(int64_t), {ovms::Precision::I64, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(int32_t), {ovms::Precision::I32, ovms::StatusCode::OK}},
    {typeid(int16_t), {ovms::Precision::I16, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(int8_t), {ovms::Precision::I8, ovms::StatusCode::OK}},
    {typeid(bool), {ovms::Precision::BOOL, ovms::StatusCode::OK}},
    {typeid(double), {ovms::Precision::FP64, ovms::StatusCode::INVALID_PRECISION}},
    {typeid(void), {ovms::Precision::BIN, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR}}};

TYPED_TEST(KFSGRPCContentFieldsSupportTest, MPTensorCheckExpectedStatusCode) {
    const std::string pbtxtContentMPtensor = R"(
        input_stream: "TENSOR:in"
        output_stream: "TENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "TENSOR:in"
        output_stream: "TENSOR:out"
        }
    )";
    this->CreateConfigAndPbtxt(pbtxtContentMPtensor);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)this->configFilePath.c_str(),
        (char*)"--port",
        (char*)this->port.c_str()};
    int argc = 5;
    this->server.setShutdownRequest(0);
    this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
        EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
    });
    // prepare data
    std::vector<TypeParam> data = prepareData<TypeParam>(this->elemCount);
    preparePredictRequest(this->request,
        {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION_TO_STATUS_MP_TENSOR[typeid(TypeParam)].first}}},
        data, this->putDataInInputContents);
    const std::string servableName{"mediapipeDummy"};
    this->request.mutable_model_name()->assign(servableName);
    this->performInference(TYPE_TO_OVMS_PRECISION_TO_STATUS_MP_TENSOR[typeid(TypeParam)].second);
}

TYPED_TEST(KFSGRPCContentFieldsSupportTest, IMAGETensorCheckExpectedStatusCode) {
    const std::string pbtxtContentIMAGEtensor = R"(
        input_stream: "IMAGE:in"
        output_stream: "IMAGE:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "IMAGE:in"
        output_stream: "IMAGE:out"
        }
    )";
    this->CreateConfigAndPbtxt(pbtxtContentIMAGEtensor);
    char* argv[] = {(char*)"ovms",
        (char*)"--config_path",
        (char*)this->configFilePath.c_str(),
        (char*)"--port",
        (char*)this->port.c_str()};
    int argc = 5;
    this->server.setShutdownRequest(0);
    this->t = std::make_unique<std::thread>([&argc, &argv, this]() {
        EXPECT_EQ(EXIT_SUCCESS, this->server.start(argc, argv));
    });
    // prepare data
    std::vector<TypeParam> data = prepareData<TypeParam>(this->elemCount);
    preparePredictRequest(this->request,
        {{"in", {{1, 10}, TYPE_TO_OVMS_PRECISION[typeid(TypeParam)]}}},
        data, this->putDataInInputContents);
    const std::string servableName{"mediapipeDummy"};
    this->request.mutable_model_name()->assign(servableName);
    this->performInference(ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TYPED_TEST(KFSGRPCContentFieldsSupportTest, OVTensorInvalidContentSize) {
    const std::string pbtxtContentOVTensor = R"(
        input_stream: "OVTENSOR:in"
        output_stream: "OVTENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "OVTENSOR:in"
        output_stream: "OVTENSOR:out"
        }
    )";
    this->performInvalidContentSizeTest(pbtxtContentOVTensor, ovms::StatusCode::INVALID_VALUE_COUNT);
}

std::unordered_map<std::type_index, ovms::StatusCode> TYPE_TO_STATUS_MP_TENSOR_INVALID_CONTENT_SIZE{
    {typeid(float), ovms::StatusCode::INVALID_VALUE_COUNT},
    {typeid(uint64_t), ovms::StatusCode::INVALID_PRECISION},
    {typeid(uint32_t), ovms::StatusCode::INVALID_PRECISION},
    {typeid(uint16_t), ovms::StatusCode::INVALID_PRECISION},
    {typeid(uint8_t), ovms::StatusCode::INVALID_VALUE_COUNT},
    {typeid(int64_t), ovms::StatusCode::INVALID_PRECISION},
    {typeid(int32_t), ovms::StatusCode::INVALID_VALUE_COUNT},
    {typeid(int16_t), ovms::StatusCode::INVALID_PRECISION},
    {typeid(int8_t), ovms::StatusCode::INVALID_VALUE_COUNT},
    {typeid(bool), ovms::StatusCode::INVALID_VALUE_COUNT},
    {typeid(double), ovms::StatusCode::INVALID_PRECISION},
    {typeid(void), ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR}};

TYPED_TEST(KFSGRPCContentFieldsSupportTest, MPTensorInvalidContentSize) {
    const std::string pbtxtContentMPTensor = R"(
        input_stream: "TENSOR:in"
        output_stream: "TENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "TENSOR:in"
        output_stream: "TENSOR:out"
        }
    )";
    this->performInvalidContentSizeTest(pbtxtContentMPTensor, TYPE_TO_STATUS_MP_TENSOR_INVALID_CONTENT_SIZE[typeid(TypeParam)]);
}

TYPED_TEST(KFSGRPCContentFieldsSupportTest, TFTensorInvalidContentSize) {
    const std::string pbtxtContentTFTensor = R"(
        input_stream: "TFTENSOR:in"
        output_stream: "TFTENSOR:out"
        node {
        calculator: "PassThroughCalculator"
        input_stream: "TFTENSOR:in"
        output_stream: "TFTENSOR:out"
        }
    )";
    this->performInvalidContentSizeTest(pbtxtContentTFTensor, ovms::StatusCode::INVALID_VALUE_COUNT);
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowAddTest,
    ::testing::ValuesIn(mediaGraphsAdd),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });
INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowDummyTest,
    ::testing::ValuesIn(mediaGraphsDummy),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });

TEST(WhitelistRegistered, OutputStreamHandlers) {
    ASSERT_THAT(mediapipe::OutputStreamHandlerRegistry::GetRegisteredNames(), UnorderedElementsAre(
                                                                                  "InOrderOutputStreamHandler"));
}

TEST(WhitelistRegistered, InputStreamHandlers) {
    ASSERT_THAT(mediapipe::InputStreamHandlerRegistry::GetRegisteredNames(), UnorderedElementsAre(
                                                                                 "BarrierInputStreamHandler",
                                                                                 "DefaultInputStreamHandler",
                                                                                 "EarlyCloseInputStreamHandler",
                                                                                 "FixedSizeInputStreamHandler",
                                                                                 "ImmediateInputStreamHandler",
                                                                                 "MuxInputStreamHandler",
                                                                                 "SyncSetInputStreamHandler",
                                                                                 "TimestampAlignInputStreamHandler"));
}

TEST(WhitelistRegistered, MediapipeCalculatorsList) {
    std::unordered_set<std::string> expected({
#if (PYTHON_DISABLE == 0)
        // Expected when building with python
        "CalculatorRunnerSinkCalculator",
        "CalculatorRunnerSourceCalculator",
        "PyTensorOvTensorConverterCalculator",   // integral OVMS calculator
        "PythonExecutorCalculator",  // integral OVMS calculator
        "HttpLLMCalculator",  // integral OVMS calculator
#endif
        "OpenAIChatCompletionsMockCalculator",  // OVMS test calculator
        "AddHeaderCalculator",
        "AddNumbersMultiInputsOutputsTestCalculator",
        "AddOne3CycleIterationsTestCalculator",
        "AddOneSingleStreamTestCalculator",
        "AddSidePacketToSingleStreamTestCalculator",
        "AlignmentPointsRectsCalculator",
        "AnnotationOverlayCalculator",
        "AnomalyCalculator",
        "AnomalySerializationCalculator",
        "AssociationNormRectCalculator",
        "BeginLoopDetectionCalculator",
        "BeginLoopFloatCalculator",
        "BeginLoopGpuBufferCalculator",
        "BeginLoopImageCalculator",
        "BeginLoopImageFrameCalculator",
        "BeginLoopIntCalculator",
        "BeginLoopMatrixCalculator",
        "BeginLoopMatrixVectorCalculator",
        "BeginLoopModelApiDetectionCalculator",
        "BeginLoopNormalizedLandmarkListVectorCalculator",
        "BeginLoopNormalizedRectCalculator",
        "BeginLoopRectanglePredictionCalculator",
        "BeginLoopStringCalculator",
        "BeginLoopTensorCalculator",
        "BeginLoopUint64tCalculator",
        "BoxDetectorCalculator",
        "BoxTrackerCalculator",
        "CallbackCalculator",
        "CallbackPacketCalculator",
        "CallbackWithHeaderCalculator",
        "ClassificationCalculator",
        "ClassificationListVectorHasMinSizeCalculator",
        "ClassificationListVectorSizeCalculator",
        "ClassificationSerializationCalculator",
        "ClipDetectionVectorSizeCalculator",
        "ClipNormalizedRectVectorSizeCalculator",
        "ColorConvertCalculator",
        "ConcatenateBoolVectorCalculator",
        "ConcatenateClassificationListCalculator",
        "ConcatenateClassificationListVectorCalculator",
        "ConcatenateDetectionVectorCalculator",
        "ConcatenateFloatVectorCalculator",
        "ConcatenateImageVectorCalculator",
        "ConcatenateInt32VectorCalculator",
        "ConcatenateJointListCalculator",
        "ConcatenateLandmarListVectorCalculator",
        "ConcatenateLandmarkListCalculator",
        "ConcatenateLandmarkListVectorCalculator",
        "ConcatenateLandmarkVectorCalculator",
        "ConcatenateNormalizedLandmarkListCalculator",
        "ConcatenateNormalizedLandmarkListVectorCalculator",
        "ConcatenateRenderDataVectorCalculator",
        "ConcatenateStringVectorCalculator",
        "ConcatenateTensorVectorCalculator",
        "ConcatenateTfLiteTensorVectorCalculator",
        "ConcatenateUInt64VectorCalculator",
        "ConstantSidePacketCalculator",
        "CountingSourceCalculator",
        "CropCalculator",
        "DefaultSidePacketCalculator",
        "DequantizeByteArrayCalculator",
        "DetectionCalculator",
        "DetectionClassificationCombinerCalculator",
        "DetectionClassificationResultCalculator",
        "DetectionClassificationSerializationCalculator",
        "DetectionExtractionCalculator",
        "DetectionLabelIdToTextCalculator",
        "DetectionLetterboxRemovalCalculator",
        "DetectionProjectionCalculator",
        "DetectionSegmentationCombinerCalculator",
        "DetectionSegmentationResultCalculator",
        "DetectionSegmentationSerializationCalculator",
        "DetectionSerializationCalculator",
        "DetectionsToRectsCalculator",
        "DetectionsToRenderDataCalculator",
        "EmbeddingsCalculator",
        "RerankCalculator",
        "EmptyLabelCalculator",
        "EmptyLabelClassificationCalculator",
        "EmptyLabelDetectionCalculator",
        "EmptyLabelRotatedDetectionCalculator",
        "EmptyLabelSegmentationCalculator",
        "EndLoopAffineMatrixCalculator",
        "EndLoopBooleanCalculator",
        "EndLoopClassificationListCalculator",
        "EndLoopDetectionCalculator",
        "EndLoopFloatCalculator",
        "EndLoopGpuBufferCalculator",
        "EndLoopImageCalculator",
        "EndLoopImageFrameCalculator",
        "EndLoopImageSizeCalculator",
        "EndLoopLandmarkListVectorCalculator",
        "EndLoopMatrixCalculator",
        "EndLoopModelApiDetectionClassificationCalculator",
        "EndLoopModelApiDetectionSegmentationCalculator",
        "EndLoopNormalizedLandmarkListVectorCalculator",
        "EndLoopNormalizedRectCalculator",
        "EndLoopPolygonPredictionsCalculator",
        "EndLoopRectanglePredictionsCalculator",
        "EndLoopRenderDataCalculator",
        "EndLoopTensorCalculator",
        "EndLoopTfLiteTensorCalculator",
        "ErrorInProcessTestCalculator",
        "ExceptionDuringCloseCalculator",
        "ExceptionDuringGetContractCalculator",
        "ExceptionDuringOpenCalculator",
        "ExceptionDuringProcessCalculator",
        "FaceLandmarksToRenderDataCalculator",
        "FeatureDetectorCalculator",
        "FlowLimiterCalculator",
        "FlowPackagerCalculator",
        "FlowToImageCalculator",
        "FromImageCalculator",
        "GateCalculator",
        "GetClassificationListVectorItemCalculator",
        "GetDetectionVectorItemCalculator",
        "GetLandmarkListVectorItemCalculator",
        "GetNormalizedLandmarkListVectorItemCalculator",
        "GetNormalizedRectVectorItemCalculator",
        "GetRectVectorItemCalculator",
        "GraphProfileCalculator",
        "HandDetectionsFromPoseToRectsCalculator",
        "HandLandmarksToRectCalculator",
        "HttpSerializationCalculator",
        "ImageCloneCalculator",
        "ImageCroppingCalculator",
        "ImagePropertiesCalculator",
        "ImageToTensorCalculator",
        "ImageTransformationCalculator",
        "ImmediateMuxCalculator",
        "InferenceCalculatorCpu",
        "InputSidePacketUserTestCalc",
        "InstanceSegmentationCalculator",
        "InverseMatrixCalculator",
        "IrisToRenderDataCalculator",
        "KeypointDetectionCalculator",
        "LandmarkLetterboxRemovalCalculator",
        "LandmarkListVectorSizeCalculator",
        "LandmarkProjectionCalculator",
        "LandmarkVisibilityCalculator",
        "LandmarksRefinementCalculator",
        "LandmarksSmoothingCalculator",
        "LandmarksToDetectionCalculator",
        "LandmarksToRenderDataCalculator",
        "LongLoadingCalculator",
        "MakePairCalculator",
        "MatrixMultiplyCalculator",
        "MatrixSubtractCalculator",
        "MatrixToVectorCalculator",
        "MediaPipeInternalSidePacketToPacketStreamCalculator",
        "MergeCalculator",
        "MergeDetectionsToVectorCalculator",
        "MergeGpuBuffersToVectorCalculator",
        "MergeImagesToVectorCalculator",
        "ModelInferHttpRequestCalculator",
        "ModelInferRequestImageCalculator",
        "MotionAnalysisCalculator",
        "MuxCalculator",
        "NegativeCalculator",
        "NoOutputStreamsProducedCalculator",
        "NonMaxSuppressionCalculator",
        "NonZeroCalculator",
        "NormalizedLandmarkListVectorHasMinSizeCalculator",
        "NormalizedRectVectorHasMinSizeCalculator",
        "OverlayCalculator",
        "OVMSOVCalculator",
        "OVMSTestImageInputPassthroughCalculator",
        "OVMSTestKFSPassCalculator",
        "OpenCvEncodedImageToImageFrameCalculator",
        "OpenCvImageEncoderCalculator",
        "OpenCvPutTextCalculator",
        "OpenCvVideoDecoderCalculator",
        "OpenCvVideoEncoderCalculator",
        "OpenVINOConverterCalculator",
        "OpenVINOInferenceAdapterCalculator",
        "OpenVINOInferenceCalculator",
        "OpenVINOModelServerSessionCalculator",
        "OpenVINOTensorsToClassificationCalculator",
        "OpenVINOTensorsToDetectionsCalculator",
#ifndef _WIN32  // TODO windows: stdc++20 required
        "PacketClonerCalculator",
#endif
        "PacketGeneratorWrapperCalculator",
        "PacketInnerJoinCalculator",
        "PacketPresenceCalculator",
        "PacketResamplerCalculator",
        "PacketSequencerCalculator",
        "PacketThinnerCalculator",
        "PassThroughCalculator",
        "PreviousLoopbackCalculator",
        "QuantizeFloatVectorCalculator",
        "RectToRenderDataCalculator",
        "RectToRenderScaleCalculator",
        "RectTransformationCalculator",
        "RefineLandmarksFromHeatmapCalculator",
        "ResourceProviderCalculator",
        "RoiTrackingCalculator",
        "RotatedDetectionCalculator",
        "RotatedDetectionSerializationCalculator",
        "RoundRobinDemuxCalculator",
        "SegmentationCalculator",
        "SegmentationSerializationCalculator",
        "SegmentationSmoothingCalculator",
        "SequenceShiftCalculator",
        "SerializationCalculator",
        "SetLandmarkVisibilityCalculator",
        "SidePacketToStreamCalculator",
        "SplitAffineMatrixVectorCalculator",
        "SplitClassificationListVectorCalculator",
        "SplitDetectionVectorCalculator",
        "SplitFloatVectorCalculator",
        "SplitImageVectorCalculator",
        "SplitJointListCalculator",
        "SplitLandmarkListCalculator",
        "SplitLandmarkVectorCalculator",
        "SplitMatrixVectorCalculator",
        "SplitNormalizedLandmarkListCalculator",
        "SplitNormalizedLandmarkListVectorCalculator",
        "SplitNormalizedRectVectorCalculator",
        "SplitTensorVectorCalculator",
        "SplitTfLiteTensorVectorCalculator",
        "SplitUint64tVectorCalculator",
        "SsdAnchorsCalculator",
        "StreamToSidePacketCalculator",
        "StringToInt32Calculator",
        "StringToInt64Calculator",
        "StringToIntCalculator",
        "StringToUint32Calculator",
        "StringToUint64Calculator",
        "StringToUintCalculator",
        "SwitchDemuxCalculator",
        "SwitchMuxCalculator",
        "TensorsToClassificationCalculator",
        "TensorsToDetectionsCalculator",
        "TensorsToFloatsCalculator",
        "TensorsToLandmarksCalculator",
        "TensorsToSegmentationCalculator",
        "TfLiteConverterCalculator",
        "TfLiteCustomOpResolverCalculator",
        "TfLiteInferenceCalculator",
        "TfLiteModelCalculator",
        "TfLiteTensorsToDetectionsCalculator",
        "TfLiteTensorsToFloatsCalculator",
        "TfLiteTensorsToLandmarksCalculator",
        "ThresholdingCalculator",
        "ToImageCalculator",
        "TrackedDetectionManagerCalculator",
#ifndef _WIN32  // TODO windows: 'opencv2/optflow.hpp': No such file - will be available with opencv cmake on windows
        "Tvl1OpticalFlowCalculator",
#endif
        "TwoInputCalculator",
        "UpdateFaceLandmarksCalculator",
        "VideoPreStreamCalculator",
        "VisibilityCopyCalculator",
        "VisibilitySmoothingCalculator",
        "WarpAffineCalculator",
        "WarpAffineCalculatorCpu",
        "WorldLandmarkProjectionCalculator" });

    ASSERT_THAT(mediapipe::CalculatorBaseRegistry::GetRegisteredNames(), UnorderedElementsAreArray(expected)) << readableSetError(mediapipe::CalculatorBaseRegistry::GetRegisteredNames(), expected);
}

TEST(WhitelistRegistered, MediapipeSubgraphList) {
    std::unordered_set<std::string> expected({"FaceDetection",
        "FaceDetectionFrontDetectionToRoi",
        "FaceDetectionFrontDetectionsToRoi",
        "FaceDetectionShortRange",
        "FaceDetectionShortRangeByRoiCpu",
        "FaceDetectionShortRangeCpu",
        "FaceLandmarkCpu",
        "FaceLandmarkFrontCpu",
        "FaceLandmarkLandmarksToRoi",
        "FaceLandmarksFromPoseCpu",
        "FaceLandmarksFromPoseToRecropRoi",
        "FaceLandmarksModelLoader",
        "FaceLandmarksToRoi",
        "FaceTracking",
        "HandLandmarkCpu",
        "HandLandmarkModelLoader",
        "HandLandmarksFromPoseCpu",
        "HandLandmarksFromPoseToRecropRoi",
        "HandLandmarksLeftAndRightCpu",
        "HandLandmarksToRoi",
        "HandRecropByRoiCpu",
        "HandTracking",
        "HandVisibilityFromHandLandmarksFromPose",
        "HandWristForPose",
        "HolisticLandmarkCpu",
        "HolisticTrackingToRenderData",
        "InferenceCalculator",
        "IrisLandmarkCpu",
        "IrisLandmarkLandmarksToRoi",
        "IrisLandmarkLeftAndRightCpu",
        "IrisRendererCpu",
        "PoseDetectionCpu",
        "PoseDetectionToRoi",
        "PoseLandmarkByRoiCpu",
        "PoseLandmarkCpu",
        "PoseLandmarkFiltering",
        "PoseLandmarkModelLoader",
        "PoseLandmarksAndSegmentationInverseProjection",
        "PoseLandmarksToRoi",
        "PoseSegmentationFiltering",
        "SwitchContainer",
        "TensorsToFaceLandmarks",
        "TensorsToFaceLandmarksWithAttention",
        "TensorsToPoseLandmarksAndSegmentation"});

    ASSERT_THAT(mediapipe::SubgraphRegistry::GetRegisteredNames(), UnorderedElementsAreArray(expected)) << readableSetError(mediapipe::SubgraphRegistry::GetRegisteredNames(), expected);
}
