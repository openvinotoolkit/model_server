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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "mediapipe/calculators/ovms/modelapiovmsadapter.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

class MediapipeFlowTest : public ::testing::TestWithParam<std::string> {
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
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!server.isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
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

class MediapipeTfLiteTensorTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/relative_paths/config_tflite_passthrough.json");
    }
};

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
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);

    // Checking that KFSPASS calculator copies requestData1 to the reponse so that we expect requestData1 on output
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}

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
                 << "OVMS deserialization & serialization of TfLiteTensors is not finised as well";
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
        cv::Mat imageRaw = cv::imread("/ovms/src/test/binaryutils/rgb4x4.jpg", cv::IMREAD_UNCHANGED);
        ASSERT_TRUE(!imageRaw.empty());
        cv::Mat image;
        size_t matFormat = convertKFSDataTypeToMatFormat(datatype);
        size_t matFormatWithChannels = CV_MAKETYPE(matFormat, 3);
        imageRaw.convertTo(image, matFormatWithChannels);

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
        cv::Mat imageRaw = cv::imread("/ovms/src/test/binaryutils/grayscale.jpg", cv::IMREAD_UNCHANGED);
        ASSERT_TRUE(!imageRaw.empty());
        cv::Mat grayscaled;
        size_t matFormat = convertKFSDataTypeToMatFormat(datatype);
        size_t matFormatWithChannels = CV_MAKETYPE(matFormat, 1);
        imageRaw.convertTo(grayscaled, matFormatWithChannels);

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
    cv::Mat imageRaw = cv::imread("/ovms/src/test/binaryutils/rgb4x4.jpg", cv::IMREAD_UNCHANGED);
    ASSERT_TRUE(!imageRaw.empty());
    cv::Mat image;
    size_t matFormat = convertKFSDataTypeToMatFormat("UINT8");
    size_t matFormatWithChannels = CV_MAKETYPE(matFormat, 3);
    imageRaw.convertTo(image, matFormatWithChannels);
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
    cv::Mat imageRaw = cv::imread("/ovms/src/test/binaryutils/rgb4x4.jpg", cv::IMREAD_UNCHANGED);
    ASSERT_TRUE(!imageRaw.empty());
    cv::Mat image;
    size_t matFormat = convertKFSDataTypeToMatFormat("INT64");
    size_t matFormatWithChannels = CV_MAKETYPE(matFormat, 3);
    imageRaw.convertTo(image, matFormatWithChannels);
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
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowTest, InferWithParams) {
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
    ov::AnyMap getRTInfo() const override {
        std::vector<std::string> mockLabels;
        for (size_t i = 0; i < 5; i++) {
            mockLabels.emplace_back(std::to_string(i));
        }
        ov::AnyMap configuration = {
            {"layout", "data:HWCN"},
            {"resize_type", "unnatural"},
            {"labels", mockLabels}};
        return configuration;
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
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config.json"));

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
    checkModelInfo(*servableMetadataRtInfo);
    OVMS_ServableMetadataDelete(servableMetadata);
}

TEST(Mediapipe, MetadataDummy) {
    ConstructorEnabledModelManager manager;
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", "/ovms/src/test/mediapipe/graphdummy.pbtxt"};
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

class DummyMediapipeGraphDefinition : public MediapipeGraphDefinition {
public:
    std::string inputConfig;

public:
    DummyMediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config,
        std::string inputConfig) :
        MediapipeGraphDefinition(name, config, nullptr, nullptr) {}

    // Do not read from path - use predefined config contents
    Status validateForConfigFileExistence() override {
        this->chosenConfig = this->inputConfig;
        return StatusCode::OK;
    }
};

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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME);
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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME);
}

TEST(Mediapipe, MetadataEmptyConfig) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = "";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

const std::vector<std::string> mediaGraphsDummy{"mediaDummy",
    "mediaDummyADAPTFULL"};
const std::vector<std::string> mediaGraphsAdd{"mediapipeAdd",
    "mediapipeAddADAPTFULL"};

class MediapipeConfig : public MediapipeFlowTest {
public:
    void TearDown() override {}
};

const std::string NAME = "Name";
TEST_F(MediapipeConfig, MediapipeGraphDefinitionNonExistentFile) {
    ConstructorEnabledModelManager manager;
    MediapipeGraphConfig mgc{"noname", "/ovms/NONEXISTENT_FILE"};
    MediapipeGraphDefinition mgd(NAME, mgc);
    EXPECT_EQ(mgd.validate(manager), StatusCode::FILE_INVALID);
}

TEST_F(MediapipeConfig, MediapipeAdd) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsAdd) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        EXPECT_NE(graphDefinition, nullptr);
        EXPECT_EQ(graphDefinition->getStatus().isAvailable(), true);
    }
    manager.join();
}

TEST_F(MediapipeConfig, MediapipeTfLiteTagsNotSupported) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_tflite_passthrough.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName("mpTfLiteTensorDummy");
    EXPECT_NE(graphDefinition, nullptr);
    EXPECT_EQ(graphDefinition->getStatus().getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    manager.join();
}

TEST_F(MediapipeConfig, MediapipeDummyWithDag) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_dag.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsDummy) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        EXPECT_NE(graphDefinition, nullptr);
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
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!server.isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
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

TEST_F(MediapipeConfig, MediapipeFullRelativePaths) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfig) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_add_subconfig.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfigBasePath) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy_subconfig_base_path.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("graphaddadapterfull");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("graphadd");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsNegative) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy_negative.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), false);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), false);

    manager.join();
}

class MediapipeConfigChanges : public TestWithTempDir {
    void SetUp() override {
        TestWithTempDir::SetUp();
    }

public:
    static const std::string mgdName;
    static const std::string configFileWithGraphPathToReplace;
    static const std::string configFileWithGraphPathToReplaceWithoutModel;
    static const std::string configFileWithGraphPathToReplaceAndSubconfig;
    static const std::string configFileWithoutGraph;
    static const std::string pbtxtContent;
    static const std::string pbtxtContentNonexistentCalc;
    template <typename Request, typename Response>
    static void checkStatus(ModelManager& manager, ovms::StatusCode code) {
        std::shared_ptr<MediapipeGraphExecutor> executor;
        Request request;
        Response response;
        auto status = manager.createPipeline(executor, mgdName, &request, &response);
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

TEST_F(MediapipeConfigChanges, AddProperGraphThenRetireThenAddAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/subconfig.json";
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

TEST_F(MediapipeConfigChanges, AddImroperGraphThenFixWithReloadThenBreakAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/subconfig.json";
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
    createConfigFileWithContent(configFileContent, subconfigFilePath);
    configFileContent = configFileWithGraphPathToReplaceAndSubconfig;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    const std::string subconfigPathToReplace{"SUBCONFIG_PATH"};
    configFileContent.replace(configFileContent.find(subconfigPathToReplace), subconfigPathToReplace.size(), subconfigFilePath);
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
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), MediapipeGraphDefinition::getStreamNamePair("TENSOR:out"));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::TFTENSOR), MediapipeGraphDefinition::getStreamNamePair("TFTENSOR:out"));
    EXPECT_EQ(streamNameTypePair_t("input", mediapipe_packet_type_enum::OVTENSOR), MediapipeGraphDefinition::getStreamNamePair("OVTENSOR:input"));
    EXPECT_EQ(streamNameTypePair_t("input", mediapipe_packet_type_enum::KFS_REQUEST), MediapipeGraphDefinition::getStreamNamePair("REQUEST:input"));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::KFS_RESPONSE), MediapipeGraphDefinition::getStreamNamePair("RESPONSE:out"));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MEDIAPIPE_IMAGE), MediapipeGraphDefinition::getStreamNamePair("IMAGE:out"));
    // string after suffix doesn't matter
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), MediapipeGraphDefinition::getStreamNamePair("TENSOR1:out"));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::MPTENSOR), MediapipeGraphDefinition::getStreamNamePair("TENSOR_1:out"));
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::KFS_RESPONSE), MediapipeGraphDefinition::getStreamNamePair("RESPONSE_COSTAM:out"));
    // number as additional part doesn't affect recognized type
    EXPECT_EQ(streamNameTypePair_t("in", mediapipe_packet_type_enum::MPTENSOR), MediapipeGraphDefinition::getStreamNamePair("TENSOR:1:in"));
    // negative
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::UNKNOWN), MediapipeGraphDefinition::getStreamNamePair("TENSO:out"));             // negative - non-matching tag
    EXPECT_EQ(streamNameTypePair_t("out", mediapipe_packet_type_enum::UNKNOWN), MediapipeGraphDefinition::getStreamNamePair("SOME_STRANGE_TAG:out"));  // negative - non-matching tag
    EXPECT_EQ(streamNameTypePair_t("in", mediapipe_packet_type_enum::UNKNOWN), MediapipeGraphDefinition::getStreamNamePair("in"));
}

// TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfigNegative) {
//     ConstructorEnabledModelManager manager;
//     auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_add_subconfig_negative.json");
//     EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
//     manager.join();
// }

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
