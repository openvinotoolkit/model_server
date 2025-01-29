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
#include <opencv2/opencv.hpp>
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
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

/***
 * Test assumptions about MP framework
 * */
class MediapipeFrameworkTest : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
    }

    void SetUp() override {
    }
    void TearDown() {
        if (server.isLive()) {
            server.setShutdownRequest(1);
            t->join();
            server.setShutdownRequest(0);
        }
    }
};

class MediapipeNegativeFrameworkTest : public MediapipeFrameworkTest {
};

// purpose of this test is to ensure there is no hang in case of one of the graph nodes
// not producing output packet
TEST_F(MediapipeNegativeFrameworkTest, NoOutputPacketProduced) {
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_no_calc_output_stream.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_no_calc_output_stream"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    auto status = impl.ModelInfer(nullptr, &request, &response);
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
}

TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringProcess) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_process.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_process"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringGetContract) {
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_getcontract.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_getcontract"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringGetOpen) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_open.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_open"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringClose) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_close.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_close"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
