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
#include <memory>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../grpcservermodule.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../precision.hpp"
#include "../server.hpp"
#include "test_utils.hpp"

using namespace ovms;

class MediapipeValidationTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    const ovms::Module* grpcModule;
    KFSInferenceServiceImpl* impl;
    ::KFSRequest request;
    ::KFSResponse response;

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
        grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
        impl = &dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    }
    void prepareSingleInput() {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
        preparePredictRequest(request, inputsMeta);
        request.mutable_model_name()->assign("mediapipeDummyADAPTFULL");
    }
    void prepareDoubleInput() {
        request.Clear();
        response.Clear();
        inputs_info_t inputsMeta{
            {"in1", {DUMMY_MODEL_SHAPE, precision}},
            {"in2", {DUMMY_MODEL_SHAPE, precision}}};
        std::vector<float> requestData1{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
        std::vector<float> requestData2{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
        preparePredictRequest(request, inputsMeta, requestData1);
        request.mutable_model_name()->assign("mediapipeAddADAPTFULL");
    }
    void SetUp() override {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_all_graphs_adapter_full.json");
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

TEST_F(MediapipeValidationTest, Ok1Input) {
    prepareSingleInput();
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
}

TEST_F(MediapipeValidationTest, Ok2Inputs) {
    prepareDoubleInput();
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
}

TEST_F(MediapipeValidationTest, TooManyInputs) {
    prepareSingleInput();
    request.add_inputs()->CopyFrom(request.inputs(0));
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, NotEnoughInputs) {
    prepareSingleInput();
    request.clear_inputs();
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, MultipleInputsSameName) {
    prepareDoubleInput();
    request.mutable_inputs(1)->set_name("in1");
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, InputWithUnexpectedName) {
    prepareDoubleInput();
    request.mutable_inputs(1)->set_name("in3");
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, DataInNonRawField) {
    prepareSingleInput();
    request.mutable_raw_input_contents()->Clear();
    for (int i = 0; i < 10; i++)
        request.mutable_inputs(0)->mutable_contents()->mutable_fp32_contents()->Add();
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, NoDataInRawField) {
    prepareSingleInput();
    request.mutable_raw_input_contents()->Clear();
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, NegativeShape) {
    prepareSingleInput();
    request.mutable_inputs(0)->mutable_shape()->Set(0, -1);
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, ZeroShape) {
    prepareSingleInput();
    request.mutable_inputs(0)->mutable_shape()->Set(0, 0);
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, BufferShorterThanExpected) {
    prepareSingleInput();
    request.mutable_inputs(0)->mutable_shape()->Set(0, 20);
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MediapipeValidationTest, BufferLargerThanExpected) {
    prepareSingleInput();
    request.mutable_inputs(0)->mutable_shape()->Set(1, 1);
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// input has weird precision
TEST_F(MediapipeValidationTest, WrongPrecision) {
    prepareSingleInput();
    request.mutable_inputs(0)->set_datatype("unknown");
    ASSERT_EQ(impl->ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}
