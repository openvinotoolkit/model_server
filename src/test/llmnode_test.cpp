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
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <pybind11/embed.h>

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_graph_executor_impl.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../python/pythoninterpretermodule.hpp"
#include "../python/pythonnoderesources.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

#include "opencv2/opencv.hpp"
#include "python/python_backend.hpp"
#include "test_utils.hpp"

using namespace ovms;

class LLMFlowTest : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::STRING;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
    }

    void SetUp() override {
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class LLMFlowKfsTest : public LLMFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/llm/config_llm_dummy_kfs.json");
    }
};

// --------------------------------------- OVMS LLM nodes tests

TEST_F(LLMFlowKfsTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "llmDummyKFS";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<std::string> requestData1{"What is OpenVINO?"};
    std::vector<std::string> requestData2{""};
    preparePredictRequest(request, inputsMeta, requestData1, false);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // Checking that KFSPASS calculator copies requestData1 to the reponse so that we expect requestData1 on output
    //checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}
