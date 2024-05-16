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
// Test disabled by default - needs LLM models to work in /workspace directory:
// openvino_detokenizer.bin  openvino_detokenizer.xml  openvino_model.bin  openvino_model.xml  openvino_tokenizer.bin  openvino_tokenizer.xml
// Model that was used:
// https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
// converted with optimum cli
TEST_F(LLMFlowKfsTest, DISABLED_Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "llmDummyKFS";
    request.Clear();
    response.Clear();
    std::vector<std::string> requestData1{"What is OpenVINO?"};
    std::string expectedResponse = "\n\nOpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on a variety";

    prepareInferStringRequest(request, "in", requestData1, false);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    const std::string& content = response.raw_output_contents(0);
    ASSERT_EQ(expectedResponse, content);
}

TEST_F(LLMFlowKfsTest, LLMNodeNameMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/workspace/"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_NAME);
}

TEST_F(LLMFlowKfsTest, LLMNodeOptionsMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode2"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_OPTIONS);
}

TEST_F(LLMFlowKfsTest, LLMNodeNameExists) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out1"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/workspace/"
                }
            }
        }
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/workspace/"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_NAME_ALREADY_EXISTS);
}

TEST_F(LLMFlowKfsTest, LLMNodeBadWorkspacePath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/bad_path_to_workspace/"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMFlowKfsTest, LLMNodeBadWorkspacePathEmpty) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: ""
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMFlowKfsTest, LLMNodeBadWorkspacePathFile) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/ovms/src/test/llm/config_llm_dummy_kfs.json"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMFlowKfsTest, LLMNodeResourceInitFailed) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "REQUEST:in"
    output_stream: "RESPONSE:out"
        node {
            name: "llmNode"
            calculator: "LLMCalculator"
            input_side_packet: "LLM_NODE_RESOURCES:py"
            input_stream: "REQUEST:in"
            output_stream: "RESPONSE:out2"
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    workspace_path: "/"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
    ASSERT_EQ(mediapipeDummy.getLLMNodeResources("llmNode"), nullptr);
}
