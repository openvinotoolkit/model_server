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
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
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

#include "../python/python_backend.hpp"
#include "opencv2/opencv.hpp"
#include "test_utils.hpp"

namespace py = pybind11;
using namespace ovms;
using namespace py::literals;

using testing::HasSubstr;
using testing::Not;

/*
Tests in that file base on single fixture with static setup and tear down.
We do this because we don't want to restart interpreter in the tests.
It's launching along with the server and even though most tests will not use the server, the interpreter remains initialized.
*/
std::unique_ptr<std::thread> serverThread;

class PythonFlowTest : public ::testing::Test {
protected:
    ovms::ExecutionContext defaultExecutionContext{ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::Predict};
    std::unique_ptr<MediapipeServableMetricReporter> reporter;

public:
    void SetUp() override {
        this->reporter = std::make_unique<MediapipeServableMetricReporter>(nullptr, nullptr, "");  // disabled metric reporter
    }

    static void SetUpTestSuite() {
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/python/mediapipe_add_python_node.json");
        ovms::Server::instance().setShutdownRequest(0);
        std::string port = "9178";
        randomizePort(port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath.c_str(),
            (char*)"--port",
            (char*)port.c_str(),
            (char*)"--file_system_poll_wait_seconds",
            (char*)"0"};
        int argc = 7;
        serverThread.reset(new std::thread([&argc, &argv]() {
            EXPECT_EQ(EXIT_SUCCESS, ovms::Server::instance().start(argc, argv));
        }));
        EnsureServerStartedWithTimeout(ovms::Server::instance(), 5);
    }
    static void TearDownTestSuite() {
        ovms::Server::instance().setShutdownRequest(1);
        serverThread->join();
        ovms::Server::instance().setShutdownRequest(0);
        std::string path = getGenericFullPathForTmp("/tmp/pythonNodeTestRemoveFile.txt");
        ASSERT_TRUE(!std::filesystem::exists(path));
    }
};

static PythonBackend* getPythonBackend() {
    return dynamic_cast<const ovms::PythonInterpreterModule*>(ovms::Server::instance().getModule(PYTHON_INTERPRETER_MODULE_NAME))->getPythonBackend();
}

// --------------------------------------- OVMS initializing Python nodes tests

TEST_F(PythonFlowTest, InitializationPass) {
    ModelManager* manager;
    manager = &(dynamic_cast<const ovms::ServableManagerModule*>(ovms::Server::instance().getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    auto graphDefinition = manager->getMediapipeFactory().findDefinitionByName("mediapipePythonBackend");
    ASSERT_NE(graphDefinition, nullptr);
    EXPECT_TRUE(graphDefinition->getStatus().isAvailable());
}

TEST_F(PythonFlowTest, FinalizationPass) {
    ModelManager* manager;
    std::string path = getGenericFullPathForTmp("/tmp/pythonNodeTestRemoveFile.txt");
    manager = &(dynamic_cast<const ovms::ServableManagerModule*>(ovms::Server::instance().getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    auto graphDefinition = manager->getMediapipeFactory().findDefinitionByName("mediapipePythonBackend");
    ASSERT_NE(graphDefinition, nullptr);
    EXPECT_TRUE(graphDefinition->getStatus().isAvailable());
    ASSERT_TRUE(std::filesystem::exists(path));
}

TEST_F(PythonFlowTest, ExecutorWithEmptyOptions) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID);
}

TEST_F(PythonFlowTest, PythonNodeFileDoesNotExist) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/22symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST);
}

TEST_F(PythonFlowTest, PythonNodeClassDoesNotExist) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_missing_class.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(PythonFlowTest, PythonNodeExecuteNotImplemented) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_execute_no_method.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(PythonFlowTest, PythonNodeNameAlreadyExist) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_execute_wrong_return_value.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out3"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_NAME_ALREADY_EXISTS);
}

TEST_F(PythonFlowTest, PythonNodeInitFailed) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_no_method.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
}

TEST_F(PythonFlowTest, PythonNodeInitFailedImportOutsideTheClassError) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_import_outside_class_error.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(PythonFlowTest, PythonNodeInitException) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_throw_exception.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(PythonFlowTest, PythonNodeOptionsMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_MISSING_OPTIONS);
}

TEST_F(PythonFlowTest, PythonNodeNameMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_no_method.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_MISSING_NAME);
}

TEST_F(PythonFlowTest, PythonNodeNameDoesNotExist) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_execute_wrong_return_value.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.getPythonNodeResources("pythonNode4"), nullptr);
}

TEST_F(PythonFlowTest, PythonNodeInitMembers) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_initialize_with_class_members.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResources* nodeRes = mediapipeDummy.getPythonNodeResources("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    py::gil_scoped_acquire acquire;
    try {
        using namespace py::literals;

        // Casting and recasting needed for ASSER_EQ to work
        std::string modelName = nodeRes->ovmsPythonModel.get()->attr("model_name").cast<std::string>();
        std::string expectedName = py::str("testModel").cast<std::string>();

        ASSERT_EQ(modelName, expectedName);
        int executionTime = nodeRes->ovmsPythonModel.get()->attr("execution_time").cast<int>();
        ASSERT_EQ(executionTime, 300);
        py::list modelInputs = nodeRes->ovmsPythonModel.get()->attr("model_inputs");

        py::list expectedInputs = py::list();
        expectedInputs.attr("append")(py::str("input1"));
        expectedInputs.attr("append")(py::str("input2"));

        for (pybind11::size_t i = 0; i < modelInputs.size(); i++) {
            py::str inputName = py::cast<py::str>(modelInputs[i]);
            ASSERT_EQ(inputName.cast<std::string>(), expectedInputs[i].cast<std::string>());
        }
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << "Python pybind exception: " << e.what();
    } catch (...) {
        ASSERT_EQ(1, 0) << "General error ";
    }
}

TEST_F(PythonFlowTest, PythonNodePassInitArguments) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR_IN1:in1"
    input_stream: "OVMS_PY_TENSOR_IN2:in2"
    output_stream: "OVMS_PY_TENSOR_OUT1:out1"
    output_stream: "OVMS_PY_TENSOR_OUT2:out2"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "IN1:in1"
            input_stream: "IN2:in2"
            output_stream: "OUT1:out1"
            output_stream: "OUT2:out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_initialize_with_arguments.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResources* nodeRes = mediapipeDummy.getPythonNodeResources("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    py::gil_scoped_acquire acquire;
    try {
        using namespace py::literals;

        // Casting and recasting needed for ASSER_EQ to work
        std::string modelName = nodeRes->ovmsPythonModel.get()->attr("node_name").cast<std::string>();
        std::string expectedName = py::str("pythonNode2").cast<std::string>();
        ASSERT_EQ(modelName, expectedName);

        py::list inputStream = nodeRes->ovmsPythonModel.get()->attr("input_names");
        py::list expectedInputs = py::list();
        expectedInputs.attr("append")(py::str("in1"));
        expectedInputs.attr("append")(py::str("in2"));

        for (pybind11::size_t i = 0; i < inputStream.size(); i++) {
            py::str inputName = py::cast<py::str>(inputStream[i]);
            ASSERT_EQ(inputName.cast<std::string>(), expectedInputs[i].cast<std::string>());
        }

        py::list outputStream = nodeRes->ovmsPythonModel.get()->attr("output_names");
        py::list expectedOutputs = py::list();
        expectedOutputs.attr("append")(py::str("out1"));
        expectedOutputs.attr("append")(py::str("out2"));

        for (pybind11::size_t i = 0; i < outputStream.size(); i++) {
            py::str outputName = py::cast<py::str>(outputStream[i]);
            ASSERT_EQ(outputName.cast<std::string>(), expectedOutputs[i].cast<std::string>());
        }
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << "Python pybind exception: " << e.what();
    } catch (...) {
        ASSERT_EQ(1, 0) << "General error ";
    }
}

TEST_F(PythonFlowTest, PythonNodePassArgumentsToConstructor) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResources* nodeRes = mediapipeDummy.getPythonNodeResources("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    py::gil_scoped_acquire acquire;
    try {
        using namespace py::literals;

        // Casting and recasting needed for ASSER_EQ to work
        py::dict modelOutputs = nodeRes->ovmsPythonModel.get()->attr("model_outputs");
        py::int_ size = modelOutputs.size();
        ASSERT_EQ(size, 0);
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << "Python pybind exception: " << e.what();
    } catch (...) {
        ASSERT_EQ(1, 0) << "General error";
    }
}

TEST_F(PythonFlowTest, PythonNodeLoopbackDefinedOnlyOnInput) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, PythonNodeLoopbackDefinedOnlyOnOutput) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

// InputStreamHandler is validated only on CalculatorNode::PrepareForRun/
// This is called in Graph::Run/StartRun.
// This is called only when input side packets are ready, meaning only when first request arrives.
// With current MediaPipe implementation it is impossible to validate indexes during graph loading.
TEST_F(PythonFlowTest, DISABLED_PythonNodeLoopback_SyncSet_WrongIndex) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',  # correct index
                back_edge: true
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:1"  # wrong index
                        }
                    }
                }
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;

    // Should not be OK, but MediaPipe does not validate it at initialization phase
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);

    // Current state allows graph to validate and perform inference which leads to RET_CHECK in PrepareForRun
}

TEST_F(PythonFlowTest, PythonNodeLoopback_StreamInfo_WrongIndex) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_info: {
                tag_index: 'LOOPBACK:1',  # wrong index
                back_edge: true
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"  # correct index
                        }
                    }
                }
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, PythonNodeLoopback_StreamInfo_BackEdgeFalse) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: false  # should be true
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

// MediaPipe understands that loopback is not a cycle and there is no data source that will ever feed it.
// Loopback input in the node is not connected to graph input or output from another node.
TEST_F(PythonFlowTest, PythonNodeLoopback_StreamInfo_Missing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

// MediaPipe does allow such connection and treats loopback as connected (due to the back edge)
// However, such graph will hang since nothing feeds loopback with initial data
// During the hang, the graph node will wait for secondary input (besides "input")
TEST_F(PythonFlowTest, PythonNodeLoopback_SyncSet_Missing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR1:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    GTEST_SKIP() << "Cycle found, the graph will wait for data forever as expected";

    KFSRequest req;
    KFSResponse res;

    req.set_model_name("mediaDummy");

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);  // hangs
}
TEST_F(PythonFlowTest, PythonNodeLoopback_Correct) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            input_stream: "LOOPBACK:loopback"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
            output_stream: "OUTPUT:output"
            output_stream: "LOOPBACK:loopback"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    req.set_model_name("mediaDummy");

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}
// Wrapper on the OvmsPyTensor of datatype FP32 and shape (1, num_elements)
// where num_elements is the size of C++ float array. See createTensor static method.
template <typename T>
class SimpleTensor {
public:
    std::string name;
    std::string datatype;
    void* data;
    int numElements;
    size_t size;
    std::vector<py::ssize_t> shape;
    std::unique_ptr<PyObjectWrapper<py::object>> pyTensor;

    static SimpleTensor createTensor(const std::string& name, T* data, const std::string& datatype, int numElements) {
        SimpleTensor tensor;
        tensor.name = name;
        tensor.data = (void*)data;
        tensor.datatype = datatype;
        tensor.numElements = numElements;
        tensor.size = numElements * sizeof(T);
        tensor.shape = std::vector<py::ssize_t>{1, numElements};
        EXPECT_TRUE(getPythonBackend()->createOvmsPyTensor(tensor.name, (void*)tensor.data, tensor.shape, tensor.datatype, tensor.size, tensor.pyTensor));
        return tensor;
    }

    static std::vector<T> readVectorFromOutput(const std::string& outputName, int numElements, const mediapipe::CalculatorRunner* runner, int packetIndex = 0) {
        const PyObjectWrapper<py::object>& pyOutput = runner->Outputs().Tag(outputName).packets[packetIndex].Get<PyObjectWrapper<py::object>>();
        T* outputData = (T*)pyOutput.getProperty<void*>("ptr");
        std::vector<T> output;
        output.assign(outputData, outputData + numElements);
        return output;
    }

    std::vector<T> getIncrementedVector(int value = 1) {
        // SimpleTensor is expected to hold data in shape (1, X),
        // therefore we iterate over the second dimension as it holds the actual data
        std::vector<T> output;
        T* fpData = (T*)data;
        for (int i = 0; i < shape[1]; i++) {
            output.push_back(fpData[i] + value);
        }
        return output;
    }
};

// ---------------------------------- OVMS deserialize and serialize tests

// This part duplicates some parts of tests in mediapipeflow_test.cpp file.
// We could think about moving them there in the future, but for now we need to keep
// all tests involving Python interpreter in a single test suite.

class MockedMediapipeGraphExecutorPy : public ovms::MediapipeGraphExecutor {
public:
    MockedMediapipeGraphExecutorPy(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
        stream_types_mapping_t inputTypes,
        stream_types_mapping_t outputTypes,
        std::vector<std::string> inputNames, std::vector<std::string> outputNames,
        const std::shared_ptr<PythonNodeResourcesMap>& pythonNodeResourcesMap,
        PythonBackend* pythonBackend,
        MediapipeServableMetricReporter* mediapipeServableMetricReporter, GraphIdGuard&& guard) :
        MediapipeGraphExecutor(name, version, config, inputTypes, outputTypes, inputNames, outputNames, pythonNodeResourcesMap, {}, pythonBackend, mediapipeServableMetricReporter, std::move(guard)) {}
};

TEST_F(PythonFlowTest, SerializePyObjectWrapperToKServeResponse) {
    ovms::stream_types_mapping_t mapping;
    mapping["python_result"] = mediapipe_packet_type_enum::OVMS_PY_TENSOR;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;
    const ::mediapipe::CalculatorGraphConfig config;
    std::shared_ptr<GenAiServableMap> gasm = std::make_shared<GenAiServableMap>();
    std::shared_ptr<PythonNodeResourcesMap> pnsm = std::make_shared<PythonNodeResourcesMap>();
    std::shared_ptr<GraphQueue> queue = std::make_shared<GraphQueue>(config, pnsm, gasm, 1);
    GraphIdGuard guard(queue);
    auto executor = MockedMediapipeGraphExecutorPy("", "", config, mapping, mapping, inputNames, outputNames, pnsm, getPythonBackend(), this->reporter.get(), std::move(guard));

    std::string datatype = "FP32";
    std::string name = "python_result";
    int numElements = 3;
    float input[] = {1.0, 2.0, 3.0};
    SimpleTensor<float> tensor = SimpleTensor<float>::createTensor(name, input, datatype, numElements);

    ::inference::ModelInferResponse response;

    ::mediapipe::Packet packet = ::mediapipe::Adopt<PyObjectWrapper<py::object>>(tensor.pyTensor.release());
    ASSERT_EQ(onPacketReadySerializeImpl("id", name, "1", name, mediapipe_packet_type_enum::OVMS_PY_TENSOR, packet, response), StatusCode::OK);
    ASSERT_EQ(response.outputs_size(), 1);
    auto output = response.outputs(0);
    ASSERT_EQ(output.datatype(), "FP32");
    ASSERT_EQ(output.shape_size(), 2);
    ASSERT_EQ(output.shape(0), 1);
    ASSERT_EQ(output.shape(1), 3);
    ASSERT_EQ(response.raw_output_contents_size(), 1);
    ASSERT_EQ(response.raw_output_contents().at(0).size(), 3 * sizeof(float));
    std::vector<float> expectedOutputData{1.0, 2.0, 3.0};
    std::vector<float> outputData;
    const float* outputDataPtr = reinterpret_cast<const float*>(response.raw_output_contents().at(0).data());
    outputData.assign(outputDataPtr, outputDataPtr + numElements);
    ASSERT_EQ(expectedOutputData, outputData);
}

// ---------------------------------- PythonExecutorCalculcator tests

static void addInputItem(const std::string& tag, std::unique_ptr<PyObjectWrapper<py::object>>& input, int64_t timestamp,
    mediapipe::CalculatorRunner* runner) {
    runner->MutableInputs()->Tag(tag).packets.push_back(
        mediapipe::Adopt<PyObjectWrapper<py::object>>(input.release()).At(mediapipe::Timestamp(timestamp)));
}

static void addInputSidePacket(std::string tag, std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>& input,
    int64_t timestamp, mediapipe::CalculatorRunner* runner) {
    runner->MutableSidePackets()->Tag(tag) = mediapipe::MakePacket<std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>>(input).At(mediapipe::Timestamp(timestamp));
}

static std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>> prepareInputSidePacket(const std::string& handlerPath, PythonBackend* pythonBackend) {
    // Create side packets
    auto fsHandlerPath = std::filesystem::path(handlerPath);
    fsHandlerPath.replace_extension();

    std::string parentPath = fsHandlerPath.parent_path().string();
    std::string filename = fsHandlerPath.filename().string();

    py::gil_scoped_acquire acquire;
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(parentPath.c_str());
    py::module_ script = py::module_::import(filename.c_str());
    py::object OvmsPythonModel = script.attr("OvmsPythonModel");
    py::object pythonModel = OvmsPythonModel();

    std::shared_ptr<PythonNodeResources> nodeResources = std::make_shared<PythonNodeResources>(pythonBackend);
    nodeResources->ovmsPythonModel = std::make_unique<py::object>(pythonModel);

    std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>> nodesResources{{"pythonNode", nodeResources}};
    return nodesResources;
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOut) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleThreeOut) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR1:input_a"
    input_stream: "OVMS_PY_TENSOR2:input_b"
    input_stream: "OVMS_PY_TENSOR3:input_c"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT1:input_a"
            input_stream: "INPUT2:input_b"
            input_stream: "INPUT3:input_c"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/concatenate_input_arrays.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data1{1.0f, 20.0f, 3.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input_a", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, 3}, ovms::Precision::FP32}, data1, true);

    const std::vector<float> data2{1.0f, 20.0f, 3.0f, 1.0f};
    prepareKFSInferInputTensor(req, "input_b", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, 4}, ovms::Precision::FP32}, data2, true);

    const std::vector<float> data3{1.0f, 20.0f, 3.0f, 1.0f, 20.0f};
    prepareKFSInferInputTensor(req, "input_c", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, 5}, ovms::Precision::FP32}, data3, true);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    ASSERT_EQ(res.model_name(), "mediaDummy");
    ASSERT_EQ(res.outputs_size(), 1);
    ASSERT_EQ(res.raw_output_contents_size(), 1);
    ASSERT_EQ(res.mutable_raw_output_contents(0)->size(), (data1.size() + data2.size() + data3.size()) * sizeof(float));
    std::vector<float> expectedData{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f};  // concatenated vectors data1, data2, data3
    ASSERT_EQ(std::memcmp(res.mutable_raw_output_contents(0)->data(), expectedData.data(), res.mutable_raw_output_contents(0)->size()), 0);
}

TEST_F(PythonFlowTest, PythonCalculatorTestReturnCustomDatatype) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/return_custom_datatype.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    auto it = res.outputs().begin();
    const auto& output_proto = *it;
    ASSERT_EQ(output_proto.datatype(), "9w");  // "9w is memoryview format of numpy array containing single string element: 'my string'"
}

TEST_F(PythonFlowTest, PythonCalculatorTestReturnNotListOrIteratorObject) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/return_none_object.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(PythonFlowTest, PythonCalculatorTestReturnListWithNonTensorObject) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/return_non_tensor_object.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutMultiNodeNoTags) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:first"
    output_stream: "OVMS_PY_TENSOR:third"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "first"
            output_stream: "second"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "second"
            output_stream: "third"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("third", data, req, res, 2 /* expect +2 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutMultiNodeOnlyTags) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:first"
    output_stream: "OVMS_PY_TENSOR:last"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:first"
            output_stream: "OUTPUT:second"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:second"
            output_stream: "OUTPUT:third"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:third"
            output_stream: "OUTPUT:last"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("last", data, req, res, 3 /* expect +3 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutMultiNodeTagsAndIndexes) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:first"
    output_stream: "OVMS_PY_TENSOR:last"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:0:first"
            output_stream: "OUTPUT:0:second"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:0:second"
            output_stream: "OUTPUT:0:third"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:0:third"
            output_stream: "OUTPUT:0:last"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("last", data, req, res, 3 /* expect +3 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutTwoConvertersOnTheOutside) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVTENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:output1"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                    key: "OVMS_PY_TENSOR"
                    value: "input"
                    }
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output1"
            output_stream: "OUTPUT:output2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output2"
            output_stream: "OVTENSOR:out"
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestConvertersUnsupportedTypeInPythonTensor) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVTENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:output1"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                    key: "OVMS_PY_TENSOR"
                    value: "input"
                    }
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output1"
            output_stream: "OUTPUT:output2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/return_custom_datatype.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output2"
            output_stream: "OVTENSOR:out"
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<int64_t> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I64}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutTwoConvertersInTheMiddle) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:output1"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output1"
            output_stream: "OVTENSOR:output2"
        }
        node {
            name: "pythonNode3"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:output2"
            output_stream: "OVMS_PY_TENSOR:output3"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                    key: "OVMS_PY_TENSOR"
                    value: "input2"
                    }
                }
            }
        }
        node {
            name: "pythonNode4"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output3"
            output_stream: "OUTPUT:out"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 2 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInTwoOutTwoParallelExecutors) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:in"
    output_stream: "OVMS_PY_TENSOR1:out1"
    output_stream: "OVMS_PY_TENSOR2:out2"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out1"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment_by_2.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out1", data, req, res, 1 /* expect +1 */, 1, "mediaDummy", 2);
    checkDummyResponse("out2", data, req, res, 2 /* expect +2 */, 1, "mediaDummy", 2);
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInTwoOutTwoParallelExecutorsWithConverters) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVTENSOR1:out1"
    output_stream: "OVTENSOR2:out2"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:output1"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                    key: "OVMS_PY_TENSOR"
                    value: "output1"
                    }
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output1"
            output_stream: "OUTPUT:output2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output1"
            output_stream: "OUTPUT:output3"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment_by_2.py"
                }
            }
        }
        node {
            name: "pythonNode4"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output2"
            output_stream: "OVTENSOR:out1"
        }
        node {
            name: "pythonNode5"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output3"
            output_stream: "OVTENSOR:out2"
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out1", data, req, res, 1 /* expect +1 */, 1, "mediaDummy", 2);
    checkDummyResponse("out2", data, req, res, 2 /* expect +2 */, 1, "mediaDummy", 2);
}

TEST_F(PythonFlowTest, ConverterWithInvalidInputOutputTags) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "INVALID1:in"
            output_stream: "INVALID2:out"
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithOVTENSORInputAndOutput) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVTENSOR:out"
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithOVMS_PY_TENSORInputAndOutput) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithMissingOptions) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithEmptyOptions) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithEmptyTagMap) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {}
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithInvalidTagInMap) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                        key: "INVALID"
                        value: "TAG"
                    }
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}

TEST_F(PythonFlowTest, ConverterWithValidAndInvalidTagInMap) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:out"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                        key: "INVALID"
                        value: "TAG"
                    }
                    tag_to_output_tensor_names {
                        key: "OVMS_PY_TENSOR"
                        value: "input"
                    }
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
}

TEST_F(PythonFlowTest, PythonCalculatorTestMultiInMultiOut) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR1:input1"
    input_stream: "OVMS_PY_TENSOR2:input2"
    input_stream: "OVMS_PY_TENSOR3:input3"
    output_stream: "OVMS_PY_TENSOR1:output1"
    output_stream: "OVMS_PY_TENSOR2:output2"
    output_stream: "OVMS_PY_TENSOR3:output3"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT1:input1"
            input_stream: "INPUT2:input2"
            input_stream: "INPUT3:input3"
            output_stream: "OUTPUT1:output1"
            output_stream: "OUTPUT2:output2"
            output_stream: "OUTPUT3:output3"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data1{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    const std::vector<float> data2{20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f, 1.0f};
    const std::vector<float> data3{3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f, 1.0f, 20.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input1", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data1, false);
    prepareKFSInferInputTensor(req, "input2", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data2, false);
    prepareKFSInferInputTensor(req, "input3", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data3, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);
    checkDummyResponse("output1", data1, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("output2", data2, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("output3", data3, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
}

static void setupTestPipeline(std::shared_ptr<MediapipeGraphExecutor>& pipeline, std::unique_ptr<DummyMediapipeGraphDefinition>& mediapipeDummy) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    mediapipeDummy = std::make_unique<DummyMediapipeGraphDefinition>("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy->inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy->validate(manager), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy->create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);
}

TEST_F(PythonFlowTest, PythonCalculatorScalarNoShape) {
    KFSRequest req;
    KFSResponse res;

    float inputScalar = 6.0;
    const std::vector<float> data{inputScalar};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{ovms::signed_shape_t{}, ovms::Precision::FP32}, data, false);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setupTestPipeline(pipeline, mediapipeDummy);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    ASSERT_EQ(res.model_name(), "mediaDummy");
    ASSERT_EQ(res.outputs_size(), 1);
    ASSERT_EQ(res.raw_output_contents_size(), 1);
    ASSERT_EQ(res.outputs().begin()->name(), "output") << "Did not find:"
                                                       << "output";
    const auto& output = *res.outputs().begin();
    std::string* content = res.mutable_raw_output_contents(0);

    ASSERT_EQ(output.shape_size(), 0);
    ASSERT_EQ(content->size(), sizeof(float));

    ASSERT_EQ(*((float*)content->data()), inputScalar + 1);
}

TEST_F(PythonFlowTest, PythonCalculatorZeroDimension) {
    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{ovms::signed_shape_t{1, 32, 32, 0, 1}, ovms::Precision::FP32}, data, false);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setupTestPipeline(pipeline, mediapipeDummy);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    ASSERT_EQ(res.model_name(), "mediaDummy");
    ASSERT_EQ(res.outputs_size(), 1);
    ASSERT_EQ(res.raw_output_contents_size(), 1);
    ASSERT_EQ(res.outputs().begin()->name(), "output") << "Did not find:"
                                                       << "output";
    const auto& output = *res.outputs().begin();
    std::string* content = res.mutable_raw_output_contents(0);

    ASSERT_EQ(output.shape_size(), 5);
    ASSERT_EQ(content->size(), 0);
}

TEST_F(PythonFlowTest, PythonCalculatorTestBadExecute) {
    const std::vector<std::pair<std::string, std::string>> BAD_EXECUTE_SCRIPTS_CASES{
        {"bad_execute_wrong_signature", "Error occurred during graph execution"},
        {"bad_execute_illegal_operation", "Error occurred during graph execution"},
        {"bad_execute_import_error", "Error occurred during graph execution"},
        {"bad_execute_wrong_return_value", "Error occurred during graph execution"}};

    for (const auto& testCase : BAD_EXECUTE_SCRIPTS_CASES) {
        std::string handlerPath = testCase.first;
        std::string testPbtxt = R"(
            calculator: "PythonExecutorCalculator"
            name: "pythonNode"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            options: {
                [mediapipe.PythonExecutorCalculatorOptions.ext]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/<FILENAME>.py"
                }
            }
        )";

        adjustConfigForTargetPlatform(testPbtxt);

        const std::string handlerPathToReplace{"<FILENAME>"};
        testPbtxt.replace(testPbtxt.find(handlerPathToReplace), handlerPathToReplace.size(), handlerPath);

        mediapipe::CalculatorRunner runner(testPbtxt);
        py::gil_scoped_acquire acquire;
        try {
            // Create side packets
            std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>> nodesResources = prepareInputSidePacket(handlerPath, getPythonBackend());
            addInputSidePacket("PYTHON_NODE_RESOURCES", nodesResources, 0, &runner);

            // Prepare inputs
            std::string datatype = "FP32";
            std::string inputName = "INPUT";
            int numElements = 3;
            float input1[] = {1.0, 1.0, 1.0};
            SimpleTensor<float> tensor1 = SimpleTensor<float>::createTensor(inputName, input1, datatype, numElements);
            addInputItem(inputName, tensor1.pyTensor, 0, &runner);

            // Run calculator
            {
                py::gil_scoped_release release;
                auto status = runner.Run();
                ASSERT_TRUE(absl::IsInternal(status));
                std::string expectedMessage = testCase.second;
                ASSERT_TRUE(status.message().find(expectedMessage) != std::string::npos);
            }
        } catch (const pybind11::error_already_set& e) {
            ASSERT_EQ(1, 0) << e.what();
        }
    }
}

TEST_F(PythonFlowTest, ConverterCalculator_OvTensorDimensionSizeExceeded) {
    std::string testPbtxt = R"(
        calculator: "PyTensorOvTensorConverterCalculator"
        name: "conversionNode"
        input_stream: "OVTENSOR:input"
        output_stream: "OVMS_PY_TENSOR:output"
        node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                        key: "OVMS_PY_TENSOR"
                        value: "input"
                    }
                }
            }
    )";

    mediapipe::CalculatorRunner runner(testPbtxt);

    void* ptr = (void*)1;  // Pointer != NULL allows creating fake ov::Tensor
    int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    uint64_t badDimension = static_cast<uint64_t>(maxInt64) + 1;

    std::unique_ptr<ov::Tensor> tensor = std::make_unique<ov::Tensor>(ov::element::u8, ov::Shape{1, badDimension}, ptr);
    runner.MutableInputs()->Tag("OVTENSOR").packets.push_back(mediapipe::Adopt<ov::Tensor>(tensor.release()).At(mediapipe::Timestamp(0)));

    py::gil_scoped_acquire acquire;
    try {
        py::gil_scoped_release release;
        auto status = runner.Run();
        ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.code() << " " << status.message();
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << e.what();
    }
}

TEST_F(PythonFlowTest, ConverterCalculator_UnsupportedOvTensorTypes) {
    std::string testPbtxt = R"(
        calculator: "PyTensorOvTensorConverterCalculator"
        name: "conversionNode"
        input_stream: "OVTENSOR:input"
        output_stream: "OVMS_PY_TENSOR:output"
        node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                        key: "OVMS_PY_TENSOR"
                        value: "input"
                    }
                }
            }
    )";

    // Cannot create ov::tensor with those type: Precision::BIN, Precision::CUSTOM, Precision::MIXED, Precision::Q78, Precision::DYNAMIC, Precision::UNDEFINED
    static std::vector<Precision> unsupportedPrecision = {Precision::I4, Precision::U4, Precision::U1};

    for (Precision ovmsPrecision : unsupportedPrecision) {
        mediapipe::CalculatorRunner runner(testPbtxt);

        std::cout << "Testing precision: " << toString(ovmsPrecision) << std::endl;
        std::unique_ptr<ov::Tensor> tensor = std::make_unique<ov::Tensor>(ovmsPrecisionToIE2Precision(ovmsPrecision), ov::Shape{1, 1});
        runner.MutableInputs()->Tag("OVTENSOR").packets.push_back(mediapipe::Adopt<ov::Tensor>(tensor.release()).At(mediapipe::Timestamp(0)));

        py::gil_scoped_acquire acquire;
        try {
            py::gil_scoped_release release;
            auto status = runner.Run();
            ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.code() << " " << status.message();
        } catch (const pybind11::error_already_set& e) {
            ASSERT_EQ(1, 0) << e.what();
        }
    }
}

TEST_F(PythonFlowTest, ConverterCalculator_PyTensorDimensionNegative) {
    std::string testPbtxt = R"(
        calculator: "PyTensorOvTensorConverterCalculator"
        name: "conversionNode"
        input_stream: "OVMS_PY_TENSOR:input"
        output_stream: "OVTENSOR:output"
    )";

    mediapipe::CalculatorRunner runner(testPbtxt);

    py::gil_scoped_acquire acquire;
    try {
        std::string datatype = "FP32";
        std::string name = "python_result";
        int numElements = 3;
        float input[] = {1.0, 2.0, 3.0};
        py::ssize_t badDimension = -numElements;
        std::unique_ptr<PyObjectWrapper<py::object>> pyTensor;
        getPythonBackend()->createOvmsPyTensor(name, (void*)input, std::vector<py::ssize_t>{1, badDimension}, datatype, numElements * sizeof(float), pyTensor);

        runner.MutableInputs()->Tag("OVMS_PY_TENSOR").packets.push_back(mediapipe::Adopt<PyObjectWrapper<py::object>>(pyTensor.release()).At(mediapipe::Timestamp(0)));

        py::gil_scoped_release release;
        auto status = runner.Run();
        ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.code() << " " << status.message();
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << e.what();
    }
}

TEST_F(PythonFlowTest, ConverterCalculator_PyTensorBufferMismatch) {
    std::string testPbtxt = R"(
        calculator: "PyTensorOvTensorConverterCalculator"
        name: "conversionNode"
        input_stream: "OVMS_PY_TENSOR:input"
        output_stream: "OVTENSOR:output"
    )";

    mediapipe::CalculatorRunner runner(testPbtxt);

    py::gil_scoped_acquire acquire;
    try {
        std::string datatype = "FP32";
        std::string name = "python_result";
        int numElements = 3;
        float input[] = {1.0, 2.0, 3.0};
        std::unique_ptr<PyObjectWrapper<py::object>> pyTensor;
        getPythonBackend()->createOvmsPyTensor(name, (void*)input, std::vector<py::ssize_t>{1, numElements}, datatype, numElements * sizeof(float) * 2 /*too large*/, pyTensor);

        runner.MutableInputs()->Tag("OVMS_PY_TENSOR").packets.push_back(mediapipe::Adopt<PyObjectWrapper<py::object>>(pyTensor.release()).At(mediapipe::Timestamp(0)));

        py::gil_scoped_release release;
        auto status = runner.Run();
        ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.code() << " " << status.message();
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << e.what();
    }
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutMultiRunWithErrors) {
    ConstructorEnabledModelManager manager;
    std::string firstTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(firstTestPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = firstTestPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;
    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);
    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");

    req.Clear();
    res.Clear();
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I32}, {}, false);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);

    req.Clear();
    res.Clear();
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);
    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, FinalizePassTest) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_finalize_pass.py"
                }
            }
        }
    )"};
    adjustConfigForTargetPlatform(pbTxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(), ""), StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, RelativeBasePath) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "relative_base_path.py"
                }
            }
        }
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(),
                  getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/python/scripts")),
        StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, RelativeBasePath2) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "python/scripts/relative_base_path.py"
                }
            }
        }
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(),
                  getGenericFullPathForSrcTest("/ovms/src/test/mediapipe")),
        StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, RelativeBasePath3) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "python/scripts/relative_base_path.py"
                }
            }
        }
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(),
                  getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/")),
        StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, RelativeHandlerPath) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "good_finalize_pass.py"
                }
            }
        }
    )"};
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(),
                  getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/python/scripts")),
        StatusCode::OK);

    ASSERT_EQ(nodeResources->handlerPath, getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/python/scripts/good_finalize_pass.py"));

    nodeResources->finalize();
}

TEST_F(PythonFlowTest, AbsoluteHandlerPath) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/relative_base_path.py"
                }
            }
        }
    )"};
    adjustConfigForTargetPlatform(pbTxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(), "this_string_doesnt_matter_since_handler_path_is_absolute"), StatusCode::OK);

    // Can't use getGenericFullPathForSrcTest due to mixed separators in the final path
    ASSERT_EQ(nodeResources->handlerPath, getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/python/scripts/relative_base_path.py"));
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, FinalizeMissingPassTest) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_finalize_pass.py"
                }
            }
        }
    )"};
    adjustConfigForTargetPlatform(pbTxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(), ""), StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, FinalizeDestructorRemoveFileTest) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_finalize_remove_file.py"
                }
            }
        }
    )"};
    adjustConfigForTargetPlatform(pbTxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::string path = getGenericFullPathForTmp("/tmp/pythonNodeTestRemoveFile.txt");
    {
        std::shared_ptr<PythonNodeResources> nodeResouce = nullptr;
        ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResouce, config.node(0), getPythonBackend(), ""), StatusCode::OK);

        ASSERT_TRUE(std::filesystem::exists(path));
        // nodeResources destructor calls finalize and removes the file
    }

    ASSERT_TRUE(!std::filesystem::exists(path));
}

TEST_F(PythonFlowTest, FinalizeException) {
    std::string pbTxt{R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_finalize_exception.py"
                }
            }
        }
    )"};
    adjustConfigForTargetPlatform(pbTxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend(), ""), StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, ReloadWithDifferentScriptName) {
    ConstructorEnabledModelManager manager;
    std::string firstTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(firstTestPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = firstTestPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");

    // ------- reload to a script with different name ----------

    std::string reloadedTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment_by_2.py"
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(reloadedTestPbtxt);
    mediapipeDummy.inputConfig = reloadedTestPbtxt;
    ASSERT_EQ(mediapipeDummy.reload(manager, mgc), StatusCode::OK);

    pipeline = nullptr;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    req.Clear();
    res.Clear();
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("output", data, req, res, 2 /* expect +2 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, FailingToInitializeOneNodeDestructsAllResources) {
    ConstructorEnabledModelManager manager;
    std::string firstTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:inter"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/good_finalize_remove_file.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:inter"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_throw_exception.py"
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(firstTestPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = firstTestPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
    ASSERT_EQ(mediapipeDummy.getPythonNodeResources("pythonNode1"), nullptr);
    ASSERT_EQ(mediapipeDummy.getPythonNodeResources("pythonNode2"), nullptr);
    ASSERT_EQ(mediapipeDummy.getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

// Negative Request Tests
// We cannot inherit from PythonFlowTest due to the fact of having 1 python interpreter
class PythonFlowSymmetricIncrementFixture {
private:
    ConstructorEnabledModelManager manager;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    std::shared_ptr<MediapipeGraphExecutor> pipeline;

public:
    PythonFlowSymmetricIncrementFixture(const std::string& scriptName = "symmetric_increment.py") {
        init(scriptName);  // This is required to allow using ASSERT inside constructor
    }

    void init(const std::string& scriptName) {
        std::string firstTestPbtxt = R"(
        input_stream: "OVMS_PY_TENSOR:input"
        output_stream: "OVMS_PY_TENSOR:output"
            node {
                name: "pythonNode"
                calculator: "PythonExecutorCalculator"
                input_side_packet: "PYTHON_NODE_RESOURCES:py"
                input_stream: "INPUT:input"
                output_stream: "OUTPUT:output"
                node_options: {
                    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                        handler_path: "/ovms/src/test/mediapipe/python/scripts/<REPLACE>"
                    }
                }
            }
        )";
        adjustConfigForTargetPlatform(firstTestPbtxt);
        const std::string replPhrase = "<REPLACE>";
        std::size_t pos = firstTestPbtxt.find(replPhrase);
        ASSERT_NE(pos, std::string::npos);
        firstTestPbtxt.replace(pos, replPhrase.length(), scriptName);

        ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
        mediapipeDummy = std::make_unique<DummyMediapipeGraphDefinition>("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
        mediapipeDummy->inputConfig = firstTestPbtxt;
        ASSERT_EQ(mediapipeDummy->validate(manager), StatusCode::OK);

        ASSERT_EQ(mediapipeDummy->create(pipeline), StatusCode::OK);
        ASSERT_NE(pipeline, nullptr);
    }
    std::shared_ptr<MediapipeGraphExecutor> getPipeline() {
        return pipeline;
    }
};

const std::vector<std::string> knownDatatypes{
    "BOOL", "UINT8", "UINT16", "UINT32", "UINT64", "INT8",
    "INT16", "INT32", "INT64", "FP16", "FP32", "FP64"};

TEST_F(PythonFlowTest, Negative_BufferTooSmall) {
    PythonFlowSymmetricIncrementFixture fixture;
    for (const std::string& datatype : knownDatatypes) {
        KFSRequest req;
        KFSResponse res;

        req.set_model_name("mediaDummy");
        prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 1}, datatype}, std::vector<float>{}, false);

        // Make the metadata larger than actual buffer
        auto& inputMeta = *req.mutable_inputs()->begin();
        inputMeta.clear_shape();
        inputMeta.add_shape(1);
        inputMeta.add_shape(1000000);
        inputMeta.add_shape(20);

        ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    }
}

TEST_F(PythonFlowTest, Negative_BufferTooLarge) {
    PythonFlowSymmetricIncrementFixture fixture;
    for (const std::string& datatype : knownDatatypes) {
        KFSRequest req;
        KFSResponse res;

        req.set_model_name("mediaDummy");
        prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 4}, datatype}, std::vector<float>{}, false);

        // Make the metadata smaller than actual buffer
        auto& inputMeta = *req.mutable_inputs()->begin();
        inputMeta.clear_shape();
        inputMeta.add_shape(1);
        inputMeta.add_shape(1);

        ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    }
}

// Metadata shape is ignored for custom types.
// The shape is inherited from buffer length, therefore it is always correct.
TEST_F(PythonFlowTest, Positive_BufferTooSmall_Custom) {
    PythonFlowSymmetricIncrementFixture fixture("symmetric_increment_by_2.py");
    KFSRequest req;
    KFSResponse res;

    req.set_model_name("mediaDummy");

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32 /*Overridden to "my custom type" below*/}, data, false);

    // Make the metadata larger than actual buffer
    auto& inputMeta = *req.mutable_inputs()->begin();
    inputMeta.clear_shape();
    inputMeta.add_shape(1);
    inputMeta.add_shape(1000000);
    inputMeta.add_shape(20);
    inputMeta.set_datatype("my custom type");

    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    constexpr const size_t dataLength = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);

    ASSERT_EQ(res.model_name(), "mediaDummy");
    ASSERT_EQ(res.outputs_size(), 1);
    ASSERT_EQ(res.raw_output_contents_size(), 1);
    // Finding the output with given name
    const auto& output_proto = *res.outputs().begin();
    ASSERT_EQ(output_proto.shape_size(), 1);
    ASSERT_EQ(output_proto.shape(0), dataLength);
    const auto* content = res.mutable_raw_output_contents(0);
    ASSERT_EQ(content->size(), dataLength);

    // The input data is treated as uint8 and each byte gets +2 addition.
    std::vector<uint8_t> expectedData(dataLength);
    std::memcpy(expectedData.data(), data.data(), dataLength);
    for (size_t i = 0; i < dataLength; i++) {
        expectedData[i] += 2;
    }

    EXPECT_EQ(0, std::memcmp(content->data(), expectedData.data(), dataLength))
        << readableError<uint8_t>(expectedData.data(), (unsigned char*)content->data(), dataLength);
}

// Metadata shape is ignored for custom types.
// The shape is inherited from buffer length, therefore it is always correct.
TEST_F(PythonFlowTest, Positive_BufferTooLarge_Custom) {
    PythonFlowSymmetricIncrementFixture fixture("symmetric_increment_by_2.py");
    KFSRequest req;
    KFSResponse res;

    req.set_model_name("mediaDummy");

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32 /*Overridden to "my custom type" below*/}, data, false);

    // Make the metadata smaller than actual buffer
    auto& inputMeta = *req.mutable_inputs()->begin();
    inputMeta.clear_shape();
    inputMeta.add_shape(1);
    inputMeta.add_shape(1);
    inputMeta.set_datatype("my custom type");

    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    constexpr const size_t dataLength = DUMMY_MODEL_OUTPUT_SIZE * sizeof(float);

    ASSERT_EQ(res.model_name(), "mediaDummy");
    ASSERT_EQ(res.outputs_size(), 1);
    ASSERT_EQ(res.raw_output_contents_size(), 1);
    // Finding the output with given name
    const auto& output_proto = *res.outputs().begin();
    ASSERT_EQ(output_proto.shape_size(), 1);
    ASSERT_EQ(output_proto.shape(0), dataLength);
    const auto* content = res.mutable_raw_output_contents(0);
    ASSERT_EQ(content->size(), dataLength);

    // The input data is treated as uint8 and each byte gets +2 addition.
    std::vector<uint8_t> expectedData(dataLength);
    std::memcpy(expectedData.data(), data.data(), dataLength);
    for (size_t i = 0; i < dataLength; i++) {
        expectedData[i] += 2;
    }

    EXPECT_EQ(0, std::memcmp(content->data(), expectedData.data(), dataLength))
        << readableError<uint8_t>(expectedData.data(), (unsigned char*)content->data(), dataLength);
}

TEST_F(PythonFlowTest, Negative_ExpectedBytesAmountOverflow) {
    PythonFlowSymmetricIncrementFixture fixture;
    KFSRequest req;
    KFSResponse res;
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 4}, "FP32"}, std::vector<float>{}, false);

    auto& inputMeta = *req.mutable_inputs()->begin();
    // Shape way over acceptable values
    inputMeta.clear_shape();
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    // Shape just above the size_t limit
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 5 + 1);
    inputMeta.add_shape(5);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    // Shape below size_t limit, but when multiplied by itemsize it overflows
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 4 + 1);
    inputMeta.add_shape(1);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PythonFlowTest, Negative_ExpectedBytesAmountOverflowTensorContent) {
    PythonFlowSymmetricIncrementFixture fixture;
    KFSRequest req;
    KFSResponse res;
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 4}, "FP32"}, std::vector<float>{}, true);

    auto& inputMeta = *req.mutable_inputs()->begin();
    // Shape way over acceptable values
    inputMeta.clear_shape();
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    // Shape just above the size_t limit
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 5 + 1);
    inputMeta.add_shape(5);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
    // Shape below size_t limit, but when multiplied by itemsize it overflows
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 4 + 1);
    inputMeta.add_shape(1);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext), StatusCode::INVALID_CONTENT_SIZE);
}

TEST_F(PythonFlowTest, Negative_NodeProducesUnexpectedTensor) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input"
            output_stream: "OUTPUT:abcd"  # symmetric_increment.py produces Tensor called "output" which is unexpected
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:abcd"
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    // Mediapipe failed to execute. Unexpected Tensor found in the outputs.
    // Script produced tensor called "abcd", but "output" was expected.
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(PythonFlowTest, Negative_NodeFiresProcessWithoutAllInputs) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input1"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input1"
            output_stream: "OUTPUT1:output1"
            output_stream: "OUTPUT2:output2"  # symmetric_increment.py will not produce output2 tensor
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT1:output1"
            input_stream: "INPUT2:output2" # this input will never arrive as pythonNode1 does not produce it
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_execute_read_more_than_one_input.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input1", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    // pythonNode2 will run with incomple inputs, but the handler script expects full set of inputs
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(PythonFlowTest, Positive_NodeFiresProcessWithoutAllInputs) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:input1"
    output_stream: "OVMS_PY_TENSOR:output"
        node {
            name: "pythonNode1"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:input1"
            output_stream: "OUTPUT1:output1"
            output_stream: "OUTPUT2:output2"  # symmetric_increment.py will not produce output2 tensor
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT1:output1"
            input_stream: "INPUT2:output2" # this input will never arrive as pythonNode1 does not produce it
            output_stream: "OUTPUT:output"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input1", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);
    checkDummyResponse("output", data, req, res, 2 /* expect +2 */, 1, "mediaDummy");
}

void setUpConverterPrecisionTest(std::shared_ptr<MediapipeGraphExecutor>& pipeline, std::unique_ptr<DummyMediapipeGraphDefinition>& mediapipeDummy) {
    ConstructorEnabledModelManager manager{"", getPythonBackend()};
    std::string testPbtxt = R"(
    input_stream: "OVTENSOR:in"
    output_stream: "OVTENSOR:out"
        node {
            name: "pythonNode1"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVTENSOR:in"
            output_stream: "OVMS_PY_TENSOR:output1"
            node_options: {
                [type.googleapis.com / mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
                    tag_to_output_tensor_names {
                    key: "OVMS_PY_TENSOR"
                    value: "output1"
                    }
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:output1"
            output_stream: "OUTPUT:output2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/single_io_increment.py"
                }
            }
        }
        node {
            name: "pythonNode3"
            calculator: "PyTensorOvTensorConverterCalculator"
            input_stream: "OVMS_PY_TENSOR:output2"
            output_stream: "OVTENSOR:out"
        }
    )";

    adjustConfigForTargetPlatform(testPbtxt);

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    mediapipeDummy = std::make_unique<DummyMediapipeGraphDefinition>("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy->inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy->validate(manager), StatusCode::OK);

    ASSERT_EQ(mediapipeDummy->create(pipeline), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);
}

TEST_F(PythonFlowTest, PythonCalculatorTest_INT8) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<int8_t> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I8}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_UINT8) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<uint8_t> data{1, 20, 3, 1, 20, 3, 1, 20, 3, 5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::U8}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_INT16) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<int16_t> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I16}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_UINT16) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<uint16_t> data{1, 20, 3, 1, 20, 3, 1, 20, 3, 5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::U16}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_INT32) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<int32_t> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_INT64) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<int64_t> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::I64}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_FP32) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTest_FP64) {
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    std::unique_ptr<DummyMediapipeGraphDefinition> mediapipeDummy;
    setUpConverterPrecisionTest(pipeline, mediapipeDummy);

    KFSRequest req;
    KFSResponse res;

    const std::vector<double> data{1, 20, 3, -1, 20, 3, 1, 20, 3, -5};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP64}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext), StatusCode::OK);  //, smr), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}
