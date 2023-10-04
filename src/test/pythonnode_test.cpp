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
#include <pybind11/embed.h>

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/pythonnoderesource.hpp"
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

namespace py = pybind11;
using namespace ovms;
using namespace py::literals;

using testing::HasSubstr;
using testing::Not;

class PythonFlowTest : public ::testing::TestWithParam<std::string> {
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

class DummyMediapipeGraphDefinition : public MediapipeGraphDefinition {
public:
    std::string inputConfig;

    PythonNodeResource* getPythonNodeResource(const std::string& nodeName) {
        auto it = this->pythonNodeResources.find(nodeName);
        if (it == std::end(pythonNodeResources)) {
            return nullptr;
        } else {
            return it->second.get();
        }
    }

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

class MediapipeFlowPythonNodeTest : public PythonFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/python/mediapipe_add_python_node.json");
    }
};

TEST_F(MediapipeFlowPythonNodeTest, InitializationPass) {
    ModelManager* manager;
    manager = &(dynamic_cast<const ovms::ServableManagerModule*>(server.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    auto graphDefinition = manager->getMediapipeFactory().findDefinitionByName("mediapipePythonBackend");
    ASSERT_NE(graphDefinition, nullptr);
    EXPECT_TRUE(graphDefinition->getStatus().isAvailable());
}

class MediapipePythonNodeTest : public ::testing::Test {
};

TEST_F(MediapipePythonNodeTest, PythonNodeFileDoesNotExist) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/22script2.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST);
}

TEST_F(MediapipePythonNodeTest, PythonNodeNameAlreadyExist) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/script.py"
                }
            }
        }
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out3"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/script2.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_NAME_ALREADY_EXISTS);
}

TEST_F(MediapipePythonNodeTest, PythonNodeInitFailed) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/fail_script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(MediapipePythonNodeTest, PythonNodeReturnFalse) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/return_false_script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(MediapipePythonNodeTest, PythonNodeInitException) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/exception_script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(MediapipePythonNodeTest, PythonNodeOptionsMissing) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_MISSING_OPTIONS);
}

TEST_F(MediapipePythonNodeTest, PythonNodeNameMissing) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/fail_script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_MISSING_NAME);
}

TEST_F(MediapipePythonNodeTest, PythonNodeNameDoesNotExist) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.getPythonNodeResource("pythonNode4"), nullptr);
}

TEST_F(MediapipePythonNodeTest, PythonNodeInitMembers) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/script.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResource* nodeRes = mediapipeDummy.getPythonNodeResource("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    try {
        py::gil_scoped_acquire acquire;
        using namespace py::literals;
        py::module_ sys = py::module_::import("sys");

        // Casting and recasting needed for ASSER_EQ to work
        std::string sModelMame = nodeRes->nodeResourceObject.get()->attr("model_name").cast<std::string>();
        std::string sExpectedName = py::str("testModel").cast<std::string>();

        ASSERT_EQ(sModelMame, sExpectedName);
        py::int_ executionTime = nodeRes->nodeResourceObject.get()->attr("execution_time");
        ASSERT_EQ(executionTime, 300);
        py::list modelInputs = nodeRes->nodeResourceObject.get()->attr("model_inputs");

        py::list expectedInputs = py::list();
        expectedInputs.attr("append")(py::str("input1"));
        expectedInputs.attr("append")(py::str("input2"));

        for (pybind11::size_t i = 0; i < modelInputs.size(); i++) {
            py::str inputName = py::cast<py::str>(modelInputs[i]);
            ASSERT_EQ(inputName.cast<std::string>(), expectedInputs[i].cast<std::string>());
        }
    } catch (const std::exception& e) {
        std::cout << "Python pybind exception: " << e.what() << std::endl;
        ASSERT_EQ(1, 0);
    } catch (...) {
        std::cout << "Python pybind exception: " << std::endl;
        ASSERT_EQ(1, 0);
    }
}

TEST_F(MediapipePythonNodeTest, PythonNodePassArgumentsToConstructor) {
    // Must be here - does not work when added to test::SetUp
    // Initialize Python interpreter
    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "in"
    output_stream: "out"
        node {
            name: "pythonNode2"
            calculator: "PythonBackendCalculator"
            input_side_packet: "PYOBJECT:pyobject"
            input_stream: "in"
            output_stream: "out2"
            node_options: {
                [type.googleapis.com / mediapipe.PythonBackendCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/script2.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResource* nodeRes = mediapipeDummy.getPythonNodeResource("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    try {
        py::gil_scoped_acquire acquire;
        using namespace py::literals;
        py::module_ sys = py::module_::import("sys");

        // Casting and recasting needed for ASSER_EQ to work
        py::dict modelOutputs = nodeRes->nodeResourceObject.get()->attr("model_outputs");
        py::int_ size = modelOutputs.size();
        ASSERT_EQ(size, 0);
    } catch (const std::exception& e) {
        std::cout << "Python pybind exception: " << e.what() << std::endl;
        ASSERT_EQ(1, 0);
    } catch (...) {
        std::cout << "Python pybind exception: " << std::endl;
        ASSERT_EQ(1, 0);
    }
}
