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
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../python/pythonnoderesources.hpp"
#include "../pythoninterpretermodule.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "mediapipe/calculators/ovms/modelapiovmsadapter.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_runner.h"
#pragma GCC diagnostic pop

#include "opencv2/opencv.hpp"
#include "python/python_backend.hpp"
#include "pythoninterpretermodule.hpp"
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

class PythonFlowTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {
protected:
    ovms::ExecutionContext defaultExecutionContext{ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::Predict};

public:
    static void SetUpTestSuite() {
        std::string configPath = "/ovms/src/test/mediapipe/python/mediapipe_add_python_node.json";
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
        auto start = std::chrono::high_resolution_clock::now();
        while ((ovms::Server::instance().getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!ovms::Server::instance().isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
    }
    static void TearDownTestSuite() {
        ovms::Server::instance().setShutdownRequest(1);
        serverThread->join();
        ovms::Server::instance().setShutdownRequest(0);
        std::string path = std::string("/tmp/pythonNodeTestRemoveFile.txt");
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
    std::string path = std::string("/tmp/pythonNodeTestRemoveFile.txt");
    manager = &(dynamic_cast<const ovms::ServableManagerModule*>(ovms::Server::instance().getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager());
    auto graphDefinition = manager->getMediapipeFactory().findDefinitionByName("mediapipePythonBackend");
    ASSERT_NE(graphDefinition, nullptr);
    EXPECT_TRUE(graphDefinition->getStatus().isAvailable());
    ASSERT_TRUE(std::filesystem::exists(path));
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

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST);
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
        py::int_ executionTime = nodeRes->ovmsPythonModel.get()->attr("execution_time");
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

TEST_F(PythonFlowTest, PythonNodeLoopbackDefinedOnlyOnOutput) {
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
            output_stream: "LOOPBACK:loopback"
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR);
}
// TODO: Add test with only input LOOPBACK
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
        getPythonBackend()->createOvmsPyTensor(tensor.name, (void*)tensor.data, tensor.shape, tensor.datatype, tensor.size, tensor.pyTensor);
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
    Status serializePacket(const std::string& name, ::inference::ModelInferResponse& response, const ::mediapipe::Packet& packet) const {
        return ovms::MediapipeGraphExecutor::serializePacket(name, response, packet);
    }

    MockedMediapipeGraphExecutorPy(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
        stream_types_mapping_t inputTypes,
        stream_types_mapping_t outputTypes,
        std::vector<std::string> inputNames, std::vector<std::string> outputNames,
        const PythonNodeResourcesMap& pythonNodeResourcesMap,
        PythonBackend* pythonBackend) :
        MediapipeGraphExecutor(name, version, config, inputTypes, outputTypes, inputNames, outputNames, pythonNodeResourcesMap, pythonBackend) {}
};

TEST_F(PythonFlowTest, SerializePyObjectWrapperToKServeResponse) {
    ovms::stream_types_mapping_t mapping;
    mapping["python_result"] = mediapipe_packet_type_enum::OVMS_PY_TENSOR;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;
    const ::mediapipe::CalculatorGraphConfig config;
    PythonNodeResourcesMap pythonNodeResourcesMap;
    auto executor = MockedMediapipeGraphExecutorPy("", "", config, mapping, mapping, inputNames, outputNames, pythonNodeResourcesMap, getPythonBackend());

    std::string datatype = "FP32";
    std::string name = "python_result";
    int numElements = 3;
    float input[] = {1.0, 2.0, 3.0};
    SimpleTensor<float> tensor = SimpleTensor<float>::createTensor(name, input, datatype, numElements);

    ::inference::ModelInferResponse response;

    ::mediapipe::Packet packet = ::mediapipe::Adopt<PyObjectWrapper<py::object>>(tensor.pyTensor.release());
    ASSERT_EQ(executor.serializePacket(name, response, packet), StatusCode::OK);
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

static void clearInputStream(std::string tag, mediapipe::CalculatorRunner* runner) {
    runner->MutableInputs()->Tag(tag).packets.clear();
}

static void addInputSidePacket(std::string tag, std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>& input,
    int64_t timestamp, mediapipe::CalculatorRunner* runner) {
    runner->MutableSidePackets()->Tag(tag) = mediapipe::MakePacket<std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>>(input).At(mediapipe::Timestamp(timestamp));
}

static std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>> prepareInputSidePacket(const std::string& handlerPath, PythonBackend* pythonBackend) {
    // Create side packets
    auto fsHandlerPath = std::filesystem::path(handlerPath);
    fsHandlerPath.replace_extension();

    std::string parentPath = fsHandlerPath.parent_path();
    std::string filename = fsHandlerPath.filename();

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("output", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    auto it = res.outputs().begin();
    const auto& output_proto = *it;
    ASSERT_EQ(output_proto.datatype(), "9w");  // "9w is memoryview format of numpy array containing single string element: 'my string'"
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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "first", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("out", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("out1", data, req, res, 1 /* expect +1 */, 1, "mediaDummy", 2);
    checkDummyResponse("out2", data, req, res, 2 /* expect +2 */, 1, "mediaDummy", 2);
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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
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

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);
    checkDummyResponse("output1", data1, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("output2", data2, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("output3", data3, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
}

static void setupTestPipeline(std::shared_ptr<MediapipeGraphExecutor>& pipeline) {
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
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);
}

TEST_F(PythonFlowTest, PythonCalculatorScalarNoShape) {
    KFSRequest req;
    KFSResponse res;

    float inputScalar = 6.0;
    const std::vector<float> data{inputScalar};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{ovms::signed_shape_t{}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    setupTestPipeline(pipeline);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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

    ServableMetricReporter* smr{nullptr};
    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    setupTestPipeline(pipeline);
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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
        {"bad_execute_wrong_signature", "Error occurred during Python code execution"},
        {"bad_execute_illegal_operation", "Error occurred during Python code execution"},
        {"bad_execute_import_error", "Error occurred during Python code execution"},
        {"bad_execute_wrong_return_value", "Python execute function received or returned bad value"}};

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

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOutMultiRunWithErrors) {
    std::string testPbtxt = R"(
        calculator: "PythonExecutorCalculator"
        name: "pythonNode"
        input_side_packet: "PYTHON_NODE_RESOURCES:py"
        input_stream: "INPUT:in"
        output_stream: "OUTPUT:out"
        options: {
            [mediapipe.PythonExecutorCalculatorOptions.ext]: {
                handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_identity_fp32_only.py"
            }
        }
    )";

    mediapipe::CalculatorRunner runner(testPbtxt);
    py::gil_scoped_acquire acquire;
    try {
        std::string handlerPath = "/ovms/src/test/mediapipe/python/scripts/symmetric_identity_fp32_only.py";
        // Create side packets
        std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>> nodesResources = prepareInputSidePacket(handlerPath, getPythonBackend());
        addInputSidePacket("PYTHON_NODE_RESOURCES", nodesResources, 0, &runner);

        std::string inputName = "INPUT";
        int numElements = 3;

        // Prepare good inputs
        float input1[] = {1.0, 1.0, 1.0};
        SimpleTensor<float> tensor1 = SimpleTensor<float>::createTensor(inputName, input1, "FP32", numElements);
        addInputItem(inputName, tensor1.pyTensor, 0, &runner);

        // Run calculator
        {
            py::gil_scoped_release release;
            ASSERT_EQ(runner.Run(), absl::OkStatus());
            clearInputStream(inputName, &runner);
        }

        // Prepare bad inputs
        int input2[] = {2, 2, 2};
        SimpleTensor<int> tensor2 = SimpleTensor<int>::createTensor(inputName, input2, "INT32", numElements);
        addInputItem(inputName, tensor2.pyTensor, 1, &runner);

        // Run calculator
        {
            py::gil_scoped_release release;
            auto status = runner.Run();
            ASSERT_TRUE(absl::IsInternal(status));
            std::string expectedMessage = "Error occurred during Python code execution";
            ASSERT_TRUE(status.message().find(expectedMessage) != std::string::npos);
            clearInputStream(inputName, &runner);
        }

        // Prepare good inputs
        float input3[] = {3.0, 3.0, 3.0};
        SimpleTensor<float> tensor3 = SimpleTensor<float>::createTensor(inputName, input3, "FP32", numElements);
        addInputItem(inputName, tensor3.pyTensor, 2, &runner);

        // Run calculator
        {
            py::gil_scoped_release release;
            ASSERT_EQ(runner.Run(), absl::OkStatus());
            clearInputStream(inputName, &runner);
        }
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << e.what();
    }
}

/* TODO: 
    - bad input stream element (py::object that is not pyovms.Tensor)
    - bad output stream element (py::object that is not pyovms.Tensor)
*/

TEST_F(PythonFlowTest, FinalizePassTest) {
    const std::string pbTxt{R"(
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
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend()), StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, FinalizeMissingPassTest) {
    const std::string pbTxt{R"(
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
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend()), StatusCode::OK);
    nodeResources->finalize();
}

TEST_F(PythonFlowTest, FinalizeDestructorRemoveFileTest) {
    const std::string pbTxt{R"(
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
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::string path = std::string("/tmp/pythonNodeTestRemoveFile.txt");
    {
        std::shared_ptr<PythonNodeResources> nodeResouce = nullptr;
        ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResouce, config.node(0), getPythonBackend()), StatusCode::OK);

        ASSERT_TRUE(std::filesystem::exists(path));
        // nodeResources destructor calls finalize and removes the file
    }

    ASSERT_TRUE(!std::filesystem::exists(path));
}

TEST_F(PythonFlowTest, FinalizeException) {
    const std::string pbTxt{R"(
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
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(pbTxt, &config));

    std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
    ASSERT_EQ(PythonNodeResources::createPythonNodeResources(nodeResources, config.node(0), getPythonBackend()), StatusCode::OK);
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

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
    mediapipeDummy.inputConfig = firstTestPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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

    mediapipeDummy.inputConfig = reloadedTestPbtxt;
    ASSERT_EQ(mediapipeDummy.reload(manager, mgc), StatusCode::OK);

    pipeline = nullptr;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    req.Clear();
    res.Clear();
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

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

        const std::string replPhrase = "<REPLACE>";
        std::size_t pos = firstTestPbtxt.find(replPhrase);
        ASSERT_NE(pos, std::string::npos);
        firstTestPbtxt.replace(pos, replPhrase.length(), scriptName);

        ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
        mediapipeDummy = std::make_unique<DummyMediapipeGraphDefinition>("mediaDummy", mgc, firstTestPbtxt, getPythonBackend());
        mediapipeDummy->inputConfig = firstTestPbtxt;
        ASSERT_EQ(mediapipeDummy->validate(manager), StatusCode::OK);

        ASSERT_EQ(mediapipeDummy->create(pipeline, nullptr, nullptr), StatusCode::OK);
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
        prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 1}, datatype}, {}, false);

        // Make the metdata larger than actual buffer
        auto& inputMeta = *req.mutable_inputs()->begin();
        inputMeta.clear_shape();
        inputMeta.add_shape(1);
        inputMeta.add_shape(1000000);
        inputMeta.add_shape(20);

        ServableMetricReporter* defaultReporter{nullptr};
        ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::INVALID_CONTENT_SIZE);
    }
}

TEST_F(PythonFlowTest, Negative_BufferTooLarge) {
    PythonFlowSymmetricIncrementFixture fixture;
    for (const std::string& datatype : knownDatatypes) {
        KFSRequest req;
        KFSResponse res;

        req.set_model_name("mediaDummy");
        prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 4}, datatype}, {}, false);

        // Make the metdata smaller than actual buffer
        auto& inputMeta = *req.mutable_inputs()->begin();
        inputMeta.clear_shape();
        inputMeta.add_shape(1);
        inputMeta.add_shape(1);

        ServableMetricReporter* defaultReporter{nullptr};
        ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::INVALID_CONTENT_SIZE);
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
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32 /*Overriden below*/}, data, false);

    // Make the metdata larger than actual buffer
    auto& inputMeta = *req.mutable_inputs()->begin();
    inputMeta.clear_shape();
    inputMeta.add_shape(1);
    inputMeta.add_shape(1000000);
    inputMeta.add_shape(20);
    inputMeta.set_datatype("my custom type");

    ServableMetricReporter* defaultReporter{nullptr};
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::OK);

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
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::Precision::FP32 /*Overriden below*/}, data, false);

    // Make the metdata smaller than actual buffer
    auto& inputMeta = *req.mutable_inputs()->begin();
    inputMeta.clear_shape();
    inputMeta.add_shape(1);
    inputMeta.add_shape(1);
    inputMeta.set_datatype("my custom type");

    ServableMetricReporter* defaultReporter{nullptr};
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::OK);

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
    prepareKFSInferInputTensor(req, "input", std::tuple<ovms::signed_shape_t, const std::string>{{1, 4}, "FP32"}, {}, false);

    ServableMetricReporter* defaultReporter{nullptr};
    auto& inputMeta = *req.mutable_inputs()->begin();
    // Shape way over acceptable values
    inputMeta.clear_shape();
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    inputMeta.add_shape(10000000);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::INVALID_CONTENT_SIZE);
    // Shape just above the size_t limit
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 5 + 1);
    inputMeta.add_shape(5);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::INVALID_CONTENT_SIZE);
    // Shape below size_t limit, but when multiplied by itemsize it overflows
    inputMeta.clear_shape();
    inputMeta.add_shape(std::numeric_limits<size_t>::max() / 4 + 1);
    inputMeta.add_shape(1);
    ASSERT_EQ(fixture.getPipeline()->infer(&req, &res, this->defaultExecutionContext, defaultReporter), StatusCode::INVALID_CONTENT_SIZE);
}
