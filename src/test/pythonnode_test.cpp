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
#include "../mediapipe_internal/pythonnoderesource.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
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
It's launching along with the server and even most tests will not use the server, the interpreter remains initialized.
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
            (char*)port.c_str()};
        int argc = 5;
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
        MediapipeGraphDefinition(name, config, nullptr, nullptr, getPythonBackend()) {}

    // Do not read from path - use predefined config contents
    Status validateForConfigFileExistence() override {
        this->chosenConfig = this->inputConfig;
        return StatusCode::OK;
    }
};

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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED);
}

TEST_F(PythonFlowTest, PythonNodeReturnFalse) {
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
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/bad_initialize_return_false.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.getPythonNodeResource("pythonNode4"), nullptr);
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResource* nodeRes = mediapipeDummy.getPythonNodeResource("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    py::gil_scoped_acquire acquire;
    try {
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
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    PythonNodeResource* nodeRes = mediapipeDummy.getPythonNodeResource("pythonNode2");
    ASSERT_TRUE(nodeRes != nullptr);

    py::gil_scoped_acquire acquire;
    try {
        using namespace py::literals;
        py::module_ sys = py::module_::import("sys");

        // Casting and recasting needed for ASSER_EQ to work
        py::dict modelOutputs = nodeRes->nodeResourceObject.get()->attr("model_outputs");
        py::int_ size = modelOutputs.size();
        ASSERT_EQ(size, 0);
    } catch (const pybind11::error_already_set& e) {
        ASSERT_EQ(1, 0) << "Python pybind exception: " << e.what();
    } catch (...) {
        ASSERT_EQ(1, 0) << "General error";
    }
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
        getPythonBackend()->createOvmsPyTensor(tensor.name, (void*)tensor.data, tensor.shape, tensor.datatype, tensor.size, tensor.pyTensor);
        return tensor;
    }

    static std::vector<T> readVectorFromOutput(const std::string& outputName, int numElements, const mediapipe::CalculatorRunner* runner) {
        const PyObjectWrapper<py::object>& pyOutput = runner->Outputs().Tag(outputName).packets[0].Get<PyObjectWrapper<py::object>>();
        T* outputData = (T*)pyOutput.getProperty<void*>("ptr");
        std::vector<T> output;
        output.assign(outputData, outputData + numElements);
        return output;
    }

    std::vector<T> getIncrementedVector() {
        // SimpleTensor is expected to hold data in shape (1, X),
        // therefore we iterate over the second dimension as it holds the actual data
        std::vector<T> output;
        T* fpData = (T*)data;
        for (int i = 0; i < shape[1]; i++) {
            output.push_back(fpData[i] + 1);
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
        const std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>>& pythonNodeResources,
        PythonBackend* pythonBackend) :
        MediapipeGraphExecutor(name, version, config, inputTypes, outputTypes, inputNames, outputNames, pythonNodeResources, pythonBackend) {}
};

TEST_F(PythonFlowTest, SerializePyObjectWrapperToKServeResponse) {
    ovms::stream_types_mapping_t mapping;
    mapping["python_result"] = mediapipe_packet_type_enum::OVMS_PY_TENSOR;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;
    const ::mediapipe::CalculatorGraphConfig config;
    std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> pythonNodeResources;
    auto executor = MockedMediapipeGraphExecutorPy("", "", config, mapping, mapping, inputNames, outputNames, pythonNodeResources, getPythonBackend());

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

static void addInputSidePacket(std::string tag, std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>>& input,
    int64_t timestamp, mediapipe::CalculatorRunner* runner) {
    runner->MutableSidePackets()->Tag(tag) = mediapipe::MakePacket<std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>>>(input).At(mediapipe::Timestamp(timestamp));
}

static std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> prepareInputSidePacket(const std::string& handlerPath, PythonBackend* pythonBackend) {
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

    std::shared_ptr<PythonNodeResource> nodeResource = std::make_shared<PythonNodeResource>(pythonBackend);
    nodeResource->nodeResourceObject = std::make_unique<py::object>(pythonModel);

    std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> nodesResources{{"pythonNode", nodeResource}};
    return nodesResources;
}

TEST_F(PythonFlowTest, PythonCalculatorTestSingleInSingleOut) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("OUTPUT", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");
}

TEST_F(PythonFlowTest, PythonCalculatorTestMultiInMultiOut) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR1:in1"
    input_stream: "OVMS_PY_TENSOR2:in2"
    input_stream: "OVMS_PY_TENSOR3:in3"
    output_stream: "OVMS_PY_TENSOR1:out1"
    output_stream: "OVMS_PY_TENSOR2:out2"
    output_stream: "OVMS_PY_TENSOR3:out3"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT1:in1"
            input_stream: "INPUT2:in2"
            input_stream: "INPUT3:in3"
            output_stream: "OUTPUT1:out1"
            output_stream: "OUTPUT2:out2"
            output_stream: "OUTPUT3:out3"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt);
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
    prepareKFSInferInputTensor(req, "in1", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data1, false);
    prepareKFSInferInputTensor(req, "in2", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data2, false);
    prepareKFSInferInputTensor(req, "in3", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data3, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("OUTPUT1", data1, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("OUTPUT2", data2, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
    checkDummyResponse("OUTPUT3", data3, req, res, 1 /* expect +1 */, 1, "mediaDummy", 3);
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
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
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
            std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> nodesResources = prepareInputSidePacket(handlerPath, getPythonBackend());
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
        std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> nodesResources = prepareInputSidePacket(handlerPath, getPythonBackend());
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

    std::shared_ptr<PythonNodeResource> nodeResource = nullptr;
    ASSERT_EQ(PythonNodeResource::createPythonNodeResource(nodeResource, config.node(0).node_options(0), getPythonBackend()), StatusCode::OK);
    nodeResource->finalize();
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

    std::shared_ptr<PythonNodeResource> nodeResource = nullptr;
    ASSERT_EQ(PythonNodeResource::createPythonNodeResource(nodeResource, config.node(0).node_options(0), getPythonBackend()), StatusCode::OK);
    nodeResource->finalize();
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
        std::shared_ptr<PythonNodeResource> nodeResouce = nullptr;
        ASSERT_EQ(PythonNodeResource::createPythonNodeResource(nodeResouce, config.node(0).node_options(0), getPythonBackend()), StatusCode::OK);

        ASSERT_TRUE(std::filesystem::exists(path));
        // nodeResource destructor calls finalize and removes the file
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

    std::shared_ptr<PythonNodeResource> nodeResource = nullptr;
    ASSERT_EQ(PythonNodeResource::createPythonNodeResource(nodeResource, config.node(0).node_options(0), getPythonBackend()), StatusCode::OK);
    nodeResource->finalize();
}

TEST_F(PythonFlowTest, ReloadWithDifferentScriptName) {
    ConstructorEnabledModelManager manager;
    std::string firstTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
            node_options: {
                [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
                    handler_path: "/ovms/src/test/mediapipe/python/scripts/symmetric_increment.py"
                }
            }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, firstTestPbtxt);
    mediapipeDummy.inputConfig = firstTestPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);

    std::shared_ptr<MediapipeGraphExecutor> pipeline;
    ASSERT_EQ(mediapipeDummy.create(pipeline, nullptr, nullptr), StatusCode::OK);
    ASSERT_NE(pipeline, nullptr);

    KFSRequest req;
    KFSResponse res;

    const std::vector<float> data{1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, 1.0f, 20.0f, 3.0f, -5.0f};
    req.set_model_name("mediaDummy");
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data, false);

    ServableMetricReporter* smr{nullptr};
    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("OUTPUT", data, req, res, 1 /* expect +1 */, 1, "mediaDummy");

    // ------- reload to a script with different name ----------

    std::string reloadedTestPbtxt = R"(
    input_stream: "OVMS_PY_TENSOR:in"
    output_stream: "OVMS_PY_TENSOR:out"
        node {
            name: "pythonNode"
            calculator: "PythonExecutorCalculator"
            input_side_packet: "PYTHON_NODE_RESOURCES:py"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
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
    prepareKFSInferInputTensor(req, "in", std::tuple<ovms::signed_shape_t, const ovms::Precision>{{1, DUMMY_MODEL_OUTPUT_SIZE}, ovms::fromString("FP32")}, data, false);

    ASSERT_EQ(pipeline->infer(&req, &res, this->defaultExecutionContext, smr), StatusCode::OK);

    checkDummyResponse("OUTPUT", data, req, res, 2 /* expect +2 */, 1, "mediaDummy");
}
