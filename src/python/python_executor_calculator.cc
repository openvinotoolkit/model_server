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
#include <unordered_map>

#include "pythonnoderesources.hpp"
#include "src/python/ovms_py_tensor.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 6001 6385 6386 6326 6011 6246 4456)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

#include "python_backend.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

namespace mediapipe {

const std::string PYTHON_SESSION_SIDE_PACKET_TAG = "PYTHON_NODE_RESOURCES";

class PythonExecutorCalculator : public CalculatorBase {
    std::shared_ptr<PythonNodeResources> nodeResources;
    std::unique_ptr<PyObjectWrapper<py::iterator>> pyIteratorPtr;
    bool hasLoopback{false};
    // The calculator manages timestamp for outputs to work independently of inputs
    // this way we can support timestamp continuity for more than one request in streaming scenario.
    mediapipe::Timestamp outputTimestamp;

    static void setInputsAndOutputsPacketTypes(CalculatorContract* cc) {
        for (const std::string& tag : cc->Inputs().GetTags()) {
            if (tag == "LOOPBACK") {
                cc->Inputs().Tag(tag).Set<bool>();
            } else {
                cc->Inputs().Tag(tag).Set<PyObjectWrapper<py::object>>();
            }
        }

        for (const std::string& tag : cc->Outputs().GetTags()) {
            if (tag == "LOOPBACK") {
                cc->Outputs().Tag(tag).Set<bool>();
            } else {
                cc->Outputs().Tag(tag).Set<PyObjectWrapper<py::object>>();
            }
        }
    }

    void validateInputTensor(const py::object& pyInput) {
        try {
            nodeResources->pythonBackend->validateOvmsPyTensor(pyInput);
        } catch (UnexpectedPythonObjectError& e) {
            throw UnexpectedInputPythonObjectError(e);
        }
    }

    void prepareInputs(CalculatorContext* cc, std::vector<py::object>* pyInputs) {
        for (const std::string& tag : cc->Inputs().GetTags()) {
            if (tag != "LOOPBACK") {
                if (cc->Inputs().Tag(tag).IsEmpty()) {
                    LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Received empty packet on input: " << tag
                              << ". Execution will continue without that input.";
                    continue;
                }
                const py::object& pyInput = cc->Inputs().Tag(tag).Get<PyObjectWrapper<py::object>>().getObject();
                validateInputTensor(pyInput);
                pyInputs->push_back(pyInput);
            }
        }
    }

    void validateOutputTensor(const py::object& pyOutput) {
        try {
            nodeResources->pythonBackend->validateOvmsPyTensor(pyOutput);
        } catch (UnexpectedPythonObjectError& e) {
            throw UnexpectedOutputPythonObjectError(e);
        }
    }

    void pushOutputs(CalculatorContext* cc, py::list pyOutputs, mediapipe::Timestamp& timestamp, bool pushLoopback) {
        py::gil_scoped_acquire acquire;
        for (py::handle pyOutputHandle : pyOutputs) {
            py::object pyOutput = pyOutputHandle.cast<py::object>();
            validateOutputTensor(pyOutput);
            std::string outputName = pyOutput.attr("name").cast<std::string>();

            auto it = nodeResources->outputsNameTagMapping.find(outputName);
            if (it == nodeResources->outputsNameTagMapping.end()) {
                throw UnexpectedOutputTensorError(outputName);
            }

            std::string outputTag = it->second;
            if (cc->Outputs().HasTag(outputTag)) {
                std::unique_ptr<PyObjectWrapper<py::object>> outputPtr = std::make_unique<PyObjectWrapper<py::object>>(pyOutput);
                cc->Outputs().Tag(outputTag).Add(outputPtr.release(), timestamp);
            }
        }
        if (pushLoopback) {
            timestamp++;
            cc->Outputs().Tag("LOOPBACK").Add(std::make_unique<bool>(true).release(), timestamp);
        }
    }

    bool receivedNewData(CalculatorContext* cc) {
        for (const std::string& tag : cc->Inputs().GetTags()) {
            if (tag != "LOOPBACK") {
                if (!cc->Inputs().Tag(tag).IsEmpty())
                    return true;
            }
        }
        return false;
    }

    bool generatorInitialized() {
        return pyIteratorPtr != nullptr;
    }

    bool generatorFinished() {
        return pyIteratorPtr->getObject() == py::iterator::sentinel();
    }

    void generate(CalculatorContext* cc, mediapipe::Timestamp& timestamp) {
        py::list pyOutputs = py::cast<py::list>(*pyIteratorPtr->getObject());
        pushOutputs(cc, std::move(pyOutputs), timestamp, true);
        ++(pyIteratorPtr->getObject());  // increment iterator
    }

    void initializeGenerator(py::object generator) {
        pyIteratorPtr = std::make_unique<PyObjectWrapper<py::iterator>>(generator);
    }

    void resetGenerator() {
        pyIteratorPtr.reset();
    }

    void handleExecutionResult(CalculatorContext* cc, py::object executionResult) {
        if (py::isinstance<py::list>(executionResult)) {
            pushOutputs(cc, std::move(executionResult), outputTimestamp, false);
        } else if (py::isinstance<py::iterator>(executionResult)) {
            if (!hasLoopback)
                throw BadPythonNodeConfigurationError("Execute yielded, but LOOPBACK is not defined in the node");
            initializeGenerator(std::move(executionResult));
            generate(cc, outputTimestamp);
        } else {
            throw UnexpectedPythonObjectError(executionResult, "list or generator");
        }
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        if (cc->Inputs().HasTag("LOOPBACK") != cc->Outputs().HasTag("LOOPBACK"))
            return absl::Status(absl::StatusCode::kInvalidArgument, "If LOOPBACK is used, it must be defined on both input and output of the node");

        setInputsAndOutputsPacketTypes(cc);
        cc->InputSidePackets().Tag(PYTHON_SESSION_SIDE_PACKET_TAG).Set<PythonNodeResourcesMap>();
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open start";
        if (cc->Inputs().HasTag("LOOPBACK"))
            hasLoopback = true;

        PythonNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(PYTHON_SESSION_SIDE_PACKET_TAG).Get<PythonNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        if (it == nodeResourcesMap.end()) {
            LOG(INFO) << "Could not find initialized Python node named: " << cc->NodeName();
            RET_CHECK(false);
        }

        nodeResources = it->second;
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

#define RETURN_EXECUTION_FAILED_STATUS() \
    return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution")

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process start";
        py::gil_scoped_acquire acquire;
        try {
            if (generatorInitialized()) {
                if (receivedNewData(cc)) {
                    LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Node is already processing data. Create new stream for another request.";
                    return absl::Status(absl::StatusCode::kResourceExhausted, "Node is already processing data. Create new stream for another request.");
                }
                if (!generatorFinished()) {
                    generate(cc, outputTimestamp);
                } else {
                    LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] finished generating. Resetting the generator.";
                    resetGenerator();
                }
            } else {
                // If execute yields, first request sets initial timestamp to input timestamp, then each cycle increments it.
                // If execute returns, input timestamp is also output timestamp.

                outputTimestamp = cc->InputTimestamp();

                std::vector<py::object> pyInputs;
                prepareInputs(cc, &pyInputs);
                py::object executeResult = nodeResources->ovmsPythonModel->attr("execute")(pyInputs);
                handleExecutionResult(cc, executeResult);
            }
        } catch (const UnexpectedOutputTensorError& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (const UnexpectedOutputPythonObjectError& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution. Wrong object on execute output: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (const UnexpectedInputPythonObjectError& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution. Wrong object on execute input: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (const BadPythonNodeConfigurationError& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (const pybind11::error_already_set& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            RETURN_EXECUTION_FAILED_STATUS();
        }
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonExecutorCalculator);
}  // namespace mediapipe
