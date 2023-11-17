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

#include "../mediapipe_internal/pythonnoderesource.hpp"
#include "../python/ovms_py_tensor.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "../python/python_backend.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

namespace mediapipe {

const std::string INPUT_SIDE_PACKET_TAG = "PYTHON_NODE_RESOURCES";

class PythonExecutorCalculator : public CalculatorBase {
    /* 
    TODO: Streaming support:
        - timestamping
        - loopback input
    */
    std::shared_ptr<PythonNodeResource> nodeResources;
    std::unique_ptr<PyObjectWrapper<py::iterator>> pyIteratorPtr;

    void pushOutputs(CalculatorContext* cc, py::list pyOutputs) {
        py::gil_scoped_acquire acquire;
        for (py::handle pyOutputHandle : pyOutputs) {
            py::object pyOutput = pyOutputHandle.cast<py::object>();
            nodeResources->pythonBackend->validateOvmsPyTensor(pyOutput);
            std::string outputName = pyOutput.attr("name").cast<std::string>();
            if (cc->Outputs().HasTag(outputName)) {
                std::unique_ptr<PyObjectWrapper<py::object>> outputPtr = std::make_unique<PyObjectWrapper<py::object>>(pyOutput);
                cc->Outputs().Tag(outputName).Add(outputPtr.release(), cc->InputTimestamp());
            }
        }
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (auto& input : cc->Inputs()) {
            input.Set<PyObjectWrapper<py::object>>();
        }
        for (auto& output : cc->Outputs()) {
            output.Set<PyObjectWrapper<py::object>>();
        }
        cc->InputSidePackets().Tag(INPUT_SIDE_PACKET_TAG).Set<PythonNodesResources>();
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open start";
        PythonNodesResources nodesResourcesPtr = cc->InputSidePackets().Tag(INPUT_SIDE_PACKET_TAG).Get<PythonNodesResources>();
        auto it = nodesResourcesPtr.find(cc->NodeName());
        if (it == nodesResourcesPtr.end()) {
            LOG(INFO) << "Could not find initialized Python node named: " << cc->NodeName();
            RET_CHECK(false);
        }
        nodeResources = it->second;
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process start";
        py::gil_scoped_acquire acquire;
        try {
            py::print("PYTHON: Acquired GIL");
            if (pyIteratorPtr) {  // Iterator initialized
                py::print("PyIterator initialized block");
                if (pyIteratorPtr->getObject() != py::iterator::sentinel()) {
                    py::list pyOutputs = py::cast<py::list>(*pyIteratorPtr->getObject());
                    pushOutputs(cc, pyOutputs);
                    ++(pyIteratorPtr->getObject());
                } else {
                    LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] finished generating. Reseting the iterator.";
                    pyIteratorPtr.reset();
                }
            } else {  // Iterator not initialized, either first iteration or execute is not yielding
                py::print("PyIterator uninitialized block");
                std::vector<py::object> pyInputs;
                for (const std::string& tag : cc->Inputs().GetTags()) {
                    const py::object& pyInput = cc->Inputs().Tag(tag).Get<PyObjectWrapper<py::object>>().getObject();
                    nodeResources->pythonBackend->validateOvmsPyTensor(pyInput);
                    pyInputs.push_back(pyInput);
                }

                py::object executeResult = std::move(nodeResources->nodeResourceObject->attr("execute")(pyInputs));
                py::print(executeResult.attr("__class__").attr("__name__"));

                if (py::isinstance<py::list>(executeResult)) {
                    py::print("Regular execution (execute returned)");
                    pushOutputs(cc, executeResult);
                } else if (py::isinstance<py::iterator>(executeResult)) {
                    py::print("Iterator initialization (execute yielded)");
                    pyIteratorPtr = std::make_unique<PyObjectWrapper<py::iterator>>(executeResult);
                    py::list pyOutputs = py::cast<py::list>(*pyIteratorPtr->getObject());
                    pushOutputs(cc, pyOutputs);
                    ++(pyIteratorPtr->getObject());
                } else {
                    throw UnexpectedPythonObjectError(executeResult, "list or generator");
                }
            }
            py::print("PYTHON: Released GIL");
        } catch (const UnexpectedPythonObjectError& e) {
            // TODO: maybe some more descriptive information where to seek the issue.
            LOG(INFO) << "Wrong object on node " << cc->NodeName() << " execute input or output: " << e.what();
            return absl::Status(absl::StatusCode::kInternal, "Python execute function received or returned bad value");
        } catch (const pybind11::error_already_set& e) {
            LOG(INFO) << "Python error occurred during node " << cc->NodeName() << " execution: " << e.what();
            return absl::Status(absl::StatusCode::kInternal, "Error occurred during Python code execution");
        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            return absl::Status(absl::StatusCode::kUnknown, "Unexpected error occurred");
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            return absl::Status(absl::StatusCode::kUnknown, "Unexpected error occurred");
        }
        LOG(INFO) << "PythonExecutorCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonExecutorCalculator);
}  // namespace mediapipe
