//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

#include "../status.hpp"
#include "python_backend.hpp"
#include "python_calculators_plugin_api.hpp"
#include "pythonnoderesources.hpp"

namespace py = pybind11;

namespace ovms {
namespace {

// Handles are heap-allocated owned references; generators are heap-allocated py::iterator.
void* toHandle(py::object object) {
    return new py::object(std::move(object));
}

const py::object& fromHandle(const void* handle) {
    return *static_cast<const py::object*>(handle);
}

template <typename Body>
PythonPluginResult runWithGil(std::string& message, Body&& body) {
    py::gil_scoped_acquire acquire;
    try {
        body();
        return PythonPluginResult::OK;
    } catch (const UnexpectedOutputTensorError& e) {
        message = e.what();
    } catch (const UnexpectedOutputPythonObjectError& e) {
        message = std::string("Wrong object on execute output: ") + e.what();
    } catch (const UnexpectedInputPythonObjectError& e) {
        message = std::string("Wrong object on execute input: ") + e.what();
    } catch (const BadPythonNodeConfigurationError& e) {
        message = e.what();
    } catch (const pybind11::error_already_set& e) {
        message = e.what();
        return PythonPluginResult::PYTHON_EXCEPTION;
    } catch (const std::exception& e) {
        message = e.what();
    } catch (...) {
        message = "Unexpected error";
    }
    return PythonPluginResult::OTHER_FAILURE;
}

// Requires GIL. Appends nothing unless every output is valid.
void collectOutputs(PythonNodeResources& nodeResources, const py::list& pyOutputs, std::vector<PythonNodeOutput>& outputs) {
    std::vector<std::pair<std::string, py::object>> collected;
    collected.reserve(pyOutputs.size());
    for (py::handle pyOutputHandle : pyOutputs) {
        py::object pyOutput = pyOutputHandle.cast<py::object>();
        try {
            nodeResources.pythonBackend->validateOvmsPyTensor(pyOutput);
        } catch (UnexpectedPythonObjectError& e) {
            throw UnexpectedOutputPythonObjectError(e);
        }
        std::string outputName = pyOutput.attr("name").cast<std::string>();
        auto it = nodeResources.outputsNameTagMapping.find(outputName);
        if (it == nodeResources.outputsNameTagMapping.end()) {
            throw UnexpectedOutputTensorError(outputName);
        }
        collected.emplace_back(it->second, std::move(pyOutput));
    }
    outputs.reserve(outputs.size() + collected.size());
    for (auto& [tag, pyOutput] : collected) {
        outputs.push_back(PythonNodeOutput{tag, toHandle(std::move(pyOutput))});
    }
}

int createNodeResources(const PythonNodeConfig& config, PythonBackend* pythonBackend, std::shared_ptr<PythonNodeResources>& nodeResources) {
    return static_cast<int>(PythonNodeResources::createPythonNodeResources(nodeResources, config, pythonBackend).getCode());
}

bool hasPythonBackend(const PythonNodeResources& nodeResources) {
    return nodeResources.pythonBackend != nullptr;
}

void releaseObject(void* object) {
    if (object == nullptr) {
        return;
    }
    py::gil_scoped_acquire acquire;
    delete static_cast<py::object*>(object);
}

void releaseGenerator(void* generator) {
    if (generator == nullptr) {
        return;
    }
    py::gil_scoped_acquire acquire;
    delete static_cast<py::iterator*>(generator);
}

PythonPluginResult execute(PythonNodeResources& nodeResources, const std::vector<const void*>& inputs,
    std::vector<PythonNodeOutput>& outputs, void*& generator, std::string& message) {
    return runWithGil(message, [&]() {
        std::vector<py::object> pyInputs;
        pyInputs.reserve(inputs.size());
        for (const void* input : inputs) {
            const py::object& pyInput = fromHandle(input);
            try {
                nodeResources.pythonBackend->validateOvmsPyTensor(pyInput);
            } catch (UnexpectedPythonObjectError& e) {
                throw UnexpectedInputPythonObjectError(e);
            }
            pyInputs.push_back(pyInput);
        }
        py::object executeResult = nodeResources.ovmsPythonModel->attr("execute")(pyInputs);
        if (py::isinstance<py::list>(executeResult)) {
            collectOutputs(nodeResources, executeResult.cast<py::list>(), outputs);
        } else if (py::isinstance<py::iterator>(executeResult)) {
            generator = new py::iterator(std::move(executeResult));
        } else {
            throw UnexpectedPythonObjectError(executeResult, "list or generator");
        }
    });
}

PythonPluginResult generatorNext(PythonNodeResources& nodeResources, void* generator,
    std::vector<PythonNodeOutput>& outputs, bool& produced, std::string& message) {
    produced = false;
    return runWithGil(message, [&]() {
        auto& iterator = *static_cast<py::iterator*>(generator);
        if (iterator == py::iterator::sentinel()) {
            return;
        }
        collectOutputs(nodeResources, py::cast<py::list>(*iterator), outputs);
        produced = true;
        ++iterator;  // runs the handler up to its next yield
    });
}

PythonPluginResult createTensor(const std::string& name, const void* data, const std::vector<int64_t>& shape,
    const std::string& datatype, size_t size, void*& tensor, std::string& message) {
    return runWithGil(message, [&]() {
        PythonBackend pythonBackend;
        std::unique_ptr<PyObjectWrapper<py::object>> pyTensor;
        const std::vector<py::ssize_t> pyShape(shape.begin(), shape.end());
        if (!pythonBackend.createOvmsPyTensor(name, const_cast<void*>(data), pyShape, datatype, static_cast<py::ssize_t>(size), pyTensor, true)) {
            throw std::runtime_error("Failed to create Python tensor: " + name);
        }
        tensor = toHandle(pyTensor->getObject());
    });
}

PythonPluginResult getTensorInfo(const void* tensor, std::string& datatype, std::vector<int64_t>& shape,
    const void*& data, size_t& size, std::string& message) {
    return runWithGil(message, [&]() {
        const py::object& pyTensor = fromHandle(tensor);
        const py::object tensorClass = py::module_::import("pyovms").attr("Tensor");
        if (!py::isinstance(pyTensor, tensorClass)) {
            throw UnexpectedPythonObjectError(pyTensor, tensorClass.attr("__name__").cast<std::string>());
        }
        datatype = pyTensor.attr("datatype").cast<std::string>();
        shape = pyTensor.attr("shape").cast<std::vector<int64_t>>();
        data = pyTensor.attr("ptr").cast<void*>();
        size = pyTensor.attr("size").cast<size_t>();
    });
}

const PythonCalculatorsPluginApi pythonCalculatorsPluginApi{
    PYTHON_CALCULATORS_PLUGIN_ABI_VERSION,
    createNodeResources,
    hasPythonBackend,
    releaseObject,
    releaseGenerator,
    execute,
    generatorNext,
    createTensor,
    getTensorInfo,
};

}  // namespace
}  // namespace ovms

#if defined(_WIN32)
#define PYTHON_CALCULATORS_EXPORT __declspec(dllexport)
#else
#define PYTHON_CALCULATORS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" PYTHON_CALCULATORS_EXPORT const ovms::PythonCalculatorsPluginApi* OVMS_getPythonCalculatorsPluginApi() {
    return &ovms::pythonCalculatorsPluginApi;
}

#undef PYTHON_CALCULATORS_EXPORT
