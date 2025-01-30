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
#include "pythonnoderesources.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 6011 6326 6385 6246 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#pragma warning(pop)

#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "src/python/python_executor_calculator.pb.h"

namespace ovms {

PythonNodeResources::PythonNodeResources(PythonBackend* pythonBackend) {
    this->ovmsPythonModel = nullptr;
    this->pythonBackend = pythonBackend;
    this->handlerPath = "";
}

void PythonNodeResources::finalize() {
    if (this->ovmsPythonModel) {
        py::gil_scoped_acquire acquire;
        try {
            if (!py::hasattr(*ovmsPythonModel.get(), "finalize")) {
                SPDLOG_DEBUG("Python node resource does not have a finalize method. Python node handler_path: {} ", this->handlerPath);
                return;
            }

            ovmsPythonModel.get()->attr("finalize")();
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_ERROR("Failed to process python node finalize method. {}  Python node handler_path: {} ", e.what(), this->handlerPath);
            return;
        } catch (...) {
            SPDLOG_ERROR("Failed to process python node finalize method. Python node handler_path: {} ", this->handlerPath);
            return;
        }
    }
}

// IMPORTANT: This is an internal method meant to be run in a specific context.
// It assumes GIL is being held by the thread and doesn't handle potential errors.
// It MUST be called in the scope of py::gil_scoped_acquire and within the try - catch block
py::dict PythonNodeResources::preparePythonNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, const std::string& basePath) {
    py::dict kwargsParam = py::dict();
    std::string nodeName = graphNodeConfig.name();
    py::list inputStreams = py::list();
    py::list outputStreams = py::list();
    for (auto& name : graphNodeConfig.input_stream()) {
        inputStreams.append(getStreamName(name));
    }

    for (auto& name : graphNodeConfig.output_stream()) {
        outputStreams.append(getStreamName(name));
    }

    kwargsParam["input_names"] = inputStreams;
    kwargsParam["output_names"] = outputStreams;
    kwargsParam["node_name"] = nodeName;
    kwargsParam["base_path"] = py::str(basePath);

    return kwargsParam;
}

void createOutputTagNameMapping(std::shared_ptr<PythonNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig) {
    for (auto& name : graphNodeConfig.output_stream()) {
        std::string delimiter = ":";
        std::string streamTag, streamName;
        size_t tagDelimiterPos = name.find(delimiter, 0);

        if (tagDelimiterPos == std::string::npos) {
            // Empty tag - example: output_stream: "output"
            streamTag = "";
            streamName = name;
        } else {
            streamTag = name.substr(0, tagDelimiterPos);
            size_t indexDelimiterPos = name.find(delimiter, tagDelimiterPos + 1);
            if (indexDelimiterPos == std::string::npos) {
                // Only tag, no index - example: output_stream: "OUTPUT:output"
                streamName = name.substr(tagDelimiterPos + 1, std::string::npos);
            } else {
                // Both tag and index - example: output_stream: "OUTPUT:0:output"
                // It's permitted by MediaPipe, but PythonExecutorCalculator ignores it.
                streamName = name.substr(indexDelimiterPos + 1, std::string::npos);
            }
        }
        // PythonExecutorCalculator ignores index value, so only Tag gets mapped
        nodeResources->outputsNameTagMapping.insert({streamName, streamTag});
    }
}

Status PythonNodeResources::createPythonNodeResources(std::shared_ptr<PythonNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, PythonBackend* pythonBackend, std::string graphPath) {
    mediapipe::PythonExecutorCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);

    nodeResources = std::make_shared<PythonNodeResources>(pythonBackend);
    createOutputTagNameMapping(nodeResources, graphNodeConfig);

    auto fsHandlerPath = std::filesystem::path(nodeOptions.handler_path());

    std::string basePath;
    std::string extension = fsHandlerPath.extension().string();
    fsHandlerPath.replace_extension();
    std::string filename = fsHandlerPath.filename().string();
    if (fsHandlerPath.is_relative()) {
        basePath = (std::filesystem::path(graphPath) / fsHandlerPath.parent_path()).string();
    } else {
        basePath = fsHandlerPath.parent_path().string();
    }
    auto hpath = std::filesystem::path(basePath) / std::filesystem::path(filename + extension);
    // Make final handler path uniform with forward slashes as separators
    std::string hpathStr = hpath.string();
    std::replace(hpathStr.begin(), hpathStr.end(), '\\', '/');  // Replace backslashes with forward slashes
    nodeResources->handlerPath = hpathStr;
    if (!std::filesystem::exists(hpathStr)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Python node handler_path: {} does not exist. ", hpath.string());
        return StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST;
    }
    py::gil_scoped_acquire acquire;
    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(basePath.c_str());
        py::module_ script = py::module_::import(filename.c_str());

        if (!py::hasattr(script, "OvmsPythonModel")) {
            SPDLOG_ERROR("Error during python node initialization. No OvmsPythonModel class found in {}", nodeOptions.handler_path());
            return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
        }

        py::object OvmsPythonModel = script.attr("OvmsPythonModel");
        if (!py::hasattr(OvmsPythonModel, "execute")) {
            SPDLOG_ERROR("Error during python node initialization. OvmsPythonModel class defined in {} does not implement execute method.", nodeOptions.handler_path());
            return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
        }

        nodeResources->ovmsPythonModel = std::make_unique<py::object>(OvmsPythonModel());
        if (py::hasattr(*nodeResources->ovmsPythonModel, "initialize")) {
            py::dict kwargsParam = preparePythonNodeInitializeArguments(graphNodeConfig, basePath);
            nodeResources->ovmsPythonModel->attr("initialize")(kwargsParam);
        } else {
            SPDLOG_DEBUG("OvmsPythonModel class defined in {} does not implement initialize method.", nodeOptions.handler_path());
        }
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_ERROR("Error during python node initialization for handler_path: {} - {}", nodeOptions.handler_path(), e.what());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Error during python node initialization for handler_path: {}", nodeOptions.handler_path());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    }
    return StatusCode::OK;
}

PythonNodeResources::~PythonNodeResources() {
    SPDLOG_DEBUG("Calling Python node resource destructor");
    this->finalize();
    py::gil_scoped_acquire acquire;
    this->ovmsPythonModel.reset();
}

}  // namespace ovms
