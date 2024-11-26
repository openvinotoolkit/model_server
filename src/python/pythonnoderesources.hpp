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
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include <pybind11/embed.h>  // everything needed for embedding

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../status.hpp"
#include "src/python/python_executor_calculator.pb.h"

namespace py = pybind11;

namespace ovms {
class PythonBackend;

struct PythonNodeResources {
public:
    PythonNodeResources(const PythonNodeResources&) = delete;
    PythonNodeResources& operator=(PythonNodeResources&) = delete;

    std::unique_ptr<py::object> ovmsPythonModel;
    PythonBackend* pythonBackend;
    std::string handlerPath;
    std::unordered_map<std::string, std::string> outputsNameTagMapping;

    PythonNodeResources(PythonBackend* pythonBackend);
    ~PythonNodeResources();
    static void createOutputTagNameMapping(std::shared_ptr<PythonNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig) {
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

    static Status createPythonNodeResources(std::shared_ptr<PythonNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, PythonBackend* pythonBackend, std::string graphPath) {
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
        nodeResources->handlerPath = hpath.string();
        if (!std::filesystem::exists(hpath)) {
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

    void finalize();

private:
    static py::dict preparePythonNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, const std::string& basePath);
};
using PythonNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>;
}  // namespace ovms
