//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include "modelinstance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include <dirent.h>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include "config.hpp"
#include "customloaders.hpp"
#include "deserialization.hpp"
#include "executingstreamidguard.hpp"
#include "filesystem.hpp"
#include "layout.hpp"
#include "layout_configuration.hpp"
#include "logging.hpp"
#include "ov_utils.hpp"
#include "predict_request_validation_utils.hpp"
#include "prediction_service_utils.hpp"
#include "profiler.hpp"
#include "serialization.hpp"
#include "shape.hpp"
#include "stringutils.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"

namespace ovms {

const char* CPU_THROUGHPUT_STREAMS = "CPU_THROUGHPUT_STREAMS";
const char* NIREQ = "NIREQ";

const uint MAX_NIREQ_COUNT = 100000;

const int DEFAULT_OV_STREAMS = std::thread::hardware_concurrency() / 4;

const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;

void ModelInstance::subscribe(PipelineDefinition& pd) {
    subscriptionManager.subscribe(pd);
}

void ModelInstance::unsubscribe(PipelineDefinition& pd) {
    subscriptionManager.unsubscribe(pd);
}

Status getRequestedShape(const ModelConfig& config, const DynamicModelParameter& parameter, const std::string& name, Shape& shapeOut) {
    Shape shape;
    auto mappedName = config.getMappingInputByKey(name);
    if (config.getBatchSize().has_value() || parameter.isBatchSizeRequested()) {
        // leave shape untouched
    } else if (config.isShapeAuto(name) && parameter.isShapeRequested(name)) {
        auto status = Shape::fromFlatShape(parameter.getShape(name), shape);
        if (!status.ok()) {
            return status;
        }
    } else if (mappedName == "" && config.getShapes().count(name) && config.getShapes().at(name).shape.size()) {
        shape = config.getShapes().at(name).shape;
    } else if (config.getShapes().count(mappedName) && config.getShapes().at(mappedName).shape.size()) {
        shape = config.getShapes().at(mappedName).shape;
    } else if (config.getShapes().count(ANONYMOUS_INPUT_NAME) && config.getShapes().at(ANONYMOUS_INPUT_NAME).shape.size()) {
        shape = config.getShapes().at(ANONYMOUS_INPUT_NAME).shape;
    }
    shapeOut = shape;
    return StatusCode::OK;
}

bool hasInputWithName(std::shared_ptr<ov::Model>& model, const std::string& name) {
    try {
        model->input(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

bool hasOutputWithName(std::shared_ptr<ov::Model>& model, const std::string& name) {
    try {
        model->output(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

Status validateConfigurationAgainstNetwork(const ModelConfig& config, std::shared_ptr<ov::Model>& model) {
    if (config.isShapeAnonymousFixed() && model->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    if (config.getLayout().isSet() && model->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_LAYOUT_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    for (const auto& [name, _] : config.getShapes()) {
        if (name == ANONYMOUS_INPUT_NAME) {
            continue;
        }
        if (hasInputWithName(model, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(model, name) && !hasInputWithName(model, config.getRealInputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} not found in model", name);
            return StatusCode::CONFIG_SHAPE_IS_NOT_IN_MODEL;
        }
    }
    for (const auto& [name, _] : config.getLayouts()) {
        if (hasInputWithName(model, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (hasOutputWithName(model, name) && config.getMappingOutputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingOutputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(model, name) && !hasOutputWithName(model, name) && !hasInputWithName(model, config.getRealInputNameByValue(name)) && !hasOutputWithName(model, config.getRealOutputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} not found in model", name);
            return StatusCode::CONFIG_LAYOUT_IS_NOT_IN_MODEL;
        }
    }
    return StatusCode::OK;
}

const Layout ModelInstance::getReportedTensorLayout(const ModelConfig& config, const std::string& name, bool isInput) {
    if (isInput) {
        const auto& input = this->model->input(name);
        auto networkSpecifiedLayout = getLayoutFromRTMap(input.get_rt_info());
        if (networkSpecifiedLayout.has_value()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting input layout from RTMap: {}; for tensor name: {}", networkSpecifiedLayout.value().to_string(), name);
            return Layout::fromOvLayout(networkSpecifiedLayout.value());
        }
    } else {
        const auto& output = this->model->output(name);
        auto networkSpecifiedLayout = getLayoutFromRTMap(output.get_rt_info());
        if (networkSpecifiedLayout.has_value()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting output layout from RTMap: {}; for tensor name: {}", networkSpecifiedLayout.value().to_string(), name);
            return Layout::fromOvLayout(networkSpecifiedLayout.value());
        }
    }
    auto layout = Layout::getDefaultLayout();
    if (isInput && config.getLayout().isSet()) {
        layout = config.getLayout().getTensorLayout();
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting layout from ModelConfig: {}; for tensor name: {}", layout, name);
        return layout;
    } else if (config.getLayouts().size() > 0) {
        auto mappedName = config.getMappingInputByKey(name);
        auto it = config.getLayouts().find(mappedName == "" ? name : mappedName);
        if (it != config.getLayouts().end()) {
            layout = it->second.getTensorLayout();
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting layout from ModelConfig: {}; for tensor name: {}", layout, name);
            return layout;
        }
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting default layout: {}; for tensor name: {}", layout, name);
    return layout;
}

Status applyLayoutConfiguration(const ModelConfig& config, std::shared_ptr<ov::Model>& model, const std::string& modelName, model_version_t modelVersion) {
    ov::preprocess::PrePostProcessor preproc(model);

    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Applying layout configuration: {}", config.layoutConfigurationToString());

    for (const ov::Output<ov::Node>& input : model->inputs()) {
        try {
            std::string name = input.get_any_name();
            std::string mappedName = config.getMappingInputByKey(name).empty() ? name : config.getMappingInputByKey(name);
            if (config.getLayout().isSet()) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; single input",
                    modelName,
                    modelVersion,
                    config.getLayout().getTensorLayout(),
                    config.getLayout().getModelLayout());

                preproc.input().tensor().set_layout(ov::Layout(config.getLayout().getTensorLayout()));
                preproc.input().model().set_layout(ov::Layout(config.getLayout().getModelLayout()));
            } else if (config.getLayouts().count(mappedName) > 0) {
                auto& layout = config.getLayouts().at(mappedName);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; input name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    mappedName);

                preproc.input(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                preproc.input(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            } else {
                auto inheritedModelLayout = getLayoutFromRTMap(input.get_rt_info());
                auto guessedModelLayout = Layout::getDefaultLayout();

                ov::Layout targetModelLayout = inheritedModelLayout.has_value() ? inheritedModelLayout.value() : ov::Layout(guessedModelLayout);

                if (inheritedModelLayout.has_value()) {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (inherited from network); input name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                } else {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (default); input name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                }
                preproc.input(name).model().set_layout(targetModelLayout);
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO",
                modelName,
                modelVersion);
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    for (const ov::Output<ov::Node>& output : model->outputs()) {
        try {
            std::string name = output.get_any_name();
            std::string mappedName = config.getMappingOutputByKey(name).empty() ? name : config.getMappingOutputByKey(name);
            if (config.getLayouts().count(mappedName) > 0) {
                auto& layout = config.getLayouts().at(mappedName);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding postprocessing step: Tensor Layout:{}; Network Layout:{}; output name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    mappedName);
                preproc.output(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                preproc.output(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            } else {
                auto inheritedModelLayout = getLayoutFromRTMap(output.get_rt_info());
                auto guessedModelLayout = Layout::getDefaultLayout();

                ov::Layout targetModelLayout = inheritedModelLayout.has_value() ? inheritedModelLayout.value() : ov::Layout(guessedModelLayout);

                if (inheritedModelLayout.has_value()) {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (inherited from network); output name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                } else {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (default); output name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                }
                preproc.output(name).model().set_layout(targetModelLayout);
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO",
                modelName,
                modelVersion);
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    try {
        model = preproc.build();
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot change layout");
        return StatusCode::MODEL_NOT_LOADED;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadTensors(const ModelConfig& config, bool needsToApplyLayoutConfiguration, const DynamicModelParameter& parameter) {
    Status status = validateConfigurationAgainstNetwork(config, this->model);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during configuration validation against model");
        return status;
    }
    if (needsToApplyLayoutConfiguration) {
        status = applyLayoutConfiguration(config, this->model, getName(), getVersion());
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during layout configuration");
            return status;
        }
    }
    status = loadInputTensors(config, parameter);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during loading input tensors");
        return status;
    }
    status = loadOutputTensors(config);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during loading output tensors");
        return status;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadInputTensors(const ModelConfig& config, const DynamicModelParameter& parameter) {
    this->inputsInfo.clear();

    std::map<std::string, ov::PartialShape> modelShapes;
    bool reshapeRequired = false;

    // First pass, gather reshape info.
    for (const ov::Output<ov::Node>& input : this->model->inputs()) {
        std::string name;
        try {
            std::string name = input.get_any_name();
            ov::PartialShape shape = input.get_partial_shape();

            Shape requestedShape;
            auto status = getRequestedShape(config, parameter, name, requestedShape);
            if (!status.ok()) {
                return status;
            }
            if (requestedShape.size() > 0) {
                shape = requestedShape.createPartialShape();
            }

            modelShapes[name] = shape;
            if (input.get_partial_shape() != shape) {
                reshapeRequired = true;
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    if (reshapeRequired) {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs", getName(), getVersion());
        try {
            model->reshape(modelShapes);
        } catch (const ov::Exception& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        }
    } else {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs is not required", getName(), getVersion());
    }

    configureBatchSize(this->config, parameter);

    for (const ov::Output<ov::Node>& input : this->model->inputs()) {
        try {
            std::string name = input.get_any_name();

            ovms::Precision precision = ovElementTypeToOvmsPrecision(input.get_element_type());
            Shape shape(input.get_partial_shape());
            std::string mappingName = config.getMappingInputByKey(name);
            const Layout layout = getReportedTensorLayout(config, name, true);

            if (!layout.isCompatible(shape)) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Layout: {}; incompatible with shape: {}; for input name: {}", layout, shape.toString(), name);
                return StatusCode::LAYOUT_INCOMPATIBLE_WITH_SHAPE;
            }

            std::shared_ptr<TensorInfo> info = std::make_shared<TensorInfo>(
                name,
                mappingName,
                precision,
                shape,
                layout);

            SPDLOG_LOGGER_INFO(modelmanager_logger, "Input {}", info->asString());

            this->inputsInfo[info->getMappedName()] = std::move(info);
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOutputTensors(const ModelConfig& config) {
    this->outputsInfo.clear();

    for (const ov::Output<ov::Node>& output : this->model->outputs()) {
        try {
            std::string name = output.get_any_name();

            ovms::Precision precision = ovElementTypeToOvmsPrecision(output.get_element_type());
            Shape shape(output.get_partial_shape());
            std::string mappingName = config.getMappingOutputByKey(name);
            const Layout layout = getReportedTensorLayout(config, name, false);

            if (!layout.isCompatible(shape)) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Layout: {}; incompatible with shape: {}; for output name: {}", layout, shape.toString(), name);
                return StatusCode::LAYOUT_INCOMPATIBLE_WITH_SHAPE;
            }

            std::shared_ptr<TensorInfo> info = std::make_shared<TensorInfo>(
                name,
                mappingName,
                precision,
                shape,
                layout);

            SPDLOG_LOGGER_INFO(modelmanager_logger, "Output {}", info->asString());

            this->outputsInfo[info->getMappedName()] = std::move(info);
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    return StatusCode::OK;
}

// Temporary methods. To be replaces with proper storage class.
bool dirExists(const std::string& path) {
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return false;
    }
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }

    return false;
}

std::string findFilePathWithExtension(const std::string& path, const std::string& extension) {
    struct dirent* entry;
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return std::string();
    }
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        SPDLOG_WARN("Failed to opendir: {}", path);
        return std::string();
    }

    while ((entry = readdir(dir)) != nullptr) {
        auto name = std::string(entry->d_name);
        if (endsWith(name, extension)) {
            closedir(dir);
            if (endsWith(name, "/")) {
                return path + name;
            } else {
                return path + '/' + name;
            }
        }
    }
    closedir(dir);

    return std::string();
}

std::string ModelInstance::findModelFilePathWithExtension(const std::string& extension) const {
    return findFilePathWithExtension(path, extension);
}

uint ModelInstance::getNumOfParallelInferRequestsUnbounded(const ModelConfig& modelConfig) {
    uint numberOfParallelInferRequests = 0;
    if (modelConfig.getNireq() > 0) {
        return modelConfig.getNireq();
    }
    auto& ovmsConfig = ovms::Config::instance();
    if (ovmsConfig.nireq() > 0) {
        // nireq is set globally for all models in ovms startup parameters
        return ovmsConfig.nireq();
    }
    std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
    try {
        numberOfParallelInferRequests = compiledModel->get_property(key).as<unsigned int>();
    } catch (const ov::Exception& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    return numberOfParallelInferRequests;
}

uint ModelInstance::getNumOfParallelInferRequests(const ModelConfig& modelConfig) {
    uint nireq = getNumOfParallelInferRequestsUnbounded(modelConfig);
    if (nireq > MAX_NIREQ_COUNT) {
        SPDLOG_WARN("Invalid nireq because its value was too high: {}. Maximum value: {}", nireq, MAX_NIREQ_COUNT);
        return 0;
    } else if (nireq < 1u) {
        SPDLOG_WARN("Ignored configured nireq because it has to be above 0 and was: {}. Set to 1", nireq);
        return 1u;
    }
    return nireq;
}

std::shared_ptr<ov::Model> ModelInstance::loadOVModelPtr(const std::string& modelFile) {
    return ieCore.read_model(modelFile);
}

Status ModelInstance::loadOVModel() {
    auto& modelFile = modelFiles[0];
    SPDLOG_DEBUG("Try reading model file: {}", modelFile);
    try {
        model = loadOVModelPtr(modelFile);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOVModelUsingCustomLoader() {
    SPDLOG_DEBUG("Try reading model using a custom loader");
    try {
        std::vector<uint8_t> modelBinary;
        std::vector<uint8_t> weights;

        SPDLOG_INFO("loading ov::Model for model: {} basepath: {} <> {} version: {}", getName(), getPath(), this->config.getBasePath().c_str(), getVersion());

        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];

        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_INFO("Loader {} is not in loaded customloaders list", loaderName);
            throw std::invalid_argument("customloader not exisiting");
        }

        CustomLoaderStatus res = customLoaderInterfacePtr->loadModel(this->config.getName(),
            this->config.getBasePath(),
            getVersion(),
            this->config.getCustomLoaderOptionsConfigStr(), modelBinary, weights);

        if (res == CustomLoaderStatus::MODEL_LOAD_ERROR) {
            return StatusCode::FILE_INVALID;
        }

        if ((res == CustomLoaderStatus::INTERNAL_ERROR) || (res == CustomLoaderStatus::MODEL_BLACKLISTED)) {
            return StatusCode::INTERNAL_ERROR;
        }

        std::string strModel(modelBinary.begin(), modelBinary.end());

        if (res == CustomLoaderStatus::MODEL_TYPE_IR) {
            ov::Tensor tensorWts(ov::element::u8, ov::Shape{weights.size()});
            std::memcpy(tensorWts.data(), weights.data(), weights.size());
            model = ieCore.read_model(strModel, tensorWts);
        } else if (res == CustomLoaderStatus::MODEL_TYPE_ONNX) {
            model = ieCore.read_model(strModel, ov::Tensor());
        } else if (res == CustomLoaderStatus::MODEL_TYPE_BLOB) {
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (ov::Exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

void ModelInstance::loadCompiledModelPtr(const plugin_config_t& pluginConfig) {
    compiledModel = std::make_shared<ov::CompiledModel>(ieCore.compile_model(model, targetDevice, pluginConfig));
}

plugin_config_t ModelInstance::prepareDefaultPluginConfig(const ModelConfig& config) {
    plugin_config_t pluginConfig = config.getPluginConfig();
    // Do not add CPU_THROUGHPUT_AUTO when performance hint is specified.
    bool isPerformanceHintSpecified = pluginConfig.count("PERFORMANCE_HINT") > 0;
    if (isPerformanceHintSpecified) {
        return pluginConfig;
    }
    // For CPU and GPU, if user did not specify, calculate CPU_THROUGHPUT_STREAMS automatically
    if (config.isSingleDeviceUsed("CPU")) {
        if (pluginConfig.count("CPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["CPU_THROUGHPUT_STREAMS"] = "CPU_THROUGHPUT_AUTO";
        }
    }
    if (config.isSingleDeviceUsed("GPU")) {
        if (pluginConfig.count("GPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO";
        }
    }
    return pluginConfig;
}

Status ModelInstance::loadOVCompiledModel(const ModelConfig& config) {
    plugin_config_t pluginConfig = prepareDefaultPluginConfig(config);
    try {
        loadCompiledModelPtr(pluginConfig);
    } catch (ov::Exception& e) {
        Status status = StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    } catch (std::exception& e) {
        Status status = StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    }

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Plugin config for device: {}", targetDevice);
    for (const auto pair : pluginConfig) {
        const auto key = pair.first;
        const auto value = pair.second;
        SPDLOG_LOGGER_INFO(modelmanager_logger, "OVMS set plugin settings key: {}; value: {};", key, value.as<std::string>());
    }

    const std::string supportedConfigKey = METRIC_KEY(SUPPORTED_CONFIG_KEYS);
    std::vector<std::string> supportedConfigKeys;
    try {
        std::vector<std::string> supportedConfigKeys2 = compiledModel->get_property(supportedConfigKey).as<std::vector<std::string>>();
        supportedConfigKeys = std::move(supportedConfigKeys2);
    } catch (std::exception& e) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, CompiledModel metric key: {}; Error: {}", targetDevice, supportedConfigKey, e.what());
        return StatusCode::OK;
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, CompiledModel metric key: {}", targetDevice, supportedConfigKey);
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Logging model:{}; version: {};target device: {}; CompiledModel configuration", getName(), getVersion(), targetDevice);
    for (auto& key : supportedConfigKeys) {
        std::string value;
        try {
            auto paramValue = compiledModel->get_property(key);
            value = paramValue.as<std::string>();
        } catch (std::exception& e) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, CompiledModel config key: {}; Error: {}", targetDevice, key, e.what());
            continue;
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, CompiledModel config key: {}", targetDevice, key);
            continue;
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {}; version: {}; target device: {}, CompiledModel config key: {}, value: {}", getName(), getVersion(), targetDevice, key, value);
    }
    return StatusCode::OK;
}

Status ModelInstance::fetchModelFilepaths() {
    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        // not required if the model is loaded using a custom loader and can be returned from here
        return StatusCode::OK;
    }

    SPDLOG_DEBUG("Getting model files from path: {}", path);
    if (!dirExists(path)) {
        SPDLOG_ERROR("Missing model directory {}", path);
        return StatusCode::PATH_INVALID;
    }

    bool found = true;
    for (auto extension : OV_MODEL_FILES_EXTENSIONS) {
        auto file = findModelFilePathWithExtension(extension);
        if (file.empty()) {
            found = false;
        }
        modelFiles.push_back(file);
    }
    if (!found) {
        found = true;
        modelFiles.clear();
        for (auto extension : ONNX_MODEL_FILES_EXTENSIONS) {
            auto file = findModelFilePathWithExtension(extension);
            if (file.empty()) {
                found = false;
            }
            modelFiles.push_back(file);
        }
    }

    if (!found) {
        SPDLOG_ERROR("Could not find file for model: {} version: {} in path: {}", getName(), getVersion(), path);
        return StatusCode::FILE_INVALID;
    }

    return StatusCode::OK;
}

Status ModelInstance::prepareInferenceRequestsQueue(const ModelConfig& config) {
    uint numberOfParallelInferRequests = getNumOfParallelInferRequests(config);
    if (numberOfParallelInferRequests == 0) {
        return Status(StatusCode::INVALID_NIREQ, "Exceeded allowed nireq value");
    }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(*compiledModel, numberOfParallelInferRequests);
    SPDLOG_INFO("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
        getName(),
        getVersion(),
        getBatchSize().toString(),
        numberOfParallelInferRequests);
    return StatusCode::OK;
}

void ModelInstance::configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter) {
    if (parameter.isBatchSizeRequested()) {
        ov::set_batch(model, parameter.getBatchSize());
    } else if (config.getBatchSize().has_value()) {
        ov::set_batch(model, config.getBatchSize().value().createPartialDimension());
    }
}

Status ModelInstance::loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter) {
    bool isLayoutConfigurationChanged = !config.isLayoutConfigurationEqual(this->config);
    bool needsToApplyLayoutConfiguration = isLayoutConfigurationChanged || !this->model;

    subscriptionManager.notifySubscribers();
    this->path = config.getPath();
    this->targetDevice = config.getTargetDevice();
    this->config = config;
    auto status = fetchModelFilepaths();

    if (!status.ok()) {
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return status;
    }
    try {
        status = setCacheOptions(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        if (!this->model || isLayoutConfigurationChanged) {
            if (this->config.isCustomLoaderRequiredToLoadModel()) {
                // loading the model using the custom loader
                status = loadOVModelUsingCustomLoader();
            } else {
                status = loadOVModel();
            }
        }

        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        status = loadTensors(this->config, needsToApplyLayoutConfiguration, parameter);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = loadOVCompiledModel(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = prepareInferenceRequestsQueue(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
    } catch (const ov::Exception& e) {
        SPDLOG_ERROR("exception occurred while loading model: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::MODEL_NOT_LOADED;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("exception occurred while loading model: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::MODEL_NOT_LOADED;
    }
    this->status.setAvailable();
    modelLoadedNotify.notify_all();
    return status;
}

Status ModelInstance::setCacheOptions(const ModelConfig& config) {
    if (!config.getCacheDir().empty()) {
        if (!config.isAllowCacheSetToTrue() && (config.isCustomLoaderRequiredToLoadModel() || config.anyShapeSetToAuto() || (config.getBatchingMode() == Mode::AUTO))) {
            this->ieCore.set_property({{CONFIG_KEY(CACHE_DIR), ""}});
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has disabled caching", this->getName());
            this->cacheDisabled = true;
        } else if (config.isAllowCacheSetToTrue() && config.isCustomLoaderRequiredToLoadModel()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model: {} has allow cache set to true while using custom loader", this->getName());
            return StatusCode::ALLOW_CACHE_WITH_CUSTOM_LOADER;
        } else {
            this->ieCore.set_property({{CONFIG_KEY(CACHE_DIR), config.getCacheDir()}});
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has enabled caching", this->getName());
        }
    }
    return StatusCode::OK;
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Loading model: {}, version: {}, from path: {}, with target device: {} ...",
        config.getName(), config.getVersion(), config.getPath(), config.getTargetDevice());
    if (config.getBatchingMode() == AUTO) {
        SPDLOG_INFO("Batch size mode for model {} is set to auto", config.getName());
    } else if (config.anyShapeSetToAuto()) {
        SPDLOG_INFO("Some inputs shapes for model {} are set to auto", config.getName());
    }
    this->status = ModelVersionStatus(config.getName(), config.getVersion());
    this->status.setLoading();
    return loadModelImpl(config);
}

Status ModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to reload model: {} version: {}. Blocked by: {} inferences in progress.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    if ((this->config.isCustomLoaderRequiredToLoadModel()) && (isCustomLoaderConfigChanged)) {
        // unloading and the loading back the model
        isCustomLoaderConfigChanged = false;
        retireModel(isCustomLoaderConfigChanged);
    }
    return loadModelImpl(config, parameter);
}

Status ModelInstance::recoverFromReloadingError(const Status& status) {
    SPDLOG_WARN("Failed to perform complete reload with requested dynamic parameter. Model: {} version: {} with error: {}. Reloading to previous configuration",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to recover model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return status;
}

Status ModelInstance::reshapeWithFullReload(const Status& status, const DynamicModelParameter& parameter) {
    SPDLOG_WARN("Failed to reload model: {} version: {} with error: {}. Trying to perform complete reload with requested dynamic parameter",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config, parameter);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to reload model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return recoveryStatus;
}

Status ModelInstance::reloadModel(std::optional<Dimension> batchSize, std::map<std::string, shape_t> requestShapes, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuard) {
    // temporarily release current predictRequest lock on model loading
    unloadGuard.reset();
    // block concurrent requests for reloading/unloading - assure that after reload predict request
    // will block further requests for reloading/unloading until inference is performed
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Will reload model: {} version: {}", getName(), getVersion());

    DynamicModelParameter parameter;
    if (batchSize.has_value() && batchSize.value().isStatic()) {
        parameter = DynamicModelParameter(batchSize.value().getStaticValue());
    } else if (requestShapes.size() > 0) {
        parameter = DynamicModelParameter(requestShapes);
    } else {
        SPDLOG_DEBUG("Error: requested model: {} version: {} reload with no batchsize and shapes set.", getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }

    auto status = reloadModel(config, parameter);
    if (!status.ok()) {
        status = this->reshapeWithFullReload(status, parameter);
        if (!status.ok()) {
            return this->recoverFromReloadingError(status);
        }
    }
    unloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    return status;
}

Status ModelInstance::reloadModelIfRequired(
    Status validationStatus,
    const std::optional<Dimension>& requestedBatchSize,
    const std::map<std::string, shape_t>& requestedShapes,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Status status = validationStatus;
    if (status.batchSizeChangeRequired()) {
        try {
            status = reloadModel(requestedBatchSize, {}, modelUnloadGuardPtr);
        } catch (const std::exception& e) {
            status = Status(StatusCode::INVALID_BATCH_DIMENSION, e.what());
        }
        if (!status.ok()) {
            SPDLOG_ERROR("Model: {}, version: {} reload (batch size change) failed. Status Code: {}, Error {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (status.reshapeRequired()) {
        status = reloadModel(std::nullopt, requestedShapes, modelUnloadGuardPtr);
        if (!status.ok() && status != StatusCode::RESHAPE_ERROR) {
            SPDLOG_ERROR("Model: {}, version: {} reload (reshape) failed. Status Code: {}, Error: {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (!status.ok()) {
        SPDLOG_DEBUG("Model: {}, version: {} validation of inferRequest failed. Status Code: {}, Error: {}",
            getName(), getVersion(), status.getCode(), status.string());
    }
    return status;
}

Status ModelInstance::waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuard) {
    // order is important here for performance reasons
    // assumption: model is already loaded for most of the calls
    modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    if (getStatus().getState() == ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Model: {}, version: {} already loaded", getName(), getVersion());
        return StatusCode::OK;
    }
    modelInstanceUnloadGuard.reset();

    // wait several time since no guarantee that cv wakeup will be triggered before calling wait_for
    const uint waitLoadedTimestepMilliseconds = 100;
    const uint waitCheckpoints = waitForModelLoadedTimeoutMilliseconds / waitLoadedTimestepMilliseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    SPDLOG_DEBUG("Waiting for loaded state for model: {} version: {} with timestep: {} timeout: {} check count: {}", getName(), getVersion(),
        waitLoadedTimestepMilliseconds, waitForModelLoadedTimeoutMilliseconds, waitCheckpointsCounter);
    std::mutex cv_mtx;
    std::unique_lock<std::mutex> cv_lock(cv_mtx);
    while (waitCheckpointsCounter-- > 0) {
        if (modelLoadedNotify.wait_for(cv_lock,
                std::chrono::milliseconds(waitLoadedTimestepMilliseconds),
                [this]() {
                    return this->getStatus().getState() > ModelVersionState::LOADING;
                })) {
            SPDLOG_INFO("Waiting for model: {} version: {} loaded state for: {} time",
                getName(), getVersion(), waitCheckpoints - waitCheckpointsCounter);
        }
        modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
        if (getStatus().getState() == ModelVersionState::AVAILABLE) {
            SPDLOG_INFO("Succesfully waited for model: {}, version: {}", getName(), getVersion());
            return StatusCode::OK;
        }
        modelInstanceUnloadGuard.reset();
        if (ModelVersionState::AVAILABLE < getStatus().getState()) {
            SPDLOG_INFO("Stopped waiting for model: {} version: {} since it is unloading.", getName(), getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_INFO("Waiting for loaded state reached timeout for model: {} version: {}",
        getName(), getVersion());
    if (getStatus().getState() > ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended since it started unloading.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    } else {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended due to timeout.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
}

void ModelInstance::retireModel(bool isPermanent) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    if (isPermanent) {
        this->status.setUnloading();
    } else {
        this->status.setLoading();
    }
    unloadModelComponents();
    if (isPermanent) {
        status.setEnd();
    }
}

void ModelInstance::cleanupFailedLoad() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
    unloadModelComponents();
}

void ModelInstance::unloadModelComponents() {
    subscriptionManager.notifySubscribers();
    while (!canUnloadInstance()) {
        SPDLOG_DEBUG("Waiting to unload model: {} version: {}. Blocked by: {} inferences in progres.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    inferRequestsQueue.reset();
    compiledModel.reset();
    model.reset();
    outputsInfo.clear();
    inputsInfo.clear();
    modelFiles.clear();

    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];
        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "The loader {} is no longer available for model: {} version : {}",
                loaderName, getName(), getVersion());
        } else {
            // once model is unloaded, notify custom loader object about the unload
            customLoaderInterfacePtr->unloadModel(getName(), getVersion());
        }
    }
}

template <typename RequestType>
const Status ModelInstance::validate(const RequestType* request) {
    OVMS_PROFILE_FUNCTION();
    static const std::set<std::string> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        getInputsInfo(),
        getName(),
        getVersion(),
        optionalInputNames,
        getModelConfig().getBatchingMode(),
        getModelConfig().getShapes());
}

template const Status ModelInstance::validate(const ::inference::ModelInferRequest* request);
template const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request);

Status ModelInstance::performInference(ov::InferRequest& inferRequest) {
    OVMS_PROFILE_FUNCTION();
    try {
        OVMS_PROFILE_SYNC_BEGIN("ov::InferRequest::start_async");
        inferRequest.start_async();
        OVMS_PROFILE_SYNC_END("ov::InferRequest::start_async");
        OVMS_PROFILE_SYNC_BEGIN("ov::InferRequest::wait");
        inferRequest.wait();
        OVMS_PROFILE_SYNC_END("ov::InferRequest::wait");
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async caught an exception {}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

Status ModelInstance::infer(const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer timer;
    using std::chrono::microseconds;

    auto status = validate(requestProto);
    auto requestBatchSize = getRequestBatchSize(requestProto, this->getBatchSizeIndex());
    auto requestShapes = getRequestShapes(requestProto);
    status = reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    if (!status.ok())
        return status;
    timer.start("get infer request");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inputSink, isPipeline);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequest);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    status = serializePredictResponse(outputGetter, getOutputsInfo(), responseProto, getTensorInfoName);
    timer.stop("serialize");
    if (!status.ok())
        return status;

    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}

Status ModelInstance::infer(const ::inference::ModelInferRequest* requestProto,
    ::inference::ModelInferResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer timer;
    using std::chrono::microseconds;

    auto status = validate(requestProto);
    auto requestBatchSize = getRequestBatchSize(requestProto, this->getBatchSizeIndex());
    auto requestShapes = getRequestShapes(requestProto);
    status = reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    if (!status.ok())
        return status;
    timer.start("get infer request");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_name(), getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inputSink, isPipeline);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_name(), getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequest);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_name(), getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    status = serializePredictResponse(outputGetter, getOutputsInfo(), responseProto, getTensorInfoName);
    timer.stop("serialize");
    if (!status.ok())
        return status;

    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_name(), getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}

const size_t ModelInstance::getBatchSizeIndex() const {
    const auto& inputItr = this->inputsInfo.cbegin();
    if (inputItr == this->inputsInfo.cend()) {
        throw std::logic_error("model has no inputs");
    }
    const auto& input = inputItr->second;
    const auto batchIndex = input->getLayout().getBatchIndex();
    if (!batchIndex.has_value()) {
        throw std::logic_error("cannot get batch index");
    }
    return batchIndex.value();
}

}  // namespace ovms
