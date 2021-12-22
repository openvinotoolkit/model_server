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
#include "logging.hpp"
#include "ov_utils.hpp"
#include "predict_request_validation_utils.hpp"
#include "prediction_service_utils.hpp"
#include "serialization.hpp"
#include "shape.hpp"
#include "stringutils.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"

using namespace InferenceEngine;

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
    } else if (mappedName == "" && config.getShapes_2().count(name) && config.getShapes_2().at(name).shape.size()) {
        shape = config.getShapes_2().at(name).shape;
    } else if (config.getShapes_2().count(mappedName) && config.getShapes_2().at(mappedName).shape.size()) {
        shape = config.getShapes_2().at(mappedName).shape;
    } else if (config.getShapes_2().count(ANONYMOUS_INPUT_NAME) && config.getShapes_2().at(ANONYMOUS_INPUT_NAME).shape.size()) {
        shape = config.getShapes_2().at(ANONYMOUS_INPUT_NAME).shape;
    }
    shapeOut = shape;
    return StatusCode::OK;
}

bool hasInputWithName(std::shared_ptr<ov::Function>& network, const std::string& name) {
    try {
        network->input(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

bool hasOutputWithName(std::shared_ptr<ov::Function>& network, const std::string& name) {
    try {
        network->output(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

Status validateConfigurationAgainstNetwork(const ModelConfig& config, std::shared_ptr<ov::Function>& network) {
    if (config.isShapeAnonymousFixed() && network->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    if (config.getLayout_2().isSet() && network->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_LAYOUT_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    for (const auto& [name, _] : config.getShapes_2()) {
        if (name == ANONYMOUS_INPUT_NAME) {
            continue;
        }
        if (hasInputWithName(network, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(network, name) && !hasInputWithName(network, config.getRealInputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} not found in network", name);
            return StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK;
        }
    }
    for (const auto& [name, _] : config.getLayouts_2()) {
        if (hasInputWithName(network, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (hasOutputWithName(network, name) && config.getMappingOutputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingOutputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(network, name) && !hasOutputWithName(network, name) && !hasInputWithName(network, config.getRealInputNameByValue(name)) && !hasOutputWithName(network, config.getRealOutputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} not found in network", name);
            return StatusCode::CONFIG_LAYOUT_IS_NOT_IN_NETWORK;
        }
    }
    return StatusCode::OK;
}

InferenceEngine::Layout getReportedTensorLayout(const ModelConfig& config, const std::string& name) {
    InferenceEngine::Layout layout = InferenceEngine::Layout::ANY;
    if (config.getLayout_2().isSet()) {
        layout = TensorInfo::getLayoutFromString(config.getLayout_2().getTensorLayout());
    } else if (config.getLayouts_2().size() > 0) {
        auto mappedName = config.getMappingInputByKey(name);
        auto it = config.getLayouts_2().find(mappedName == "" ? name : mappedName);
        if (it != config.getLayouts_2().end()) {
            layout = TensorInfo::getLayoutFromString(it->second.getTensorLayout());
        }
    }
    return layout;
}

Status applyLayoutConfiguration(const ModelConfig& config, std::shared_ptr<ov::Function>& network, const std::string& modelName, model_version_t modelVersion) {
    ov::preprocess::PrePostProcessor preproc(network);

    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Applying layout configuration: {}", config.layoutConfigurationToString());

    for (const ov::Output<ov::Node>& input : network->inputs()) {
        try {
            std::string name = input.get_any_name();

            if (config.getLayout_2().isSet()) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; single input",
                    modelName,
                    modelVersion,
                    config.getLayout_2().getTensorLayout(),
                    config.getLayout_2().getModelLayout());

                // TODO: Validate rank vs layout string len?
                preproc.input().tensor().set_layout(ov::Layout(config.getLayout_2().getTensorLayout()));
                preproc.input().model().set_layout(ov::Layout(config.getLayout_2().getModelLayout()));
            } else if (config.getLayouts_2().count(name) > 0) {
                auto& layout = config.getLayouts_2().at(name);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; input name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    name);

                // TODO: Validate rank vs layout string len?
                preproc.input(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                preproc.input(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            } else {
                std::string guessedModelLayout{"N..."};
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding auto preprocessing step: Tensor Layout:; Network Layout:{}; input name: {}",
                    modelName,
                    modelVersion,
                    guessedModelLayout,
                    name);
                preproc.input(name).model().set_layout(ov::Layout(guessedModelLayout));
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

    for (const ov::Output<ov::Node>& output : network->outputs()) {
        try {
            std::string name = output.get_any_name();

            if (config.getLayouts_2().count(name) > 0) {
                auto& layout = config.getLayouts_2().at(name);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding postprocessing step: Tensor Layout:{}; Network Layout:{}; output name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    name);
                preproc.output(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                preproc.output(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            // TODO potentially allow for empty names if OV will load such model. Then potentially use empty string as input/output names
            // and adjust validation, metadata, dags for that
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
        network = preproc.build();
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot change layout");
        return StatusCode::NETWORK_NOT_LOADED;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadTensors(const ModelConfig& config, const DynamicModelParameter& parameter) {
    Status status = validateConfigurationAgainstNetwork(config, this->network_2);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during configuration validation against network");
        return status;
    }
    status = applyLayoutConfiguration(config, this->network_2, getName(), getVersion());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during layout configuration");
        return status;
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

    std::map<std::string, ov::PartialShape> networkShapes_2;
    bool reshapeRequired = false;

    // First pass, gather reshape info.
    for (const ov::Output<ov::Node>& input : this->network_2->inputs()) {
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

            networkShapes_2[name] = shape;
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
            network_2->reshape(networkShapes_2);
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

    for (const ov::Output<ov::Node>& input : this->network_2->inputs()) {
        try {
            std::string name = input.get_any_name();

            ovms::Precision precision = ovElementTypeToOvmsPrecision(input.get_element_type());
            Shape shape(input.get_partial_shape());
            std::string mappingName = config.getMappingInputByKey(name);
            InferenceEngine::Layout layout = getReportedTensorLayout(config, name);

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
            // TODO potentially allow for empty names if OV will load such model. Then potentially use empty string as input/output names
            // and adjust validation, metadata, dags for that
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

    for (const ov::Output<ov::Node>& output : this->network_2->outputs()) {
        try {
            std::string name = output.get_any_name();

            ovms::Precision precision = ovElementTypeToOvmsPrecision(output.get_element_type());
            Shape shape(output.get_partial_shape());
            std::string mappingName = config.getMappingOutputByKey(name);
            InferenceEngine::Layout layout = getReportedTensorLayout(config, name);

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
            // TODO potentially allow for empty names if OV will load such model. Then potentially use empty string as input/output names
            // and adjust validation, metadata, dags for that
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
        numberOfParallelInferRequests = execNetwork->GetMetric(key).as<unsigned int>();
    } catch (const InferenceEngine::Exception& ex) {
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

std::unique_ptr<InferenceEngine::CNNNetwork> ModelInstance::loadOVCNNNetworkPtr(const std::string& modelFile) {
    return std::make_unique<InferenceEngine::CNNNetwork>(ieCore.ReadNetwork(modelFile));
}

Status ModelInstance::loadOVCNNNetwork() {
    auto& modelFile = modelFiles[0];
    SPDLOG_DEBUG("Try reading model file: {}", modelFile);
    try {
        network = loadOVCNNNetworkPtr(modelFile);
        network_2 = ieCore_2.read_model(modelFile);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading CNNNetwork for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOVCNNNetworkUsingCustomLoader() {
    SPDLOG_DEBUG("Try reading model using a custom loader");
    try {
        std::vector<uint8_t> model;
        std::vector<uint8_t> weights;

        SPDLOG_INFO("loading CNNNetwork for model: {} basepath: {} <> {} version: {}", getName(), getPath(), this->config.getBasePath().c_str(), getVersion());

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
            this->config.getCustomLoaderOptionsConfigStr(), model, weights);

        if (res == CustomLoaderStatus::MODEL_LOAD_ERROR) {
            return StatusCode::FILE_INVALID;
        }

        if ((res == CustomLoaderStatus::INTERNAL_ERROR) || (res == CustomLoaderStatus::MODEL_BLACKLISTED)) {
            return StatusCode::INTERNAL_ERROR;
        }

        std::string strModel(model.begin(), model.end());

        if (res == CustomLoaderStatus::MODEL_TYPE_IR) {
            Blob::Ptr blobWts = make_shared_blob<uint8_t>({InferenceEngine::Precision::U8, {weights.size()}, C});
            ov::runtime::Tensor tensorWts(ov::element::u8, ov::Shape{weights.size()});
            (void)tensorWts;
            blobWts->allocate();
            // dont need to allocate
            std::memcpy(InferenceEngine::as<InferenceEngine::MemoryBlob>(blobWts)->wmap(), weights.data(), weights.size());
            std::memcpy(tensorWts.data(), weights.data(), weights.size());
            network = std::make_unique<InferenceEngine::CNNNetwork>(ieCore.ReadNetwork(strModel, blobWts));
            network_2 = ieCore_2.read_model(strModel, tensorWts);
        } else if (res == CustomLoaderStatus::MODEL_TYPE_ONNX) {
            network = std::make_unique<InferenceEngine::CNNNetwork>(ieCore.ReadNetwork(strModel, InferenceEngine::Blob::CPtr()));
            network_2 = ieCore_2.read_model(strModel, ov::runtime::Tensor());
        } else if (res == CustomLoaderStatus::MODEL_TYPE_BLOB) {
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (ov::Exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading CNNNetwork for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading CNNNetwork for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

void ModelInstance::loadExecutableNetworkPtr(const plugin_config_t& pluginConfig) {
    execNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(ieCore.LoadNetwork(*network, targetDevice, pluginConfig));
    execNetwork_2 = std::make_shared<ov::runtime::ExecutableNetwork>(ieCore_2.compile_model(network_2, targetDevice, pluginConfig));
}

plugin_config_t ModelInstance::prepareDefaultPluginConfig(const ModelConfig& config) {
    plugin_config_t pluginConfig = config.getPluginConfig();
    // For CPU and GPU, if user did not specify, calculate CPU_THROUGHPUT_STREAMS automatically
    if (config.isDeviceUsed("CPU")) {
        if (pluginConfig.count("CPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["CPU_THROUGHPUT_STREAMS"] = "CPU_THROUGHPUT_AUTO";
        }
    }
    if (config.isDeviceUsed("GPU")) {
        if (pluginConfig.count("GPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO";
        }
    }
    return pluginConfig;
}

Status ModelInstance::loadOVExecutableNetwork(const ModelConfig& config) {
    plugin_config_t pluginConfig = prepareDefaultPluginConfig(config);
    try {
        loadExecutableNetworkPtr(pluginConfig);
    } catch (ov::Exception& e) {
        Status status = StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    } catch (std::exception& e) {
        Status status = StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Plugin config for device {}:", targetDevice);
    for (const auto pair : pluginConfig) {
        const auto key = pair.first;
        const auto value = pair.second;
        SPDLOG_LOGGER_INFO(modelmanager_logger, "OVMS set plugin settings key:{}; value:{};", key, value);
    }

    const std::string supportedConfigKey = METRIC_KEY(SUPPORTED_CONFIG_KEYS);
    std::vector<std::string> supportedConfigKeys;
    try {
        std::vector<std::string> supportedConfigKeys2 = execNetwork->GetMetric(supportedConfigKey);
        supportedConfigKeys = std::move(supportedConfigKeys2);
    } catch (std::exception& e) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, ExecutableNetwork metric key: {}; Error: {}", targetDevice, supportedConfigKey, e.what());
        return StatusCode::OK;
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, ExecutableNetwork metric key: {}", targetDevice, supportedConfigKey);
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Logging model:{}; version {};target device: {}; ExecutableNetwork configuration", getName(), getVersion(), targetDevice);
    for (auto& key : supportedConfigKeys) {
        std::string value;
        try {
            auto paramValue = execNetwork->GetConfig(key);
            value = paramValue.as<std::string>();
        } catch (std::exception& e) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, ExecutableNetwork config key: {}; Error: {}", targetDevice, key, e.what());
            continue;
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting target device: {}, ExecutableNetwork config key: {}", targetDevice, key);
            continue;
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {}; version: {}; target device: {}, ExecutableNetwork config key: {}, value :{}", getName(), getVersion(), targetDevice, key, value);
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
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(*execNetwork, numberOfParallelInferRequests);
    inferRequestsQueue_2 = std::make_unique<OVInferRequestsQueue_2>(*execNetwork_2, numberOfParallelInferRequests);
    SPDLOG_INFO("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
        getName(),
        getVersion(),
        getBatchSize(),
        numberOfParallelInferRequests);
    return StatusCode::OK;
}

void ModelInstance::configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter) {
    if (parameter.isBatchSizeRequested()) {
        ov::set_batch(network_2, parameter.getBatchSize());
    } else if (config.getBatchSize().has_value()) {
        ov::set_batch(network_2, config.getBatchSize().value().createPartialDimension());
    }
}

Status ModelInstance::loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter) {
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
        if (!this->config.getCacheDir().empty()) {
            if (this->config.isCachingDisabled()) {
                this->ieCore_2.set_config({{CONFIG_KEY(CACHE_DIR), ""}});
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has disabled caching", this->getName());
            } else {
                this->ieCore_2.set_config({{CONFIG_KEY(CACHE_DIR), config.getCacheDir()}});
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has enabled caching", this->getName());
            }
        }

        if (!this->network) {
            if (this->config.isCustomLoaderRequiredToLoadModel()) {
                // loading the model using the custom loader
                status = loadOVCNNNetworkUsingCustomLoader();
            } else {
                status = loadOVCNNNetwork();
            }
        }

        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        status = loadTensors(this->config, parameter);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = loadOVExecutableNetwork(this->config);
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
        SPDLOG_ERROR("exception occurred while loading network: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::NETWORK_NOT_LOADED;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("exception occurred while loading network: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::NETWORK_NOT_LOADED;
    }
    this->status.setAvailable();
    modelLoadedNotify.notify_all();
    return status;
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
    const tensorflow::serving::PredictRequest* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Status status = validationStatus;
    if (status.batchSizeChangeRequired()) {
        status = reloadModel(getRequestBatchSize(requestProto), {}, modelUnloadGuardPtr);
        if (!status.ok()) {
            SPDLOG_ERROR("Model: {}, version: {} reload (batch size change) failed. Status Code: {}, Error {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (status.reshapeRequired()) {
        status = reloadModel(std::nullopt, getRequestShapes(requestProto), modelUnloadGuardPtr);
        if (!status.ok() && status != StatusCode::RESHAPE_ERROR) {
            SPDLOG_ERROR("Model: {}, version: {} reload (reshape) failed. Status Code: {}, Error: {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (!status.ok()) {
        SPDLOG_WARN("Model: {}, version: {} validation of inferRequest failed. Status Code: {}, Error: {}",
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
    execNetwork.reset();
    network.reset();
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

const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {
    static const std::set<const char*> optionalInputNames = {};
    return request_validation_utils::validate(
        *request,
        getInputsInfo(),
        getName(),
        getVersion(),
        optionalInputNames,
        getModelConfig().getBatchingMode(),
        getModelConfig().getShapes_2());
}

Status ModelInstance::performInference(InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.StartAsync();
        InferenceEngine::StatusCode sts = inferRequest.Wait(InferenceEngine::IInferRequest::RESULT_READY);
        if (sts != InferenceEngine::StatusCode::OK) {
            Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
            SPDLOG_ERROR("Async infer failed {}: {}", status.string(), sts);
            return status;
        }
    } catch (const InferenceEngine::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async caught an exception {}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

Status ModelInstance::performInference_2(ov::runtime::InferRequest& inferRequest) {
    try {
        inferRequest.start_async();
        inferRequest.wait();
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
    Timer timer;
    using std::chrono::microseconds;

    auto status = validate(requestProto);
    status = reloadModelIfRequired(status, requestProto, modelUnloadGuardPtr);
    if (!status.ok())
        return status;
    timer.start("get infer request");
    ExecutingStreamIdGuard_2 executingStreamIdGuard_2(getInferRequestsQueue_2());
    int executingInferId_2 = executingStreamIdGuard_2.getId();
    ov::runtime::InferRequest& inferRequest_2 = executingStreamIdGuard_2.getInferRequest();
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId_2, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    InputSink_2<ov::runtime::InferRequest&> inputSink_2(inferRequest_2);
    bool isPipeline = false;
    status = deserializePredictRequest_2<ConcreteTensorProtoDeserializator_2>(*requestProto, getInputsInfo(), inputSink_2, isPipeline);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId_2, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference_2(inferRequest_2);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId_2, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse_2(inferRequest_2, getOutputsInfo(), responseProto);
    timer.stop("serialize");
    if (!status.ok())
        return status;

    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId_2, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}
}  // namespace ovms
