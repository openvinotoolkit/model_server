//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>

#include "model_version_policy.hpp"
#include "status.hpp"

namespace ovms {

enum Mode { FIXED,
    AUTO };
using shape_t = std::vector<size_t>;

struct ShapeInfo {
    Mode shapeMode = FIXED;
    shape_t shape;

    bool operator==(const ShapeInfo& rhs) const {
        return this->shapeMode == rhs.shapeMode && this->shape == rhs.shape;
    }

    bool operator!=(const ShapeInfo& rhs) const {
        return !(*this == rhs);
    }
};

using shapes_map_t = std::unordered_map<std::string, ShapeInfo>;
using layouts_map_t = std::unordered_map<std::string, std::string>;
using mapping_config_t = std::unordered_map<std::string, std::string>;
using plugin_config_t = std::map<std::string, std::string>;
using custom_loader_options_config_t = std::map<std::string, std::string>;

const std::string ANONYMOUS_INPUT_NAME = "ANONYMOUS_INPUT_NAME";
const std::string MAPPING_CONFIG_JSON = "mapping_config.json";

/**
     * @brief This class represents model configuration
     */
class ModelConfig {
private:
    /**
         * @brief Model name
         */
    std::string name;

    /**
         * @brief Model uri path
         */
    std::string basePath;

    /**
         * @brief Model local destination path on disk after downloading from online storage
         */
    std::string localPath;

    /**
         * @brief Target device
         */
    std::string targetDevice;

    /**
         * @brief Batching mode
         */
    Mode batchingMode;

    /**
         * @brief Batch size
         */
    size_t batchSize;

    /**
         * @brief Model version policy
         */
    std::shared_ptr<ModelVersionPolicy> modelVersionPolicy;

    /**
         * @brief Nireq
         */
    uint64_t nireq;

    /**
         * @brief Plugin config
         */
    plugin_config_t pluginConfig;

    /**
         * @brief Layout for single input
         */
    std::string layout;

    /**
         * @brief Map of shapes
         */
    shapes_map_t shapes;

    /**
         * @brief Map of layouts
         */
    layouts_map_t layouts;

    /**
         * @brief Model version
         */
    model_version_t version = -1;

    /**
         * @brief Input mapping configuration
         */
    mapping_config_t mappingInputs;

    /**
         * @brief Input mapping configuration
         */
    mapping_config_t mappingOutputs;

    /**
         * @brief Shape left opening bracket in string format
         */
    static const char shapeLeft = '(';

    /**
         * @brief Shape right opening bracket in string format
         */
    static const char shapeRight = ')';

    /**
         * @brief Shape delimeter in string format
         */
    static const char shapeDelimeter = ',';

    /**
         * @brief custom_loader_options config as map
         */
    custom_loader_options_config_t customLoaderOptionsConfigMap;

    /**
         * @brief custom_loader_options config as string
         */
    std::string customLoaderOptionsStr;

public:
    /**
         * @brief Construct a new Model Config object
         * 
         * @param name 
         * @param basePath 
         * @param targetDevice 
         * @param configBatchSize 
         * @param nireq 
         */
    ModelConfig(const std::string& name = "",
        const std::string& basePath = "",
        const std::string& targetDevice = "CPU",
        const std::string& configBatchSize = "0",
        uint64_t nireq = 0,
        model_version_t version = 0,
        const std::string& localPath = "") :
        name(name),
        basePath(basePath),
        localPath(localPath),
        targetDevice(targetDevice),
        modelVersionPolicy(ModelVersionPolicy::getDefaultVersionPolicy()),
        nireq(nireq),
        pluginConfig({}),
        layout(""),
        shapes({}),
        layouts({}),
        version(version),
        mappingInputs({}),
        mappingOutputs({}) {
        setBatchingParams(configBatchSize);
    }

    /**
         * @brief Compares two ModelConfig instances and decides if models should be reloaded
         * 
         * @param rhs
         *  
         * @return true if configs are equal false otherwise
         */
    bool isReloadRequired(const ModelConfig& rhs) const;

    /**
         * @brief Compares two ModelConfig instances for shape configuration
         * 
         * @param rhs
         *  
         * @return true if configurations are equal false otherwise
         */
    bool isShapeConfigurationEqual(const ModelConfig& rhs) const;

    /**
         * @brief Get the name 
         * 
         * @return const std::string& 
         */
    const std::string& getName() const {
        return this->name;
    }

    /**
         * @brief Set the name
         * 
         * @param name 
         */
    void setName(const std::string& name) {
        this->name = name;
    }

    /**
         * @brief Get local path to specific model version where .xml and .bin is located for loading
         * 
         * @return std::string
         * */
    const std::string getPath() const {
        return getLocalPath() + "/" + std::to_string(version);
    }

    /**
         * @brief Get the base path
         * 
         * @return const std::string& 
         */
    const std::string& getBasePath() const {
        return this->basePath;
    }

    /**
         * @brief Set the base path
         * 
         * @param basePath 
         */
    void setBasePath(const std::string& basePath) {
        this->basePath = basePath;
    }

    /**
         * @brief Get the local path
         * 
         * @return const std::string& 
         */
    const std::string& getLocalPath() const {
        return this->localPath;
    }

    /**
         * @brief Set the local path
         * 
         * @param localPath 
         */
    void setLocalPath(const std::string& localPath) {
        this->localPath = localPath;
    }

    /**
         * @brief Get the target device
         * 
         * @return const std::string& 
         */
    const std::string& getTargetDevice() const {
        return this->targetDevice;
    }

    /**
         * @brief Set the target device
         * 
         * @param target device 
         */
    void setTargetDevice(const std::string& targetDevice) {
        this->targetDevice = targetDevice;
    }

    /**
         * @brief Checks if target device is heterogeneous and contains specific device
         * 
         * @param bool
         */
    bool isHeteroTargetDevice(const std::string& device) const {
        return this->targetDevice.find("HETERO") != std::string::npos && this->targetDevice.find(device) != std::string::npos;
    }

    /**
         * @brief Checks if given device name is used alone or as a part of multi device configuration
         * 
         * @param bool
         */
    bool isDeviceUsed(const std::string& device) const {
        return this->targetDevice == device || this->isHeteroTargetDevice(device);
    }

    /**
         * @brief Get the batching mode
         * 
         * @return ovms::Mode 
         */
    Mode getBatchingMode() const {
        return this->batchingMode;
    }

    /**
         * @brief Get the batch size 
         * 
         * @return size_t 
         */
    size_t getBatchSize() const {
        return this->batchSize;
    }

    /**
         * @brief Set the batching mode
         * 
         * @param batchingMode 
         */
    void setBatchingMode(Mode batchingMode) {
        this->batchingMode = batchingMode;
    }

    /**
         * @brief Set the batch size
         * 
         * @param batchSize 
         */
    void setBatchSize(size_t batchSize) {
        this->batchSize = batchSize;
    }

    /**
         * @brief Set batching mode to FIXED and batch size to value provided in a parameter. 
         * 
         * @param configBatchSize 
         */
    void setBatchingParams(size_t configBatchSize) {
        setBatchingMode(FIXED);
        setBatchSize(configBatchSize);
    }

    /**
         * @brief Extracts batching mode and batch size value from string provided in a parameter.
         * 
         * @param configBatchSize 
         */
    void setBatchingParams(const std::string& configBatchSize) {
        auto [batchingMode, effectiveBatchSize] = extractBatchingParams(configBatchSize);
        setBatchingMode(batchingMode);
        setBatchSize(effectiveBatchSize);
    }

    /**
         * @brief Extract batching mode and effective batch size from string.
         * 
         * @param configBatchSize 
         */
    static std::tuple<Mode, size_t> extractBatchingParams(std::string configBatchSize);

    /**
         * @brief Get the model version policy
         * 
         * @return std::shared_ptr<ModelVersionPolicy>
         */
    std::shared_ptr<ModelVersionPolicy> getModelVersionPolicy() {
        return this->modelVersionPolicy;
    }

    /**
         * @brief Set the model version policy
         * 
         * @param modelVersionPolicy 
         */
    void setModelVersionPolicy(std::shared_ptr<ModelVersionPolicy> modelVersionPolicy) {
        this->modelVersionPolicy = modelVersionPolicy;
    }

    /**
         * @brief Parses string for model version policy
         * 
         * @param string representing model version policy configuration

         * @return status
         */
    Status parseModelVersionPolicy(std::string command);

    /**
         * @brief Get the nireq
         * 
         * @return uint64_t 
         */
    uint64_t getNireq() const {
        return this->nireq;
    }

    /**
         * @brief Set the nireq
         * 
         * @param nireq 
         */
    void setNireq(const uint64_t nireq) {
        this->nireq = nireq;
    }

    /**
         * @brief Get the plugin config
         * 
         * @return const std::string&
         */
    const plugin_config_t& getPluginConfig() const {
        return this->pluginConfig;
    }

    /**
         * @brief Set the plugin config
         * 
         * @param pluginConfig 
         */
    void setPluginConfig(const plugin_config_t& pluginConfig) {
        this->pluginConfig = pluginConfig;
    }

    /**
         * @brief Parses json node for plugin config keys and values
         * 
         * @param json node representing plugin_config
         * 
         * @return status
         */
    Status parsePluginConfig(const rapidjson::Value& node);

    /**
         * @brief Parses string for plugin config keys and values
         * 
         * @param string representing plugin_config
         * 
         * @return status
         */
    Status parsePluginConfig(std::string command) {
        rapidjson::Document node;
        if (command.empty()) {
            return StatusCode::OK;
        }
        if (node.Parse(command.c_str()).HasParseError()) {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }

        return parsePluginConfig(node);
    }

    /**
         * @brief Parses value from json and extracts shapes info
         * 
         * @param rapidjson::Value& node
         * 
         * @return status
         */
    Status parseShapeParameter(const rapidjson::Value& node);

    /**
         * @brief Parses value from string and extracts shapes info
         * 
         * @param string
         * 
         * @return status
         */
    Status parseShapeParameter(const std::string& command);

    /**
         * @brief Returns true if any input shape specified in shapes map is in AUTO mode
         * 
         * @return bool
         */
    bool anyShapeSetToAuto() const {
        for (const auto& [name, shapeInfo] : getShapes()) {
            if (shapeInfo.shapeMode == AUTO)
                return true;
        }
        return false;
    }

    /**
         * @brief Get the shapes
         * 
         * @return const shapes_map_t& 
         */
    const shapes_map_t& getShapes() const {
        return this->shapes;
    }

    /**
         * @brief Set the shapes
         * 
         * @param shapes 
         */
    void setShapes(const shapes_map_t& shapes) {
        this->shapes = shapes;
    }

    /**
        * @brief Returns true if shape with certain name is in AUTO mode
         * 
         * @return bool
         */
    bool isShapeAuto(const std::string& name) const {
        auto it = getShapes().find(name);
        if (it == getShapes().end()) {
            it = getShapes().find(ANONYMOUS_INPUT_NAME);
        }
        if (it == getShapes().end()) {
            return false;
        }
        return it->second.shapeMode == Mode::AUTO;
    }

    bool isShapeAnonymous() const {
        return getShapes().size() == 1 && getShapes().begin()->first == ANONYMOUS_INPUT_NAME;
    }

    bool isShapeAnonymousFixed() const {
        return isShapeAnonymous() && !isShapeAuto(ANONYMOUS_INPUT_NAME);
    }

    /**
         * @brief Sets the shape from the string representation
         *
         * @param shape
         * @param str
         * @return Status
         */
    static Status parseShape(ShapeInfo& shapeInfo, const std::string& str);

    /**
         * @brief Add a single named shape
         * 
         * @param name 
         * @param shape 
         */
    void addShape(const std::string& name, const ShapeInfo& shapeInfo) {
        this->shapes[name] = shapeInfo;
    }

    void removeShape(const std::string& name) {
        this->shapes.erase(name);
    }

    /**
         * @brief Get the layouts
         * 
         * @return const std::string& 
         */
    const std::string& getLayout() const {
        return this->layout;
    }

    /**
         * @brief Set the layout
         * 
         * @param layout
         */
    void setLayout(const std::string& layout) {
        this->layout = layout;
        this->layouts.clear();
    }

    /**
         * @brief Get the layouts
         * 
         * @return const layouts_map_t& 
         */
    const layouts_map_t& getLayouts() const {
        return this->layouts;
    }

    /**
         * @brief Set the layouts
         * 
         * @param layouts 
         */
    void setLayouts(const layouts_map_t& layouts) {
        this->layouts = layouts;
        this->layout = "";
    }

    /**
         * @brief Add a named layout
         * 
         * @param name 
         * @param layout 
         */
    void addLayout(const std::string& name, const std::string& layout) {
        this->layouts[name] = layout;
        this->layout = "";
    }

    /**
         * @brief Get the version
         * 
         * @return const model_version_t& 
         */
    const model_version_t& getVersion() const {
        return this->version;
    }

    /**
         * @brief Set the version
         * 
         * @param version 
         */
    void setVersion(const model_version_t& version) {
        this->version = version;
    }

    /**
         * @brief Get the mapping for inputs
         * 
         * @return const mapping_config_t& 
         */
    const mapping_config_t& getMappingInputs() const {
        return this->mappingInputs;
    }

    /**
         * @brief Get the mapping for outputs
         * 
         * @return const mapping_config_t& 
         */
    const mapping_config_t& getMappingOutputs() const {
        return this->mappingOutputs;
    }

    /**
         * @brief Get the mapping inputs by key
         * 
         * @param key 
         * @return const std::string 
         */
    const std::string getMappingInputByKey(const std::string& key) const {
        auto it = mappingInputs.find(key);
        return it != mappingInputs.end() ? it->second : "";
    }

    /**
         * @brief Get the mapping outputs by key
         * 
         * @param key 
         * @return const std::string 
         */
    const std::string getMappingOutputByKey(const std::string& key) const {
        auto it = mappingOutputs.find(key);
        return it != mappingOutputs.end() ? it->second : "";
    }

    /**
         * @brief Set the mapping inputs
         * 
         * @param mapping 
         */
    void setMappingInputs(const mapping_config_t& mapping) {
        this->mappingInputs = mapping;
    }

    /**
         * @brief Set the mapping outputs
         * 
         * @param mapping 
         */
    void setMappingOutputs(const mapping_config_t& mapping) {
        this->mappingOutputs = mapping;
    }

    /**
         * @brief  Parses mapping_config.json for mapping input/outputs in the model
         * 
         * @return Status 
         */
    Status parseModelMapping();

    /**
     * @brief  Parses all settings from a JSON node
        * 
        * @return Status 
        */
    Status parseNode(const rapidjson::Value& v);

    /**
        * @brief Returns true if model requires a custom loader to load
         *
         * @return bool
         */
    const bool isCustomLoaderRequiredToLoadModel() const {
        return (this->customLoaderOptionsConfigMap.size() > 0);
    }

    /**
         * @brief Get the custom loader option config
         *
         * @return const custom_loader_options_config_t&
         */
    const custom_loader_options_config_t& getCustomLoaderOptionsConfigMap() const {
        return this->customLoaderOptionsConfigMap;
    }

    /**
         * @brief Get the custom loader option config
         *
         * @return const std::string
         */
    const std::string& getCustomLoaderOptionsConfigStr() const {
        return this->customLoaderOptionsStr;
    }

    /**
         * @brief Parses json node for custom_loader_options config keys and values
         *
         * @param json node representing custom_loader_options_config
         *
         * @return status
         */
    Status parseCustomLoaderOptionsConfig(const rapidjson::Value& node);
};
}  // namespace ovms
