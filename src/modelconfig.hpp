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
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>

#include "layout_configuration.hpp"
#include "modelversion.hpp"
#include "shape.hpp"
#include "status.hpp"

namespace ovms {
class ModelVersionPolicy;

using mapping_config_t = std::unordered_map<std::string, std::string>;
using plugin_config_t = std::map<std::string, ov::Any>;
using custom_loader_options_config_t = std::map<std::string, std::string>;

extern const std::string ANONYMOUS_INPUT_NAME;
extern const std::string MAPPING_CONFIG_JSON;
const uint32_t DEFAULT_MAX_SEQUENCE_NUMBER = 500;

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
         * @brief Json config directory path
         */
    std::string rootDirectoryPath;

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
    std::optional<Dimension> batchSize;

    /**
         * @brief Model version policy
         */
    std::shared_ptr<ModelVersionPolicy> modelVersionPolicy;

    /**
         * @brief Nireq
         */
    uint64_t nireq;

    /**
         * @brief Flag determining if model is stateful
         */
    bool stateful;

    /**
         * @brief Flag determining if model will be a subject to sequence cleaner scans
         */
    bool idleSequenceCleanup;

    /**
         * @brief Flag determining if model will use low latency transformation
         */
    bool lowLatencyTransformation;

    /**
         * @brief Number of maximum frames in one sequence
         */
    uint32_t maxSequenceNumber;

    /**
         * @brief Model cache directory
         */
    std::string cacheDir;

    /**
         * @brief Flag determining if allow cache option is set to true
         */
    bool isAllowCacheTrue = false;

    /**
         * @brief Model version
         */
    model_version_t version = -1;

    /**
         * @brief Plugin config
         */
    plugin_config_t pluginConfig;

    /**
     * 
         * @brief Layout for single input
         */
    LayoutConfiguration layout;

    /**
         * @brief Map of shapes
         */
    shapes_info_map_t shapes;

    /**
         * @brief Map of layouts
         */
    layout_configurations_map_t layouts;

    /**
         * @brief Input mapping configuration
         */
    mapping_config_t mappingInputs;

    /**
         * @brief Output mapping configuration
         */
    mapping_config_t mappingOutputs;

    /**
         * @brief Reversed input mapping configuration
         */
    mapping_config_t reversedMappingInputs;
    /**
         * @brief Reversed output mapping configuration
         */
    mapping_config_t reversedMappingOutputs;

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
         * @brief Allowed configurable layouts
         */
    static const std::set<std::string> configAllowedLayouts;

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
        bool stateful = false,
        bool idleSequenceCleanup = true,
        bool lowLatencyTransformation = false,
        uint32_t maxSequenceNumber = DEFAULT_MAX_SEQUENCE_NUMBER,
        const std::string& cacheDir = "",
        model_version_t version = 0,
        const std::string& localPath = "");

    /**
         * @brief Compares two ModelConfig instances and decides if models should be reloaded
         * 
         * @param rhs
         *  
         * @return true if configs are equal false otherwise
         */
    bool isReloadRequired(const ModelConfig& rhs) const;

    /**
         * @brief Compares two ModelConfig instances and decides if customloader configuration changed
         *
         * @param rhs
         *
         * @return true if customloader configuration has changed
         */
    bool isCustomLoaderConfigChanged(const ModelConfig& rhs) const;

    /**
         * @brief Compares two ModelConfig instances for batch size configuration
         * 
         * @param rhs
         *  
         * @return true if configurations are equal false otherwise
         */
    bool isBatchSizeConfigurationEqual(const ModelConfig& rhs) const;

    /**
         * @brief Compares two ModelConfig instances for layout configuration
         * 
         * @param rhs
         *  
         * @return true if configurations are equal false otherwise
         */
    bool isLayoutConfigurationEqual(const ModelConfig& rhs) const;

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
    void setBasePath(const std::string& basePath);

    /**
         * @brief Set root directory path
         *
         * @param rootDirectoryPath
         */
    void setRootDirectoryPath(const std::string& rootDirectoryPath) {
        this->rootDirectoryPath = rootDirectoryPath;
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
         * @brief Get the cache directory
         * 
         * @return const std::string& 
         */
    const std::string& getCacheDir() const {
        return this->cacheDir;
    }

    /**
         * @brief Set the cache directory
         * 
         * @param cache directory
         */
    void setCacheDir(const std::string& cacheDir) {
        this->cacheDir = cacheDir;
    }

    /**
         * @brief Get the allow cache flag
         * 
         * @return bool
         */
    bool isAllowCacheSetToTrue() const {
        return this->isAllowCacheTrue;
    }

    /**
         * @brief Set the allow cache flag
         * 
         * @param allow cache flag
         */
    void setAllowCache(const bool& allowCache) {
        this->isAllowCacheTrue = allowCache;
    }

    /**
         * @brief Checks if given device is used as single target device.
         * 
         * @param bool
         */
    bool isSingleDeviceUsed(const std::string& device) const {
        return this->targetDevice == device;
    }

    bool isDeviceUsed(const std::string& device) const;

    /**
         * @brief Get the batching mode
         * 
         * @return ovms::Mode 
         */
    Mode getBatchingMode() const {
        return this->batchingMode;
    }

    bool isDynamicParameterEnabled() const {
        return this->getBatchingMode() == Mode::AUTO || this->anyShapeSetToAuto();
    }

    /**
         * @brief Get the batch size 
         * 
         * @return size_t 
         */
    std::optional<Dimension> getBatchSize() const {
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
    void setBatchSize(std::optional<Dimension> batchSize) {
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
    static std::tuple<Mode, std::optional<Dimension>> extractBatchingParams(std::string configBatchSize);

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
     * @brief Get stateful model flag
     *
     * @return bool
     */
    const bool isStateful() const {
        return this->stateful;
    }

    /**
     * @brief Set stateful model flag
     *
     * @return bool
     */
    void setStateful(bool stateful) {
        this->stateful = stateful;
    }

    /**
     * @brief Set stateful low latency transformation flag
     *
     * @return bool
     */
    void setLowLatencyTransformation(bool lowLatencyTransformation) {
        this->lowLatencyTransformation = lowLatencyTransformation;
    }

    /**
     * @brief Get stateful low latency transformation flag
     *
     * @return bool
     */
    const bool isLowLatencyTransformationUsed() const {
        return this->lowLatencyTransformation;
    }

    /**
     * @brief Get max number of sequences handled concurrently by the model
     *
     * @return uint
     */
    uint64_t getMaxSequenceNumber() const {
        return this->maxSequenceNumber;
    }

    /**
     * @brief Set max number of sequences handled concurrently by the model
     *
     * @return uint
     */
    void setMaxSequenceNumber(const uint32_t maxSequenceNumber) {
        this->maxSequenceNumber = maxSequenceNumber;
    }

    /**
     * @brief Get stateful sequence timeout
     *
     * @return uint
     */
    bool getIdleSequenceCleanup() const {
        return this->idleSequenceCleanup;
    }

    /**
     * @brief Set stateful sequence timeout
     *
     * @return uint
     */
    void setIdleSequenceCleanup(const bool idleSequenceCleanup) {
        this->idleSequenceCleanup = idleSequenceCleanup;
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
         * @brief Parses value from json and extracts layouts info
         * 
         * @param rapidjson::Value& node
         * 
         * @return status
         */
    Status parseLayoutParameter(const rapidjson::Value& node);

    /**
         * @brief Parses value from string and extracts layouts info
         * 
         * @param string
         * 
         * @return status
         */
    Status parseLayoutParameter(const std::string& command);

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
    const shapes_info_map_t& getShapes() const {
        return this->shapes;
    }

    /**
         * @brief Set the shapes
         * 
         * @param shapes 
         */
    void setShapes(const shapes_info_map_t& shapes) {
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

    bool isCloudStored() const {
        return getLocalPath() != getBasePath();
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
    const LayoutConfiguration& getLayout() const {
        return this->layout;
    }

    /**
         * @brief Set the layout
         * 
         * @param layout
         */
    void setLayout(const LayoutConfiguration& layout) {
        this->layout = layout;
        this->layouts.clear();
    }

    /**
         * @brief Get the layouts
         * 
         * @return const layout_configurations_map_t& 
         */
    const layout_configurations_map_t& getLayouts() const {
        return this->layouts;
    }

    /**
         * @brief Set the layouts
         * 
         * @param layouts 
         */
    void setLayouts(const layout_configurations_map_t& layouts) {
        this->layouts = layouts;
        this->layout = LayoutConfiguration();
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
         * @brief Get the reversed mapping for inputs
         * 
         * @return const mapping_config_t& 
         */
    const mapping_config_t& getRealMappingInputs() const {
        return this->reversedMappingInputs;
    }

    /**
         * @brief Get the reversed mapping for outputs
         * 
         * @return const mapping_config_t& 
         */
    const mapping_config_t& getRealMappingOutputs() const {
        return this->reversedMappingOutputs;
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
         * @brief Get the real inputs by value
         * 
         * @param value 
         * @return const std::string 
         */
    const std::string getRealInputNameByValue(const std::string& value) const {
        auto it = reversedMappingInputs.find(value);
        return it != reversedMappingInputs.end() ? it->second : "";
    }

    /**
         * @brief Get the real outputs by value
         * 
         * @param value 
         * @return const std::string 
         */
    const std::string getRealOutputNameByValue(const std::string& value) const {
        auto it = reversedMappingOutputs.find(value);
        return it != reversedMappingOutputs.end() ? it->second : "";
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
         * @brief Set the reversed mapping inputs
         * 
         * @param mapping 
         */
    void setRealMappingInputs(const mapping_config_t& mapping) {
        this->reversedMappingInputs = mapping;
    }

    /**
         * @brief Set the reversed mapping outputs
         * 
         * @param mapping 
         */
    void setRealMappingOutputs(const mapping_config_t& mapping) {
        this->reversedMappingOutputs = mapping;
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
      * @brief Add custom loader option
      * 
      * @param name
      * @param value
      */
    void addCustomLoaderOption(const std::string& name, const std::string& value) {
        customLoaderOptionsConfigMap[name] = value;
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

    std::string layoutConfigurationToString() const;
};
}  // namespace ovms
