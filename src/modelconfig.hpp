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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "model_version_policy.hpp"
#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

using shape_t = std::vector<size_t>;
using shapes_map_t = std::unordered_map<std::string, shape_t>;
using layouts_map_t = std::unordered_map<std::string, std::string>;
using mapping_config_t = std::unordered_map<std::string, std::string>;
using plugin_config_t = std::map<std::string, std::string>;

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
         * @brief Device backend
         */
        std::string backend;

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
         * @brief Shape for single input
         */
        shape_t shape;

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
        model_version_t version;

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

    public:
        /**
         * @brief Construct a new ModelConfig with default values
         */
        ModelConfig() :
            name(""),
            basePath(""),
            backend("CPU"),
            batchSize(0),
            modelVersionPolicy({}),
            nireq(1),
            pluginConfig({}),
            shape({}),
            layout(""),
            shapes({}),
            layouts({}),
            version(0),
            mappingInputs({}),
            mappingOutputs({})
            {}

        /**
         * @brief Construct a new Model Config object
         * 
         * @param name 
         * @param basePath 
         * @param backend 
         * @param batchSize 
         * @param nireq 
         */
        ModelConfig(const std::string& name,
                    const std::string& basePath,
                    const std::string& backend,
                    size_t batchSize,
                    uint64_t nireq) :
                    name(name),
                    basePath(basePath),
                    backend(backend),
                    batchSize(batchSize),
                    modelVersionPolicy({}),
                    nireq(nireq),
                    pluginConfig({}),
                    shape({}),
                    layout(""),
                    shapes({}),
                    layouts({}),
                    version(0),
                    mappingInputs({}),
                    mappingOutputs({})
                    {}

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
         * @brief Get model path
         * 
         * @return std::string
         * */
        const std::string getPath() const {
            return getBasePath() + "/" + std::to_string(version);
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
         * @brief Get the backend
         * 
         * @return const std::string& 
         */
        const std::string& getBackend() const {
            return this->backend;
        }

        /**
         * @brief Set the backend
         * 
         * @param backend 
         */
        void setBackend(const std::string& backend) {
            this->backend = backend;
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
         * @brief Set the batch size
         * 
         * @param batchSize 
         */
        void setBatchSize(const size_t batchSize) {
            this->batchSize = batchSize;
        }

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
        Status parseModelVersionPolicy(std::string command) {
            rapidjson::Document node;
            if (command == "") {
                modelVersionPolicy = ModelVersionPolicy::getDefaultVersionPolicy();
                return StatusCode::OK;
            }

            if (node.Parse(command.c_str()).HasParseError()) {
                return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
            }

            if (!node.IsObject()) {
                return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
            }
            if (node.MemberCount() != 1) {
                return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
            }

            auto m = node.FindMember("all");
            if (m != node.MemberEnd()) {
                modelVersionPolicy = std::make_shared<AllModelVersionPolicy>();
                return StatusCode::OK;
            }

            m = node.FindMember("specific");
            if (m != node.MemberEnd()) {
                auto& specific = m->value;
                if (specific.MemberCount() != 1) {
                    return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
                }
                m = specific.FindMember("versions");
                if (m == specific.MemberEnd()) {
                    return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
                }
                std::vector<model_version_t> versions;
                for (auto& version : m->value.GetArray()) {
                    versions.push_back(version.GetInt64());
                }
                modelVersionPolicy = std::make_shared<SpecificModelVersionPolicy>(versions);
                return StatusCode::OK;
            }

            m = node.FindMember("latest");
            if (m != node.MemberEnd()) {
                auto& latest = m->value;
                if (latest.MemberCount() != 1) {
                    return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
                }
                m = latest.FindMember("num_versions");
                if (m == latest.MemberEnd()) {
                    return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
                }
                modelVersionPolicy = std::make_shared<LatestModelVersionPolicy>(m->value.GetInt64());
                return StatusCode::OK;
            }

            return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
        }

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
        Status parsePluginConfig(const rapidjson::Value& node) {
            if (!node.IsObject()) {
                return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
            }

            for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
                if (!it->value.IsString()) {
                    return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
                }
                pluginConfig[it->name.GetString()] = it->value.GetString();
            }

            return StatusCode::OK;
        }
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
         * @brief Get the shape
         * 
         * @return const shape_t& 
         */
        const shape_t& getShape() const {
            return this->shape;
        }

        /**
         * @brief Set the shape
         * 
         * @param shape 
         */
        void setShape(const shape_t& shape) {
            this->shape.clear();
            this->shapes.clear();
            this->shape.assign(shape.begin(), shape.end());
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
            this->shape.clear();
        }

        /**
         * @brief Legacy python mode: set the shape from the string representation
         * 
         * @param shapes 
         */
        Status setShape(const std::string& shape) {
            this->shape.clear();
            this->shapes.clear();
            return parseShape(this->shape, shape);
        }

        /**
         * @brief Sets the shape from the string representation
         *
         * @param shape
         * @param str
         * @return Status
         */
        static inline
        Status parseShape(shape_t& shape, const std::string& str) {
            std::string s = str;
            erase_spaces(s);

            // quick validation of valid characters
            if (s.find_first_not_of("0123456789(),") != std::string::npos)
                return StatusCode::SHAPE_WRONG_FORMAT;

            if (s.front() != shapeLeft || s.back() != shapeRight)
                return StatusCode::SHAPE_WRONG_FORMAT;

            s.pop_back();
            s.erase(s.begin());

            auto tokens = tokenize(s, shapeDelimeter);
            shape.clear();
            std::transform(tokens.begin(), tokens.end(), std::back_inserter(shape),
               [](const std::string& str) { return std::stoi(str); });

            return StatusCode::OK;
        }

        /**
         * @brief Add a single named shape
         * 
         * @param name 
         * @param shape 
         */
        void addShape(const std::string& name, const shape_t& shape) {
            this->shape.clear();
            this->shapes[name] = shape;
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
        Status parseModelMapping() {
            rapidjson::Document doc;
            SPDLOG_DEBUG("Parsing model:{} mapping from path:{}", getName(), getPath());
            mappingInputs.clear();
            mappingOutputs.clear();
            std::filesystem::path path = this->getPath();
            path.append(MAPPING_CONFIG_JSON);

            std::ifstream ifs(path.c_str());
            if (!ifs.good()) {
                return StatusCode::FILE_INVALID;
            }

            rapidjson::IStreamWrapper isw(ifs);
            if (doc.ParseStream(isw).HasParseError()) {
                return StatusCode::JSON_INVALID;
            }

            // Process inputs
            const auto itr = doc.FindMember("inputs");
            if (itr == doc.MemberEnd() || !itr->value.IsObject()) {
                spdlog::warn("Couldn't load inputs object from file {}", path.c_str());
            } else {
                for (const auto& key : itr->value.GetObject()) {
                    SPDLOG_DEBUG("Loaded input mapping {} => {}", key.name.GetString(), key.value.GetString());
                    mappingInputs[key.name.GetString()] = key.value.GetString();
                }
            }

            // Process outputs
            const auto it = doc.FindMember("outputs");
            if (it == doc.MemberEnd() || !it->value.IsObject()) {
                spdlog::warn("Couldn't load outputs object from file {}", path.c_str());
            } else {
                for (const auto& key : it->value.GetObject()) {
                    SPDLOG_DEBUG("Loaded output mapping {} => {}", key.name.GetString(), key.value.GetString());
                    mappingOutputs[key.name.GetString()] = key.value.GetString();
                }
            }

            return StatusCode::OK;
        }

        /**
         * @brief  Parses all settings from a JSON node
         * 
         * @return Status 
         */
        Status parseNode(const rapidjson::Value& v) {
            this->setName(v["name"].GetString());
            this->setBasePath(v["base_path"].GetString());

            // Check for optional parameters
            if (v.HasMember("batch_size")) {
                if (v["batch_size"].IsString()) {
                    // Although batch size is in, in legacy python version it was string
                    this->setBatchSize(std::atoi(v["batch_size"].GetString()));
                } else {
                    this->setBatchSize(v["batch_size"].GetUint64());
                }
            }
            if (v.HasMember("target_device"))
                this->setBackend(v["target_device"].GetString());
            if (v.HasMember("version"))
                this->setVersion(v["version"].GetInt64());
            if (v.HasMember("nireq"))
                this->setNireq(v["nireq"].GetUint64());

            if (v.HasMember("shape")) {
                // Legacy format as string
                if (v["shape"].IsString()) {
                    if (!this->setShape(v["shape"].GetString()).ok()) {
                        spdlog::error("There was an error parsing shape {}", v["shape"].GetString());
                    }
                } else {
                    if (v["shape"].IsArray()) {
                        // Shape for all inputs
                        shape_t shape;
                        for (auto& sh : v["shape"].GetArray()) {
                            shape.push_back(sh.GetUint64());
                        }
                        this->setShape(shape);
                    } else {
                        // Map of shapes
                        for (auto& s : v["shape"].GetObject()) {
                            shape_t shape;
                            // check if legacy format is used
                            if (s.value.IsString()) {
                                if (!ModelConfig::parseShape(shape, s.value.GetString()).ok()) {
                                    spdlog::error("There was an error parsing shape {}", v["shape"].GetString());
                                }
                            } else {
                                for (auto& sh : s.value.GetArray()) {
                                    shape.push_back(sh.GetUint64());
                                }
                            }
                            this->addShape(s.name.GetString(), shape);
                        }
                    }
                }
            }

            if (v.HasMember("layout")) {
                if (v["layout"].IsString()) {
                    this->setLayout(v["layout"].GetString());
                } else {
                    for (auto& s : v["layout"].GetObject()) {
                        this->addLayout(s.name.GetString(), s.value.GetString());
                    }
                }
            }

            if (v.HasMember("plugin_config")) {
                if (!parsePluginConfig(v["plugin_config"]).ok()) {
                    spdlog::error("Couldn't parse plugin config");
                }
            }

            if (v.HasMember("model_version_policy")) {
                rapidjson::StringBuffer buffer;
                buffer.Clear();
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                v["model_version_policy"].Accept(writer);
                if (!this->parseModelVersionPolicy(buffer.GetString()).ok()) {
                    spdlog::error("Couldn't parse model version policy");
                }
            } else {
                modelVersionPolicy = ModelVersionPolicy::getDefaultVersionPolicy();
            }

            return StatusCode::OK;
        }
    };
}  // namespace ovms
