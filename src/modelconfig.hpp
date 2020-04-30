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
#include <string>
#include <unordered_map>

#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

using shape_t = std::vector<size_t>;
using shapes_map_t = std::unordered_map<std::string, shape_t>;
using layouts_map_t = std::unordered_map<std::string, std::string>;
using model_version_t = int64_t;
using plugin_config_t = std::map<std::string, std::string>;

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
        std::string modelVersionPolicy;

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
            modelVersionPolicy(""),
            nireq(1),
            pluginConfig({}),
            shape({}),
            layout(""),
            shapes({}),
            layouts({}),
            version(0)
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
                    uint64_t nireq
                    ) : 
                    name(name),
                    basePath(basePath),
                    backend(backend),
                    batchSize(batchSize),
                    modelVersionPolicy(""),
                    nireq(nireq),
                    pluginConfig({}),
                    shape({}),
                    layout(""),
                    shapes({}),
                    layouts({}),
                    version(0)
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
         * @return const std::string&
         */
        const std::string& getModelVersionPolicy() const {
        	return this->modelVersionPolicy;
        }

        /**
         * @brief Set the model version policy
         * 
         * @param modelVersionPolicy 
         */
        void setModelVersionPolicy(const std::string& modelVersionPolicy) {
        	this->modelVersionPolicy = modelVersionPolicy;
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
                return Status::SHAPE_WRONG_FORMAT;

            if (s.front() != shapeLeft || s.back() != shapeRight)
                return Status::SHAPE_WRONG_FORMAT; 
            
            s.pop_back();
            s.erase(s.begin());

            auto tokens = tokenize(s, shapeDelimeter);
            shape.clear();
            std::transform(tokens.begin(), tokens.end(), std::back_inserter(shape),
               [](const std::string& str) { return std::stoi(str); });

            return Status::OK;
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
    };
}
