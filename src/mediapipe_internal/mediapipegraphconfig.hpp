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

#include <string>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "../status.hpp"

namespace ovms {

/**
     * @brief This class represents Mediapie Graph configuration
     */
class MediapipeGraphConfig {
private:
    /**
         * @brief Mediapipe Graph Name
         */
    std::string graphName;

    /**
         * @brief Mediapipe Base Path
         */
    std::string basePath;

    /**
         * @brief Mediapipe Graph Path
         */
    std::string graphPath;

    /**
         * @brief Json config directory path
         */
    std::string rootDirectoryPath;

    /**
     * @brief Json config path
     */
    std::string subconfigPath;

public:
    /**
         * @brief Construct a new Mediapie Graph configuration object
         *
         * @param graphName
         * @param basePath
         * @param graphPath
         * @param subconfigPath
         */
    MediapipeGraphConfig(const std::string& graphName = "",
        const std::string& basePath = "",
        const std::string& graphPath = "",
        const std::string& subconfigPath = "") :
        graphName(graphName),
        basePath(basePath),
        graphPath(graphPath),
        subconfigPath(subconfigPath) {
    }

    void clear() {
        graphName.clear();
        graphPath.clear();
    }

    /**
         * @brief Get the Graph name
         *
         * @return const std::string&
         */
    const std::string& getGraphName() const {
        return this->graphName;
    }

    /**
         * @brief Set the Graph name
         *
         * @param name
         */
    void setGraphName(const std::string& graphName) {
        this->graphName = graphName;
    }

    /**
         * @brief Get the Graph Path
         *
         * @return const std::string&
         */
    const std::string& getGraphPath() const {
        return this->graphPath;
    }

    /**
         * @brief Get the Base Path
         *
         * @return const std::string&
         */
    const std::string& getBasePath() const {
        return this->basePath;
    }

    /**
         * @brief Set the Graph Path
         *
         * @param graphPath
         */
    void setGraphPath(const std::string& graphPath);

    /**
         * @brief Set the Base Path
         *
         * @param basePath
         */
    void setBasePath(const std::string& basePath);

    /**
         * @brief Get the ModelsConfig Path
         *
         * @return const std::string&
         */
    const std::string& getSubconfigPath() const {
        return this->subconfigPath;
    }

    /**
         * @brief Set the Models Config Path
         *
         * @param subconfigPath
         */
    void setSubconfigPath(const std::string& subconfigPath);

    /**
         * @brief Set root directory path
         *
         * @param rootDirectoryPath
         */
    void setRootDirectoryPath(const std::string& rootDirectoryPath) {
        this->rootDirectoryPath = rootDirectoryPath;
    }

    /**
     * @brief Get the root directory path
     *
     * @return const std::string&
     */
    const std::string& getRootDirectoryPath() const {
        return this->rootDirectoryPath;
    }

    bool isReloadRequired(const MediapipeGraphConfig& rhs) const {
        if (this->graphName != rhs.graphName) {
            SPDLOG_DEBUG("MediapipeGraphConfig {} reload required due to name mismatch", this->graphName);
            return true;
        }
        if (this->basePath != rhs.basePath) {
            SPDLOG_DEBUG("MediapipeGraphConfig {} reload required due to basePath mismatch", this->graphName);
            return true;
        }
        if (this->graphPath != rhs.graphPath) {
            SPDLOG_DEBUG("MediapipeGraphConfig {} reload required due to graphPath mismatch", this->graphName);
            return true;
        }
        if (this->subconfigPath != rhs.subconfigPath) {
            SPDLOG_DEBUG("MediapipeGraphConfig {} reload required due to subconfigPath mismatch", this->graphName);
            return true;
        }
        return false;
    }

    /**
     * @brief  Parses all settings from a JSON node
        *
        * @return Status
        */
    Status parseNode(const rapidjson::Value& v) {
        try {
            this->setGraphName(v["name"].GetString());
            if (v.HasMember("base_path")) {
                std::string providedBasePath(v["base_path"].GetString());
                if (providedBasePath.back() == '/')
                    this->setBasePath(providedBasePath);
                else
                    this->setBasePath(providedBasePath + "/");
            } else {
                if (!getRootDirectoryPath().empty()) {
                    this->setBasePath(getRootDirectoryPath());
                    SPDLOG_DEBUG("base_path not defined in config so it will be set to default based on main config directory: {}", this->getBasePath());
                } else {
                    SPDLOG_ERROR("Mediapipe {} root directory path is not set.", getGraphName());
                    return StatusCode::INTERNAL_ERROR;
                }
            }
            if (v.HasMember("graph_path")) {
                this->setGraphPath(v["graph_path"].GetString());
            } else {
                this->setGraphPath(basePath + "graph.pbtxt");
                SPDLOG_DEBUG("graph_path not defined in config so it will be set to default based on base_path and graph name: {}", this->getGraphPath());
            }
            if (v.HasMember("subconfig")) {
                this->setSubconfigPath(v["subconfig"].GetString());
            } else {
                std::string defaultSubconfigPath = getBasePath() + "subconfig.json";
                SPDLOG_DEBUG("No subconfig path was provided for graph: {} so default subconfig file: {} will be loaded.", getGraphName(), defaultSubconfigPath);
                this->setSubconfigPath(defaultSubconfigPath);
            }
        } catch (std::logic_error& e) {
            SPDLOG_DEBUG("Relative path error: {}", e.what());
            return StatusCode::INTERNAL_ERROR;
        } catch (...) {
            SPDLOG_ERROR("There was an error parsing the mediapipe graph config");
            return StatusCode::JSON_INVALID;
        }
        return StatusCode::OK;
    }
};
}  // namespace ovms
