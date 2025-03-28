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
#include "mediapipegraphconfig.hpp"

#include <string>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)
#include <spdlog/spdlog.h>

#include "../filesystem.hpp"
#include "../status.hpp"

namespace ovms {

const std::string DEFAULT_GRAPH_FILENAME = "graph.pbtxt";
const std::string DEFAULT_SUBCONFIG_FILENAME = "subconfig.json";

void MediapipeGraphConfig::setBasePathWithRootPath() {
    this->basePath = this->rootDirectoryPath;
}

void MediapipeGraphConfig::setBasePath(const std::string& basePath) {
    FileSystem::setPath(this->basePath, basePath, this->rootDirectoryPath);
}

void MediapipeGraphConfig::setGraphPath(const std::string& graphPath) {
    FileSystem::setPath(this->graphPath, graphPath, this->basePath);
}

void MediapipeGraphConfig::setSubconfigPath(const std::string& subconfigPath) {
    FileSystem::setPath(this->subconfigPath, subconfigPath, this->basePath);
}

bool MediapipeGraphConfig::isReloadRequired(const MediapipeGraphConfig& rhs) const {
    // Checking OVMS configuration part
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
    // Checking if graph pbtxt has been modified
    if (currentGraphPbTxtMD5 != "") {
        std::string newGraphPbTxtMD5 = FileSystem::getFileMD5(rhs.graphPath);
        if (newGraphPbTxtMD5 != currentGraphPbTxtMD5) {
            SPDLOG_DEBUG("MediapipeGraphConfig {} reload required due to graph definition modification", this->graphName);
            return true;
        }
    }
    return false;
}

Status MediapipeGraphConfig::parseNode(const rapidjson::Value& v) {
    try {
        this->setGraphName(v["name"].GetString());
        if (v.HasMember("base_path")) {
            std::string providedBasePath(v["base_path"].GetString());
            if (providedBasePath.size() == 0)
                this->setBasePath(this->getGraphName() + FileSystem::getOsSeparator());
            else if (providedBasePath.back() == FileSystem::getOsSeparator().back())
                this->setBasePath(providedBasePath);
            else
                this->setBasePath(providedBasePath + FileSystem::getOsSeparator());
        } else { // THIS IS SPECIAL CASE FOR MODEL MESH
            if (!getRootDirectoryPath().empty()) {
                // check if pbtxt exist in eiter dir or dir/1
                this->setBasePath(this->getGraphName() + FileSystem::getOsSeparator());
                SPDLOG_DEBUG("base_path not defined in config so it will be set to default based on main config directory: {}", this->getBasePath());
            } else {
                SPDLOG_ERROR("Mediapipe {} root directory path is not set.", getGraphName());
                return StatusCode::INTERNAL_ERROR;
            }
        }
        if (v.HasMember("graph_path")) {
            this->setGraphPath(v["graph_path"].GetString());
        } else {
            this->setGraphPath(DEFAULT_GRAPH_FILENAME);
            SPDLOG_DEBUG("graph_path not defined in config so it will be set to default based on base_path and graph name: {}", this->getGraphPath());
        }
        this->setCurrentGraphPbTxtMD5(FileSystem::getFileMD5(this->graphPath));

        if (v.HasMember("subconfig")) {
            this->setSubconfigPath(v["subconfig"].GetString());
        } else {
            std::string defaultSubconfigPath = getBasePath() + "subconfig.json";
            SPDLOG_DEBUG("No subconfig path was provided for graph: {} so default subconfig file: {} will be loaded.", getGraphName(), defaultSubconfigPath);
            this->setSubconfigPath(DEFAULT_SUBCONFIG_FILENAME);
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
}  // namespace ovms
