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
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

namespace ovms {

extern const std::string DEFAULT_GRAPH_FILENAME;
extern const std::string DEFAULT_SUBCONFIG_FILENAME;
extern const std::string DEFAULT_MODELMESH_SUBCONFIG_FILENAME;

class Status;

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

    /**
     * @brief MD5 hash for graph pbtxt file
     */
    std::string currentGraphPbTxtMD5;

public:
    /**
         * @brief Construct a new Mediapie Graph configuration object
         *
         * @param graphName
         * @param basePath
         * @param graphPath
         * @param subconfigPath
         * @param currentGraphPbTxtMD5
         */
    MediapipeGraphConfig(const std::string& graphName = "",
        const std::string& basePath = "",
        const std::string& graphPath = "",
        const std::string& subconfigPath = "",
        const std::string& currentGraphPbTxtMD5 = "") :
        graphName(graphName),
        basePath(basePath),
        graphPath(graphPath),
        currentGraphPbTxtMD5(currentGraphPbTxtMD5) {
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
         * @brief Set the Base Path using RootDirectoryPath (when base_path is not defined)
         *
         * @param basePath
         */
    void setBasePathWithRootPath();

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
         * @brief Get the ModelMesh ModelsConfig Path
         *
         * @return const std::string&
         */
    const std::string& getModelMeshSubconfigPath() const {
        return this->subconfigPath;
    }

    /**
           * @brief Set the Model Mesh Models Config Path
           *
           * @param subconfigPath
           */
    void setModelMeshSubconfigPath(const std::string& subconfigPath);

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

    void setCurrentGraphPbTxtMD5(const std::string& currentGraphPbTxtMD5) {
        this->currentGraphPbTxtMD5 = currentGraphPbTxtMD5;
    }

    bool isReloadRequired(const MediapipeGraphConfig& rhs) const;

    /**
    * @brief  Parses all settings from a JSON node
    *
    * @return Status
    */
    Status parseNode(const rapidjson::Value& v);
};
}  // namespace ovms
