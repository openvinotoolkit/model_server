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

#include <optional>
#include <string>
#include <variant>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

namespace ovms {

extern const std::string DEFAULT_GRAPH_FILENAME;
extern const std::string DEFAULT_SUBCONFIG_FILENAME;
extern const std::string DEFAULT_MODELMESH_SUBCONFIG_FILENAME;

/**
 * @brief Tag type representing AUTO graph queue size (determined at runtime).
 */
struct GraphQueueAutoTag {
    bool operator==(const GraphQueueAutoTag&) const { return true; }
};

/**
 * @brief Represents the user's graph_queue_size setting.
 *
 * - std::nullopt              => user did not set this field
 * - int                       => user explicitly set a numeric value
 * - GraphQueueAutoTag         => user explicitly set "AUTO"
 */
using GraphQueueSizeValue = std::optional<std::variant<int, GraphQueueAutoTag>>;

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
     * @brief Optional model mesh subconfig path
     */
    std::string modelMeshSubconfigPath;

    /**
     * @brief MD5 hash for graph pbtxt file
     */
    std::string currentGraphPbTxtMD5;

    /**
     * @brief Graph queue size configuration.
     *
     * - std::nullopt              => user did not set this field
     * - int                       => user explicitly set a numeric size
     * - GraphQueueAutoTag         => user explicitly set "AUTO"
     */
    GraphQueueSizeValue graphQueueSize;

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
        return this->modelMeshSubconfigPath;
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

    /**
     * @brief Get the graph queue size setting.
     *
     * @return const GraphQueueSizeValue& - nullopt if not set, int or GraphQueueAutoTag
     */
    const GraphQueueSizeValue& getGraphQueueSize() const {
        return this->graphQueueSize;
    }

    /**
     * @brief Set the graph queue size to an explicit numeric value.
     */
    void setGraphQueueSize(int size) {
        this->graphQueueSize = size;
    }

    /**
     * @brief Set the graph queue size to AUTO.
     */
    void setGraphQueueSizeAuto() {
        this->graphQueueSize = GraphQueueAutoTag{};
    }

    /**
     * @brief Resolve the graph queue size setting to a concrete integer.
     *
     * Returns:
     *   -1  => queue creation disabled (user set -1)
     *    0  => queue with size 0 (user set 0)
     *   >0  => explicit size or resolved AUTO / default
     *
     * When not set (nullopt): returns -1 (queue disabled).
     * When AUTO: returns hardcoded value (TODO FIXME @atobisze determine optimal size).
     */
    int getInitialQueueSize() const {
        if (!this->graphQueueSize.has_value()) {
            return -1;  // not set - queue disabled by default
        }
        if (std::holds_alternative<GraphQueueAutoTag>(*this->graphQueueSize)) {
            return 16;  // TODO FIXME @atobisze determine optimal size based on nireq / hardware
        }
        return std::get<int>(*this->graphQueueSize);
    }

    bool isReloadRequired(const MediapipeGraphConfig& rhs) const;

    /**
    * @brief  Parses all settings from a JSON node
    *
    * @return Status
    */
    Status parseNode(const rapidjson::Value& v);

    /**
    * @brief  Logs the content of the graph configuration
    */
    void logGraphConfigContent() const;
};
}  // namespace ovms
