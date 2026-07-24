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
#include <thread>
#include <variant>

#include <spdlog/spdlog.h>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

namespace ovms {

extern const std::string DEFAULT_GRAPH_FILENAME;
extern const std::string DEFAULT_SUBCONFIG_FILENAME;
extern const std::string DEFAULT_MODELMESH_SUBCONFIG_FILENAME;

struct GraphQueueAutoTag {
    bool operator==(const GraphQueueAutoTag&) const { return true; }
};

using GraphQueueSizeValue = std::optional<std::variant<int, GraphQueueAutoTag>>;

class Status;

class MediapipeGraphConfig {
private:
    std::string graphName;
    std::string basePath;
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

    /**
     * @brief Idle unload timeout in seconds.
     * 0 (default) = feature disabled.
     * When > 0, the graph's heavy resources are freed after this many seconds
     * of zero in-flight requests, and lazily reloaded on the next inference.
     */
    int idleUnloadTimeoutSeconds = 0;

public:
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

    const std::string& getGraphName() const {
        return this->graphName;
    }

    void setGraphName(const std::string& graphName) {
        this->graphName = graphName;
    }

    const std::string& getGraphPath() const {
        return this->graphPath;
    }

    const std::string& getBasePath() const {
        return this->basePath;
    }

    void setGraphPath(const std::string& graphPath);

    /**
         * @brief Set the Base Path using RootDirectoryPath (when base_path is not defined)
         *
         * @param basePath
         */
    void setBasePathWithRootPath();

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

    const std::string& getModelMeshSubconfigPath() const {
        return this->modelMeshSubconfigPath;
    }
    void setModelMeshSubconfigPath(const std::string& subconfigPath);

    void setRootDirectoryPath(const std::string& rootDirectoryPath) {
        this->rootDirectoryPath = rootDirectoryPath;
    }

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

    void setGraphQueueSize(int size) {
        this->graphQueueSize = size;
    }

    void setGraphQueueSizeAuto() {
        this->graphQueueSize = GraphQueueAutoTag{};
    }

    /**
     * @brief Resolve the graph queue size setting to a concrete integer.
     *
     * Returns:
     *   0   => queue creation disabled (user set 0 or not set)
     *   >0  => explicit size or resolved AUTO
     *
     * Negative values are rejected at parse time (resolveGraphQueueSize).
     * When not set (nullopt): returns 0 (queue disabled).
     * When AUTO: returns hardware_concurrency() or 16 as fallback.
     */
    int getInitialQueueSize() const {
        if (!this->graphQueueSize.has_value()) {
            return 0;  // not set - queue disabled by default
        }
        if (std::holds_alternative<GraphQueueAutoTag>(*this->graphQueueSize)) {
            unsigned int hwThreads = std::thread::hardware_concurrency();
            if (hwThreads == 0) {
                SPDLOG_WARN("std::thread::hardware_concurrency() returned 0 (unknown). Falling back to graph queue size 16.");
                return 16;
            }
            return static_cast<int>(hwThreads);
        }
        return std::get<int>(*this->graphQueueSize);
    }

    int getIdleUnloadTimeoutSeconds() const {
        return this->idleUnloadTimeoutSeconds;
    }

    void setIdleUnloadTimeoutSeconds(int seconds) {
        this->idleUnloadTimeoutSeconds = seconds;
    }

    bool isReloadRequired(const MediapipeGraphConfig& rhs) const;

    /**
    * @brief  Parses all settings from a JSON node
    *
    * @return Status
    */
    Status parseNode(const rapidjson::Value& v);

    void logGraphConfigContent() const;
};
}  // namespace ovms
