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
#pragma once

#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ovms {

class MetricProvider;
class ServableNameChecker;
class Status;
class MediapipeGraphConfig;
class MediapipeGraphDefinition;
class MediapipeGraphExecutor;
class PythonBackend;

class MediapipeFactory {
    std::map<std::string, std::shared_ptr<MediapipeGraphDefinition>> definitions;
    std::map<std::string, std::string> loraAliases;  // alias -> real graph definition name
    mutable std::shared_mutex definitionsMtx;
    PythonBackend* pythonBackend{nullptr};

public:
    MediapipeFactory() = delete;
    MediapipeFactory(PythonBackend* pythonBackend = nullptr);
    Status createDefinition(const std::string& pipelineName,
        const MediapipeGraphConfig& config,
        MetricProvider& metrics,
        const ServableNameChecker& checker);

    bool definitionExists(const std::string& name) const;

public:
    Status create(std::unique_ptr<MediapipeGraphExecutor>& pipeline,
        const std::string& name) const;

    MediapipeGraphDefinition* findDefinitionByName(const std::string& name) const;
    void registerLoraAlias(const std::string& alias, const std::string& graphName);
    void clearLoraAliases(const std::string& graphName);
    Status reloadDefinition(const std::string& pipelineName,
        const MediapipeGraphConfig& config,
        const ServableNameChecker& checker);

    void retireOtherThan(std::set<std::string>&& pipelinesInConfigFile);
    Status revalidatePipelines();
    const std::vector<std::string> getMediapipePipelinesNames() const;
    const std::vector<std::string> getNamesOfAvailableMediapipePipelines() const;
    ~MediapipeFactory();
};

}  // namespace ovms
