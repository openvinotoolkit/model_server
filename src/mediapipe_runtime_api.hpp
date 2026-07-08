//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "execution_context.hpp"
#include "status.hpp"

namespace grpc_impl {
template <typename W, typename R>
class ServerReaderWriterInterface;
}

namespace inference {
class ModelInferRequest;
class ModelInferResponse;
class ModelStreamInferResponse;
}  // namespace inference

namespace ovms {

class MetricProvider;
class ServableNameChecker;
class MediapipeGraphConfig;
class MediapipeGraphExecutorInterface;
class MediapipeGraphExecutor;
class ServableDefinition;
class PythonBackend;
struct HFSettingsImpl;
struct HttpPayload;
class HttpAsyncWriter;

class MediapipeRuntimeApi {
public:
    explicit MediapipeRuntimeApi(PythonBackend* pythonBackend);
    ~MediapipeRuntimeApi();

    MediapipeRuntimeApi(const MediapipeRuntimeApi&) = delete;
    MediapipeRuntimeApi& operator=(const MediapipeRuntimeApi&) = delete;

    bool isLoaded() const;

    Status processConfig(const MediapipeGraphConfig& config,
        MetricProvider& metrics,
        const ServableNameChecker& checker);

    Status create(std::unique_ptr<MediapipeGraphExecutor>& pipeline,
        const std::string& name) const;
    Status createHandle(std::unique_ptr<MediapipeGraphExecutorInterface>& pipeline,
        const std::string& name) const;

    bool definitionExists(const std::string& name) const;
    bool aliasesConflictExcluding(const std::vector<std::string>& aliases, const std::string& ownGraphName) const;
    void retireOtherThan(const std::set<std::string>& graphsInConfigFile);
    const std::vector<std::string> getMediapipePipelinesNames() const;
    const std::vector<std::string> getNamesOfAvailableMediapipePipelines() const;
    ServableDefinition* findServableDefinitionByName(const std::string& name) const;
    Status createServableConfig(const std::string& directoryPath,
        const HFSettingsImpl& hfSettings,
        bool writeToFile) const;

private:
    struct ApiSymbols;
    std::unique_ptr<ApiSymbols> api;

    static std::string joinWithNewlines(const std::vector<std::string>& values);
    static std::string joinWithNewlines(const std::set<std::string>& values);
    static std::vector<std::string> splitNewlineDelimited(const std::string& data);
};

}  // namespace ovms
