//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <any>
#include <csignal>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace ovms {
class Config;
enum class ModuleState {
    NOT_INITIALIZED,
    STARTED_INITIALIZE,
    INITIALIZED,
    RELOADING,
    STARTED_SHUTDOWN,
    SHUTDOWN
};

class Module {
protected:
    ModuleState state = ModuleState::NOT_INITIALIZED;

public:
    virtual int start(const ovms::Config& config) = 0;
    virtual void shutdown() = 0;
    virtual ~Module() = default;
    ModuleState getState() const;
};

extern const std::string PROFILER_MODULE_NAME;
extern const std::string GRPC_SERVER_MODULE_NAME;
extern const std::string HTTP_SERVER_MODULE_NAME;
extern const std::string SERVABLE_MANAGER_MODULE_NAME;

class Server {
    mutable std::shared_mutex modulesMtx;

protected:
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    Server() = default;
    virtual std::unique_ptr<Module> createModule(const std::string& name);

public:
    static Server& instance();
    int start(int argc, char** argv);
    ModuleState getModuleState(const std::string& name) const;
    const Module* getModule(const std::string& name) const;
    bool isReady() const;
    bool isLive() const;

    void setShutdownRequest(int i);
    virtual ~Server();

    // TODO potentially to be hiden under protected and exposed only in tests by inheritance
    // #KFS_CLEANUP
    int startModules(ovms::Config& config);
    void shutdownModules(ovms::Config& config);
};
}  // namespace ovms
