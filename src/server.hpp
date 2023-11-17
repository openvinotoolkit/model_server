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
#include <csignal>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "module.hpp"

namespace ovms {
class Config;
class Status;

class ServerSettingsImpl;
class ModelsSettingsImpl;

extern const std::string PROFILER_MODULE_NAME;
extern const std::string GRPC_SERVER_MODULE_NAME;
extern const std::string HTTP_SERVER_MODULE_NAME;
extern const std::string SERVABLE_MANAGER_MODULE_NAME;
extern const std::string METRICS_MODULE_NAME;
extern const std::string PYTHON_INTERPRETER_MODULE_NAME;

class Server {
    mutable std::shared_mutex modulesMtx;
    mutable std::mutex startMtx;

protected:
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    Server() = default;
    virtual std::unique_ptr<Module> createModule(const std::string& name);

public:
    static Server& instance();
    int start(int argc, char** argv);
    Status start(ServerSettingsImpl*, ModelsSettingsImpl*, bool withPython = true);
    ModuleState getModuleState(const std::string& name) const;
    const Module* getModule(const std::string& name) const;
    bool isReady() const;
    bool isLive() const;

    void setShutdownRequest(int i);
    virtual ~Server();
    // TODO: Include withPython in ovms::Config
    Status startModules(ovms::Config& config, bool withPython = true);
    void shutdownModules();

private:
    void ensureModuleShutdown(const std::string& name);
};
}  // namespace ovms
