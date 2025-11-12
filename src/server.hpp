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
#include <utility>

#include "capi_frontend/server_settings.hpp"
#include "module.hpp"
#include "module_names.hpp"
namespace {
volatile sig_atomic_t shutdown_request = 0;
volatile sig_atomic_t ovms_exited = 0;
}  // namespace
namespace ovms {
class Config;
class Status;

class Server {
    mutable std::shared_mutex modulesMtx;
    mutable std::mutex startMtx;
    mutable std::mutex exitMtx;
    mutable std::mutex shutdownMtx;

protected:
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    Server() = default;
    virtual std::unique_ptr<Module> createModule(const std::string& name);

public:
    static Server& instance();
    int start(int argc, char** argv);
    static std::variant<std::pair<ServerSettingsImpl, ModelsSettingsImpl>, std::pair<int, std::string>> parseArgs(int argc, char** argv);
    int startServerFromSettings(ServerSettingsImpl& serverSettings, ModelsSettingsImpl& modelsSettings);
    Status startFromSettings(ServerSettingsImpl*, ModelsSettingsImpl*);
    ModuleState getModuleState(const std::string& name) const;
    const Module* getModule(const std::string& name) const;
    bool isReady() const;
    bool isLive(const std::string& moduleName) const;

    int getShutdownStatus();
    void setShutdownRequest(int i);
    int getExitStatus();
    void setExitStatus(int i);
    virtual ~Server();
    Status startModules(ovms::Config& config);
    void shutdownModules();

private:
    void ensureModuleShutdown(const std::string& name);
};
}  // namespace ovms
