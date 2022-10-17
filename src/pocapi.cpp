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
#include <future>
#include <iostream>
#include <memory>
#include <utility>

#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"

using ovms::Server;

int OVMS_Start(int argc, char** argv) {
    Server& server = Server::instance();
    return server.start(argc, argv);
}

void OVMS_Infer(char* name, float* data, float* output) {
    Server& server = Server::instance();
    std::shared_ptr<ovms::ModelInstance> instance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto module = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return;
    }
    auto servableManagerModule = dynamic_cast<const ovms::ServableManagerModule*>(module);
    auto& manager = servableManagerModule->getServableManager();
    manager.getModelInstance(name, 0, instance, modelInstanceUnloadGuardPtr);
    instance->infer(data, output);
}
