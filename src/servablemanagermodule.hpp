//****************************************************************************
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
#include <memory>

#include "module.hpp"

namespace ovms {
class Config;
class ModelManager;
class Server;

class ServableManagerModule : public Module {
protected:
    mutable std::unique_ptr<ModelManager> servableManager;

public:
    ServableManagerModule(ovms::Server& ovmsServer);
    ~ServableManagerModule();
    Status start(const ovms::Config& config) override;

    void shutdown() override;
    virtual ModelManager& getServableManager() const;
};
}  // namespace ovms
