//****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <string>

#include "../module.hpp"
#include "../capi_frontend/server_settings.hpp"

namespace ovms {

class HfPullModelModule : public Module {
protected:
    HFSettingsImpl hfSettings;
    const std::string GetProxy() const;
    const std::string GetHfToken() const;
    const std::string GetHfEndpoint() const;

public:
    HfPullModelModule();
    ~HfPullModelModule();
    Status start(const ovms::Config& config) override;
    void shutdown() override;

    Status clone() const;
};
}  // namespace ovms
