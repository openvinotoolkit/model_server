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
#include <utility>

#include "logging.hpp"
#include "module.hpp"

namespace ovms {
class Config;
class Profiler;
#ifdef MTR_ENABLED
class ProfilerModule : public Module {
    std::unique_ptr<Profiler> profiler;

public:
    ProfilerModule();
    Status start(const Config& config) override;
    void shutdown() override;
    ~ProfilerModule();
};
#endif
}  // namespace ovms
