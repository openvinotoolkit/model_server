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

#include "module.hpp"

namespace ovms {

class DeviceDisplayOwnerModule : public Module {
 public:
    Status start(const Config& config) override;
    void shutdown() override;
   ~DeviceDisplayOwnerModule() override;

 private:
    int renderNodeFd = -1;
    void* vaDisplay = nullptr;
};

}  // namespace ovms