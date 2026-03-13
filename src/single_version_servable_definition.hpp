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

#include "modelversion.hpp"
#include "servable.hpp"
#include "servable_definition.hpp"

namespace ovms {

class SingleVersionServableDefinition : public ServableDefinition, public Servable {
public:
    static constexpr model_version_t VERSION = 1;

    SingleVersionServableDefinition(const std::string& name) :
        Servable(name, VERSION) {}

    const std::string& getName() const override { return Servable::getName(); }
    model_version_t getVersion() const override { return Servable::getVersion(); }
};

}  // namespace ovms
