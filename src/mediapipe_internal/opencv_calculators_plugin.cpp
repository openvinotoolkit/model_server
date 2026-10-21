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
#include <memory>
#include <string>
#include <utility>

#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/subgraph.h"

namespace {

template <typename Registry>
void registerFactories(decltype(Registry::functions()) hostRegistry) {
    if (hostRegistry == Registry::functions()) {
        return;
    }
    // Upstream static registrations form a plugin-local discovery catalog.
    // Install their factories in the actual host registry explicitly: OVMS
    // hides its template singletons, so ELF interposition cannot share them.
    for (const auto& name : Registry::GetRegisteredNames()) {
        if (!hostRegistry->IsRegistered(name)) {
            hostRegistry->Register(name, [name]() {
                return std::move(Registry::CreateByName(name)).value();
            });
        }
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) bool OVMS_RegisterOpenCvCalculators(
    void* calculatorRegistry, void* subgraphRegistry) {
    if (calculatorRegistry == nullptr || subgraphRegistry == nullptr) {
        return false;
    }
    registerFactories<mediapipe::CalculatorBaseRegistry>(
        static_cast<decltype(mediapipe::CalculatorBaseRegistry::functions())>(calculatorRegistry));
    registerFactories<mediapipe::SubgraphRegistry>(
        static_cast<decltype(mediapipe::SubgraphRegistry::functions())>(subgraphRegistry));
    return true;
}
