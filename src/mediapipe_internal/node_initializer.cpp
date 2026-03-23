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
#include "node_initializer.hpp"

namespace ovms {
NodeInitializerRegistry& NodeInitializerRegistry::instance() {
    static NodeInitializerRegistry registry;
    return registry;
}

void NodeInitializerRegistry::add(std::unique_ptr<NodeInitializer> initializer) {
    initializers_.push_back(std::move(initializer));
}

const std::vector<std::unique_ptr<NodeInitializer>>& NodeInitializerRegistry::all() const {
    return initializers_;
}
}  // namespace ovms
