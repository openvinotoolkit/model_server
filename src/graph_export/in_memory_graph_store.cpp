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

#include "in_memory_graph_store.hpp"

#include <string>

namespace ovms {
namespace {
std::string inMemoryGraphContent;
}  // namespace

bool InMemoryGraphStore::hasContent() {
    return !inMemoryGraphContent.empty();
}

const std::string& InMemoryGraphStore::getContent() {
    return inMemoryGraphContent;
}

void InMemoryGraphStore::setContent(const std::string& content) {
    inMemoryGraphContent = content;
}

void InMemoryGraphStore::clearContent() {
    inMemoryGraphContent.clear();
}

}  // namespace ovms
