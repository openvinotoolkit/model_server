//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "session_id.hpp"

namespace ovms {

using session_id_t = uint32_t;
using session_key_t = std::string;

class NodeSessionMetadata {
    std::unordered_map<std::string, std::tuple<session_id_t, session_id_t>> details;
    std::vector<std::string> sessionsLevels;

public:
    std::vector<NodeSessionMetadata> generateSubsessions(const std::string& nodeName, session_id_t subsessionSize) const;
    std::string getSessionKey(const std::set<std::string>& ignoredNodeNames = {}) const;
    NodeSessionMetadata getCollapsedSessionMetadata(const std::set<std::string>& ignoredNodeNames) const;
    session_id_t getSubsessionSize(const std::string& subsessionName) const;
    session_id_t getShardId(const std::set<std::string>& collapsedNames = {}) const;
};
}  // namespace ovms
