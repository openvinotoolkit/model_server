//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/status.hpp"

namespace ovms {

class ModelConfig;
class ModelManager;

struct ModelGroupInfo {
    std::string groupName;
    std::set<std::string> modelNames;
    std::set<std::string> mediapipeNames;
    bool isPermanent() const { return groupName == "permanent"; }
};

class ServableGroupManager {
public:
    explicit ServableGroupManager(uint64_t idleTimeoutMicroseconds);

    bool isEnabled() const { return idleTimeoutMicroseconds > 0; }
    uint64_t getIdleTimeoutMicroseconds() const { return idleTimeoutMicroseconds; }

    void buildGroups(const std::unordered_map<std::string, ModelConfig>& modelConfigs,
        ModelManager& mm);

    std::string getGroupForServable(const std::string& servableName) const;

    bool isGroupLoaded(const std::string& groupName) const;

    Status ensureGroupLoaded(const std::string& servableName, ModelManager& mm);

    Status ensureServableLoaded(const std::string& servableName, ModelManager& mm);

    void unloadActiveGroupIfIdle(ModelManager& mm);

    void recordActivity();

    std::vector<std::string> getAllConfiguredServableNames() const;
    // needed only for tests
    std::unordered_map<std::string, ModelGroupInfo> getGroups() const;
    const std::string& getActiveGroupName() const;

private:
    bool canUnloadActiveGroup(ModelManager& mm) const;
    Status loadGroup(const std::string& groupName, ModelManager& mm);
    Status unloadGroup(const std::string& groupName, ModelManager& mm);

    uint64_t idleTimeoutMicroseconds;

    mutable std::shared_mutex groupsMtx;
    std::unordered_map<std::string, ModelGroupInfo> groups;
    std::unordered_map<std::string, std::string> servableToGroup;

    mutable std::mutex loadUnloadMtx;
    std::string activeGroupName;

    std::shared_ptr<std::atomic<int64_t>> lastActivityTimeNs;
};

}  // namespace ovms
