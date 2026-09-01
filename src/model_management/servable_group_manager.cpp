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
#include "servable_group_manager.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "src/logging.hpp"
#include "src/model.hpp"
#include "src/modelconfig.hpp"
#include "src/modelinstance.hpp"
#include "modelmanager.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "src/mediapipe_internal/mediapipefactory.hpp"
#include "src/mediapipe_internal/mediapipegraphdefinition.hpp"
#endif

namespace ovms {

ServableGroupManager::ServableGroupManager(uint64_t idleTimeoutMicroseconds) :
    idleTimeoutMicroseconds(idleTimeoutMicroseconds),
    lastActivityTimeNs(std::make_shared<std::atomic<int64_t>>(
        std::chrono::steady_clock::now().time_since_epoch().count())) {
}

void ServableGroupManager::buildGroups(const std::unordered_map<std::string, ModelConfig>& modelConfigs,
    ModelManager& mm) {
    std::unique_lock lock(groupsMtx);
    groups.clear();
    servableToGroup.clear();

    for (const auto& [name, config] : modelConfigs) {
        const std::string& groupName = config.getGroupName();
        groups[groupName].groupName = groupName;
        groups[groupName].modelNames.insert(name);
        servableToGroup[name] = groupName;
    }

#if (MEDIAPIPE_DISABLE == 0)
    // Also process mediapipe graph definitions
    for (const auto& graphName : mm.getMediapipeFactory().getMediapipePipelinesNames()) {
        MediapipeGraphDefinition* def = mm.getMediapipeFactory().findDefinitionByName(graphName);
        if (def == nullptr) {
            continue;
        }
        const std::string& groupName = def->getMediapipeGraphConfig().getGroupName();
        if (groupName.empty()) {
            // No group_name set — treat graph name as its own group
            groups[graphName].groupName = graphName;
            groups[graphName].mediapipeNames.insert(graphName);
            servableToGroup[graphName] = graphName;
        } else {
            groups[groupName].groupName = groupName;
            groups[groupName].mediapipeNames.insert(graphName);
            servableToGroup[graphName] = groupName;
        }
    }
#endif

    size_t totalServables = modelConfigs.size();
#if (MEDIAPIPE_DISABLE == 0)
    totalServables += mm.getMediapipeFactory().getMediapipePipelinesNames().size();
#endif
    SPDLOG_INFO("Model group manager built {} groups from {} servables", groups.size(), totalServables);
    for (const auto& [gname, ginfo] : groups) {
        SPDLOG_INFO("  Group '{}': {} models, {} mediapipe graphs{}",
            gname, ginfo.modelNames.size(), ginfo.mediapipeNames.size(),
            ginfo.isPermanent() ? " (permanent)" : "");
    }
}

std::string ServableGroupManager::getGroupForServable(const std::string& servableName) const {
    std::shared_lock lock(groupsMtx);
    auto it = servableToGroup.find(servableName);
    if (it != servableToGroup.end()) {
        return it->second;
    }
    return "";
}

bool ServableGroupManager::isGroupLoaded(const std::string& groupName) const {
    if (groupName.empty()) {
        return false;
    }
    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
    if (it != groups.end() && it->second.isPermanent()) {
        return true;
    }
    return activeGroupName == groupName;
}

const std::string& ServableGroupManager::getActiveGroupName() const {
    return activeGroupName;
}

void ServableGroupManager::recordActivity() {
    lastActivityTimeNs->store(
        std::chrono::steady_clock::now().time_since_epoch().count(),
        std::memory_order_relaxed);
}

std::vector<std::string> ServableGroupManager::getAllConfiguredServableNames() const {
    std::shared_lock lock(groupsMtx);
    std::vector<std::string> names;
    for (const auto& [servableName, groupName] : servableToGroup) {
        names.push_back(servableName);
    }
    return names;
}

std::unordered_map<std::string, ModelGroupInfo> ServableGroupManager::getGroups() const {
    std::shared_lock lock(groupsMtx);
    return groups;
}

bool ServableGroupManager::canUnloadActiveGroup(ModelManager& mm) const {
    std::shared_lock lock(groupsMtx);
    auto it = groups.find(activeGroupName);
    if (it == groups.end()) {
        return true;
    }
    const auto& groupInfo = it->second;

    // Check all classic models in the group
    for (const auto& modelName : groupInfo.modelNames) {
        auto model = mm.findModelByName(modelName);
        if (model == nullptr) {
            continue;
        }
        for (const auto& [version, instance] : model->getModelVersions()) {
            if (!instance->canUnloadInstance()) {
                SPDLOG_DEBUG("Cannot unload group '{}': model {} version {} has active requests",
                    activeGroupName, modelName, version);
                return false;
            }
            if (instance->getStatus().getState() == ModelVersionState::LOADING) {
                SPDLOG_DEBUG("Cannot unload group '{}': model {} version {} is loading",
                    activeGroupName, modelName, version);
                return false;
            }
        }
    }

#if (MEDIAPIPE_DISABLE == 0)
    // Check all mediapipe graphs in the group
    for (const auto& graphName : groupInfo.mediapipeNames) {
        MediapipeGraphDefinition* def = mm.getMediapipeFactory().findDefinitionByName(graphName);
        if (def == nullptr) {
            continue;
        }
        auto activeCount = def->getActiveInferenceCount();
        if (activeCount && activeCount->load(std::memory_order_acquire) > 0) {
            SPDLOG_DEBUG("Cannot unload group '{}': mediapipe graph {} has active inferences",
                activeGroupName, graphName);
            return false;
        }
    }
#endif

    return true;
}

Status ServableGroupManager::loadGroup(const std::string& groupName, ModelManager& mm) {
    SPDLOG_INFO("Loading model group '{}'", groupName);

    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
    if (it == groups.end()) {
        SPDLOG_ERROR("Model group '{}' not found", groupName);
        return StatusCode::GROUP_LOAD_FAILED;
    }
    const auto& groupInfo = it->second;
    lock.unlock();

    Status firstError = StatusCode::OK;

    // Enqueue all servables in the group via the queue and collect futures
    std::vector<std::pair<std::string, std::future<Status>>> futures;
    for (const auto& modelName : groupInfo.modelNames) {
        futures.emplace_back(modelName, mm.requestServableLoad(modelName));
    }
#if (MEDIAPIPE_DISABLE == 0)
    for (const auto& graphName : groupInfo.mediapipeNames) {
        futures.emplace_back(graphName, mm.requestServableLoad(graphName));
    }
#endif

    for (auto& [name, future] : futures) {
        auto status = future.get();
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to load '{}' in group '{}': {}", name, groupName, status.string());
            if (firstError.ok()) {
                firstError = status;
            }
        } else {
            SPDLOG_INFO("Loaded '{}' in group '{}'", name, groupName);
        }
    }

    activeGroupName = groupName;
    recordActivity();

    if (!firstError.ok()) {
        return StatusCode::GROUP_LOAD_FAILED;
    }
    SPDLOG_INFO("Model group '{}' loaded successfully", groupName);
    return StatusCode::OK;
}

Status ServableGroupManager::unloadGroup(const std::string& groupName, ModelManager& mm) {
    SPDLOG_INFO("Unloading model group '{}'", groupName);

    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
    if (it == groups.end()) {
        return StatusCode::OK;
    }
    const auto& groupInfo = it->second;
    lock.unlock();

    // Enqueue retire/unload tasks via queue and collect futures
    std::vector<std::pair<std::string, std::future<Status>>> futures;
    for (const auto& modelName : groupInfo.modelNames) {
        futures.emplace_back(modelName, mm.requestServableRetire(modelName));
    }
#if (MEDIAPIPE_DISABLE == 0)
    for (const auto& graphName : groupInfo.mediapipeNames) {
        futures.emplace_back(graphName, mm.requestServableUnload(graphName));
    }
#endif

    for (auto& [name, future] : futures) {
        auto status = future.get();
        if (!status.ok()) {
            SPDLOG_WARN("Failed to unload '{}' in group '{}': {}", name, groupName, status.string());
        } else {
            SPDLOG_INFO("Unloaded '{}' in group '{}'", name, groupName);
        }
    }

    if (activeGroupName == groupName) {
        activeGroupName.clear();
    }
    SPDLOG_INFO("Model group '{}' unloaded successfully", groupName);
    return StatusCode::OK;
}

Status ServableGroupManager::ensureGroupLoaded(const std::string& servableName, ModelManager& mm) {
    std::string groupName = getGroupForServable(servableName);
    if (groupName.empty()) {
        // Not managed by group manager — let normal flow handle it
        return StatusCode::OK;
    }

    // Permanent group is always loaded
    {
        std::shared_lock lock(groupsMtx);
        auto it = groups.find(groupName);
        if (it != groups.end() && it->second.isPermanent()) {
            recordActivity();
            return StatusCode::OK;
        }
    }

    // Already the active group
    if (activeGroupName == groupName) {
        recordActivity();
        return StatusCode::OK;
    }

    // Serialize group swaps
    std::lock_guard<std::mutex> swapLock(loadUnloadMtx);

    // Double-check after acquiring the lock
    if (activeGroupName == groupName) {
        recordActivity();
        return StatusCode::OK;
    }

    // Unload the currently active group if any
    if (!activeGroupName.empty()) {
        SPDLOG_INFO("Swapping model group from '{}' to '{}'", activeGroupName, groupName);
        // Wait for active requests to drain with bounded retry
        constexpr int kMaxRetries = 300;  // 30 seconds at 100ms intervals
        constexpr int kRetryIntervalMs = 100;
        for (int i = 0; i < kMaxRetries; ++i) {
            if (canUnloadActiveGroup(mm)) {
                break;
            }
            if (i == kMaxRetries - 1) {
                SPDLOG_ERROR("Timed out waiting for group '{}' to drain requests before swap to '{}'",
                    activeGroupName, groupName);
                return StatusCode::GROUP_UNLOAD_BLOCKED;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
        }
        auto unloadStatus = unloadGroup(activeGroupName, mm);
        if (!unloadStatus.ok()) {
            SPDLOG_ERROR("Failed to unload group '{}': {}", activeGroupName, unloadStatus.string());
            return unloadStatus;
        }
    }

    return loadGroup(groupName, mm);
}

Status ServableGroupManager::ensureServableLoaded(const std::string& servableName, ModelManager& mm) {
    std::string group = getGroupForServable(servableName);
    if (!group.empty() && !isGroupLoaded(group)) {
        auto status = ensureGroupLoaded(servableName, mm);
        if (!status.ok())
            return status;
    }
    auto future = mm.requestServableLoad(servableName);
    auto status = future.get();
    if (!status.ok())
        return status;
    recordActivity();
    return StatusCode::OK;
}

void ServableGroupManager::unloadActiveGroupIfIdle(ModelManager& mm) {
    if (!isEnabled()) {
        return;
    }
    if (activeGroupName.empty()) {
        return;
    }

    // Check if we have a permanent group as active (should not happen, but safety)
    {
        std::shared_lock lock(groupsMtx);
        auto it = groups.find(activeGroupName);
        if (it != groups.end() && it->second.isPermanent()) {
            return;
        }
    }

    // Check idle timeout
    int64_t lastActivity = lastActivityTimeNs->load(std::memory_order_relaxed);
    int64_t nowNs = std::chrono::steady_clock::now().time_since_epoch().count();
    int64_t timeoutNs = static_cast<int64_t>(idleTimeoutMicroseconds) * 1'000LL;
    if ((nowNs - lastActivity) < timeoutNs) {
        return;
    }

    // Check if we can safely unload (no active requests)
    if (!canUnloadActiveGroup(mm)) {
        SPDLOG_DEBUG("Skipping idle unload of group '{}': active requests in flight", activeGroupName);
        return;
    }

    SPDLOG_INFO("Idle unloading model group '{}' after {}us timeout", activeGroupName, idleTimeoutMicroseconds);
    std::lock_guard<std::mutex> swapLock(loadUnloadMtx);
    // Re-check after acquiring lock
    if (activeGroupName.empty()) {
        return;
    }
    if (!canUnloadActiveGroup(mm)) {
        return;
    }
    unloadGroup(activeGroupName, mm);
}

}  // namespace ovms
