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
        // Retired definitions stay in the factory after config removal; registering them
        // here would let a later wake-up resurrect a graph the user deleted.
        if (def->getStateCode() == PipelineDefinitionStateCode::RETIRED) {
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
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Model group manager built {} groups from {} servables", groups.size(), totalServables);
    for (const auto& [gname, ginfo] : groups) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "  Group '{}': {} models, {} mediapipe graphs{}",
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
    {
        std::shared_lock lock(groupsMtx);
        auto it = groups.find(groupName);
        if (it != groups.end() && it->second.isPermanent()) {
            return true;
        }
    }
    return isActiveGroup(groupName);
}

bool ServableGroupManager::isActiveGroup(const std::string& groupName) const {
    std::shared_lock lock(activeGroupNameMtx);
    return activeGroupName == groupName;
}

void ServableGroupManager::setActiveGroup(const std::string& groupName) {
    std::unique_lock lock(activeGroupNameMtx);
    activeGroupName = groupName;
}

std::string ServableGroupManager::getActiveGroupName() const {
    std::shared_lock lock(activeGroupNameMtx);
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
    // TODO @atobiszei this is vulnerable to TOCTOU
    const std::string groupName = getActiveGroupName();
    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
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
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cannot unload group '{}': model {} version {} has active requests",
                    groupName, modelName, version);
                return false;
            }
            if (instance->getStatus().getState() == ModelVersionState::LOADING) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cannot unload group '{}': model {} version {} is loading",
                    groupName, modelName, version);
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
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cannot unload group '{}': mediapipe graph {} has active inferences",
                groupName, graphName);
            return false;
        }
    }
#endif

    return true;
}

Status ServableGroupManager::loadGroup(const std::string& groupName, ModelManager& mm,
    const std::string& requestedServable) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Loading model group '{}'", groupName);

    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
    if (it == groups.end()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model group '{}' not found", groupName);
        return StatusCode::GROUP_LOAD_FAILED;
    }
    const auto& groupInfo = it->second;
    lock.unlock();

    // The servable that triggered the wake-up is loaded first and preempts queued
    // work, so time-to-first-response does not depend on group member ordering.
    std::vector<std::pair<std::string, std::future<Status>>> futures;
    const bool isRequestedInGroup = groupInfo.modelNames.count(requestedServable) > 0 ||
                                    groupInfo.mediapipeNames.count(requestedServable) > 0;
    if (isRequestedInGroup) {
        futures.emplace_back(requestedServable, mm.requestServableWakeUp(requestedServable, /*urgent=*/true));
    }
    for (const auto& modelName : groupInfo.modelNames) {
        if (modelName == requestedServable) {
            continue;
        }
        futures.emplace_back(modelName, mm.requestServableWakeUp(modelName, /*urgent=*/false));
    }
#if (MEDIAPIPE_DISABLE == 0)
    for (const auto& graphName : groupInfo.mediapipeNames) {
        if (graphName == requestedServable) {
            continue;
        }
        futures.emplace_back(graphName, mm.requestServableWakeUp(graphName, /*urgent=*/false));
    }
#endif

    // Caller waits on the servable it asked for; the rest of the group only produces logs.
    Status requestedStatus = isRequestedInGroup ? Status(StatusCode::GROUP_LOAD_FAILED) : Status(StatusCode::OK);
    for (auto& [name, future] : futures) {
        auto status = future.get();
        if (name == requestedServable) {
            requestedStatus = status;
        }
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to load '{}' in group '{}': {}", name, groupName, status.string());
        } else {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Loaded '{}' in group '{}'", name, groupName);
        }
    }

    // Set even on partial failure - loaded members must stay tracked so they can be unloaded later.
    setActiveGroup(groupName);
    recordActivity();

    if (!requestedStatus.ok()) {
        return requestedStatus;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Model group '{}' loaded successfully", groupName);
    return StatusCode::OK;
}

Status ServableGroupManager::unloadGroup(const std::string& groupName, ModelManager& mm, bool urgent) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Unloading model group '{}'", groupName);

    std::shared_lock lock(groupsMtx);
    auto it = groups.find(groupName);
    if (it == groups.end()) {
        return StatusCode::OK;
    }
    const auto& groupInfo = it->second;
    lock.unlock();

    // Enqueue put-to-sleep tasks via queue and collect futures
    std::vector<std::pair<std::string, std::future<Status>>> futures;
    for (const auto& modelName : groupInfo.modelNames) {
        futures.emplace_back(modelName, mm.requestServablePutToSleep(modelName, urgent));
    }
#if (MEDIAPIPE_DISABLE == 0)
    for (const auto& graphName : groupInfo.mediapipeNames) {
        futures.emplace_back(graphName, mm.requestServablePutToSleep(graphName, urgent));
    }
#endif

    for (auto& [name, future] : futures) {
        auto status = future.get();
        if (!status.ok()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Failed to unload '{}' in group '{}': {}", name, groupName, status.string());
        } else {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Unloaded '{}' in group '{}'", name, groupName);
        }
    }

    if (isActiveGroup(groupName)) {
        setActiveGroup("");
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Model group '{}' unloaded successfully", groupName);
    return StatusCode::OK;
}

[[nodiscard]] Status ServableGroupManager::swapToGroup(const std::string& groupName, ModelManager& mm,
    const std::string& requestedServable) {
    const std::string previousGroup = getActiveGroupName();
    if (!previousGroup.empty()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Swapping model group from '{}' to '{}'", previousGroup, groupName);
        // Wait for active requests to drain with bounded retry
        constexpr int kMaxRetries = 300;  // 30 seconds at 100ms intervals
        constexpr int kRetryIntervalMs = 100;
        for (int i = 0; i < kMaxRetries; ++i) {
            if (canUnloadActiveGroup(mm)) {
                break;
            }
            if (i == kMaxRetries - 1) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Timed out waiting for group '{}' to drain requests before swap to '{}'",
                    previousGroup, groupName);
                return StatusCode::GROUP_UNLOAD_BLOCKED;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
        }
        auto unloadStatus = unloadGroup(previousGroup, mm, /*urgent=*/true);
        if (!unloadStatus.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to unload group '{}': {}", previousGroup, unloadStatus.string());
            return unloadStatus;
        }
    }
    return loadGroup(groupName, mm, requestedServable);
}

    [[nodiscard]] Status ServableGroupManager::ensureServableLoaded(const std::string& servableName, ModelManager& mm) {
    const std::string groupName = getGroupForServable(servableName);
    if (groupName.empty()) {
        // Not managed by group manager - let normal flow handle it
        return StatusCode::OK;
    }
    if (isGroupLoaded(groupName) && mm.isServableAvailable(servableName)) {
        recordActivity();
        return StatusCode::OK;
    }

    // Serialize group swaps
    std::lock_guard<std::mutex> swapLock(loadUnloadMtx);

    // Double-check after acquiring the lock
    if (isGroupLoaded(groupName) && mm.isServableAvailable(servableName)) {
        recordActivity();
        return StatusCode::OK;
    }

    // Group is resident but this member is not - e.g. it was slept on its own per-graph
    // timeout, or it failed during the group load. No reason to swap the whole group.
    if (isGroupLoaded(groupName)) {
        auto status = mm.requestServableWakeUp(servableName, /*urgent=*/true).get();
        if (!status.ok()) {
            return status;
        }
        recordActivity();
        return StatusCode::OK;
    }

    return swapToGroup(groupName, mm, servableName);
}

void ServableGroupManager::unloadActiveGroupIfIdle(ModelManager& mm) {
    if (!isEnabled()) {
        return;
    }
    std::string groupToUnload = getActiveGroupName();
    if (groupToUnload.empty()) {
        return;
    }

    // Check if we have a permanent group as active (should not happen, but safety)
    {
        std::shared_lock lock(groupsMtx);
        auto it = groups.find(groupToUnload);
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
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Skipping idle unload of group '{}': active requests in flight", groupToUnload);
        return;
    }

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Idle unloading model group '{}' after {}us timeout", groupToUnload, idleTimeoutMicroseconds);
    std::lock_guard<std::mutex> swapLock(loadUnloadMtx);
    // Re-check after acquiring lock
    groupToUnload = getActiveGroupName();
    if (groupToUnload.empty()) {
        return;
    }
    if (!canUnloadActiveGroup(mm)) {
        return;
    }
    auto status = unloadGroup(groupToUnload, mm, /*urgent=*/false);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to idle unload group '{}': {}", groupToUnload, status.string());
    }
}

}  // namespace ovms
