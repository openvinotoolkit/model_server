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
#include "nodesessionmetadata.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <utility>

#include "../logging.hpp"

namespace ovms {

NodeSessionMetadata::NodeSessionMetadata() :
    context({ExecutionContext::Interface::GRPC, ExecutionContext::Method::Predict}) {}

NodeSessionMetadata::NodeSessionMetadata(ExecutionContext context) :
    context(context) {}

NodeSessionMetadata::NodeSessionMetadata(const std::unordered_map<std::string, std::tuple<session_id_t, session_id_t>>& details, const std::vector<std::string>& sessionsLevels, ExecutionContext context) :
    details(details),
    sessionsLevels(sessionsLevels),
    context(context) {}

std::vector<NodeSessionMetadata> NodeSessionMetadata::generateSubsessions(const std::string& nodeName, session_id_t subsessionSize) const {
    if (nodeName.size() == 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to generate subsession with empty node name");
        throw std::logic_error("Cannot generate subsession with empty parent name");
    }
    if (details.find(nodeName) != details.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to generate subsession with node name: {} but it already spawned subsession.", nodeName);
        throw std::logic_error("Cannot generate subsession with already used name");
    }
    if (subsessionSize == 0) {
        return {};
    }
    std::vector<NodeSessionMetadata> metas(subsessionSize, *this);
    uint32_t counter = 0;
    for (auto& meta : metas) {
        meta.details.insert({nodeName, {counter, subsessionSize}});
        meta.sessionsLevels.push_back(nodeName);
        meta.cached = false;
        ++counter;
    }
    SPDLOG_LOGGER_TRACE(dag_executor_logger, "Generated subsession levels: {}",
        std::accumulate(metas[0].sessionsLevels.begin(), metas[0].sessionsLevels.end(),
            std::string(), [](const std::string& lhs, const std::string& rhs) {
                if (lhs.empty()) {
                    return rhs;
                }
                return lhs + ", " + rhs; }));
    return metas;
}

std::string NodeSessionMetadata::createSessionKey(const std::set<std::string>& ignoredNodeNames) const {
    if (details.size() == 0) {
        return "";
    }
    if (std::any_of(ignoredNodeNames.begin(),
            ignoredNodeNames.end(),
            [this](auto& ignoredNodeName) {
                bool notFound = (this->details.find(ignoredNodeName) == this->details.end());
                if (notFound) {
                    SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to create session key ignoring subsession name: {} but it does not exist", ignoredNodeName);
                }
                return notFound;
            })) {
        throw std::logic_error("Tried to create session key ignoring non-existing subsession");
    }
    std::stringstream ss;
    size_t j = 0;
    for (int32_t i = sessionsLevels.size() - 1; i >= 0; --i, ++j) {
        if ((ignoredNodeNames.size() > 0) &&
            (ignoredNodeNames.size() > j) &&
            (sessionsLevels.size() >= ignoredNodeNames.size()) &&
            (ignoredNodeNames.find(sessionsLevels[i]) == ignoredNodeNames.end())) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse sessions not in LIFO order. Should collapse: {} first", sessionsLevels[i]);
            throw std::logic_error("Cannot collapse sessions not in LIFO order");
        } else {
            if (j < ignoredNodeNames.size()) {
                continue;
            }
            if (ss.tellp() > 0) {
                ss << "_";
            }
            ss << sessionsLevels[i] << "_" << std::get<0>(details.at(sessionsLevels[i]));
        }
        if (i == 0) {
            break;
        }
    }
    return ss.str();
}

std::string NodeSessionMetadata::getSessionKey(const std::set<std::string>& ignoredNodeNames) const {
    // if set not empty then we need to regenerate the cache and mark it as not cached
    // we don't want to store previous set but we want to limit recreation of the key
    if (ignoredNodeNames.size() != 0) {
        return createSessionKey(ignoredNodeNames);
    }
    if (cached) {
        return this->cachedSessionKey;
    }
    cached = true;
    this->cachedSessionKey = createSessionKey(ignoredNodeNames);
    return this->cachedSessionKey;
}

std::pair<NodeSessionMetadata, CollapseDetails> NodeSessionMetadata::getCollapsedSessionMetadata(const std::set<std::string>& ignoredNodeNames) const {
    if (ignoredNodeNames.size() == 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse subsession with empty set");
        throw std::logic_error("Tried to collapse sessions with empty set");
    }
    if (std::any_of(
            ignoredNodeNames.begin(),
            ignoredNodeNames.end(),
            [this](auto& ignoredNodeName) {
                bool notFound = (this->details.find(ignoredNodeName) == this->details.end());
                if (notFound) {
                    SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse subsession: {} but it does not exist", ignoredNodeName);
                }
                return notFound;
            })) {
        throw std::logic_error("Tried to collapse nonexisting subsession");
    }
    for (size_t i = sessionsLevels.size() - 1; i > sessionsLevels.size() - 1 - ignoredNodeNames.size(); --i) {
        if (ignoredNodeNames.find(sessionsLevels[i]) == ignoredNodeNames.end()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse sessions not in LIFO order. Should collapse: {} first", sessionsLevels[i]);
            throw std::logic_error("Cannot collapse sessions not in LIFO order");
        }
    }

    NodeSessionMetadata newMeta;
    std::copy_if(
        std::begin(details),
        std::end(details),
        std::inserter(newMeta.details, newMeta.details.begin()),
        [&ignoredNodeNames](auto& keyValuePair) {
            return ignoredNodeNames.find(keyValuePair.first) == ignoredNodeNames.end();
        });
    CollapseDetails collapsingDetails;
    for (auto& sessionLevel : sessionsLevels) {
        if (ignoredNodeNames.find(sessionLevel) != ignoredNodeNames.end()) {
            collapsingDetails.collapsedSessionNames.emplace_back(sessionLevel);
            collapsingDetails.collapsedSessionSizes.emplace_back(getSubsessionSize(sessionLevel));
        } else {
            newMeta.sessionsLevels.emplace_back(sessionLevel);
        }
    }
    return {newMeta, std::move(collapsingDetails)};
}

session_id_t NodeSessionMetadata::getSubsessionSize(const std::string& subsessionName) const {
    auto it = details.find(subsessionName);
    if (it == details.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to get non-existing subsession: {} size", subsessionName);
        throw std::logic_error("Tried to take non existing subsession size");
    }
    return std::get<1>(it->second);
}

session_id_t NodeSessionMetadata::getShardId(const std::set<std::string>& collapsedNames) const {
    if (collapsedNames.size() == 0) {
        return 0;
    }
    if (collapsedNames.size() > sessionsLevels.size()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse more subsession levels than exists");
        throw std::logic_error("Tried to collapse more subsession levels than exists");
    }
    for (size_t i = sessionsLevels.size() - 1; i > sessionsLevels.size() - 1 - collapsedNames.size(); --i) {
        if (collapsedNames.find(sessionsLevels[i]) == collapsedNames.end()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to collapse sessions not in LIFO order. Should collapse: {} first, but tried to: {}. SubsessionLevels: {}",
                sessionsLevels[i],
                std::accumulate(collapsedNames.begin(),
                    collapsedNames.end(),
                    std::string(),
                    [](const std::string& lhs, const std::string& rhs) {
                        if (lhs.empty()) {
                            return rhs;
                        }
                        return lhs + ", " + rhs;
                    }),
                std::accumulate(sessionsLevels.begin(),
                    sessionsLevels.end(),
                    std::string(),
                    [](const std::string& lhs, const std::string& rhs) {
                        if (lhs.empty()) {
                            return rhs;
                        }
                        return lhs + ", " + rhs;
                    }));
            throw std::logic_error("Cannot collapse sessions not in LIFO order");
        }
    }
    session_id_t multiplyFactor = 1;
    session_id_t shardId = 0;
    for (size_t i = 0; i < collapsedNames.size(); ++i) {
        const auto& subsessionDetails = details.at(*(sessionsLevels.rbegin() + i));
        const auto& [id, sessionSize] = subsessionDetails;
        shardId += multiplyFactor * id;
        multiplyFactor *= sessionSize;
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "getShardId calculation step shardId: {}, multiplyFactor: {}, subsessionId: {}, sessionSize: {}",
            shardId, multiplyFactor, id, sessionSize);
    }
    return shardId;
}

ExecutionContext NodeSessionMetadata::getContext() const { return this->context; }

}  // namespace ovms
