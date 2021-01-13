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
#include <sstream>
#include <utility>

#include "logging.hpp"

namespace ovms {

/*NodeSessionMetadata::NodeSessionMetadata(const NodeSessionMetadata& metadata) : details(metadata.details) {}

NodeSessionMetadata::NodeSessionMetadata(const NodeSessionMetadata&& metadata) : details(std::move(metadata.details)) {}
*/
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
    std::vector<NodeSessionMetadata> metas(subsessionSize);
    uint counter = 0;
    for (auto& meta : metas) {
        meta.details = this->details;
        meta.details.insert({nodeName, {counter, subsessionSize}});
        ++counter;
    }
    return std::move(metas);
}

std::string NodeSessionMetadata::getSessionKey(const std::set<std::string>& ignoredNodeNames) const {
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
    for (auto& [nodeName, subsessionPair] : details) {
        if (ignoredNodeNames.find(nodeName) != ignoredNodeNames.end()) {
            continue;
        }
        if (ss.tellp() > 0) {
            ss << "_";
        }
        ss << nodeName << "_" << std::get<0>(subsessionPair);
    }
    return ss.str();
}

NodeSessionMetadata NodeSessionMetadata::getCollapsedSessionMetadata(const std::set<std::string>& ignoredNodeNames) const {
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
    NodeSessionMetadata newMeta;
    std::copy_if(
        std::begin(details),
        std::end(details),
        std::inserter(newMeta.details, newMeta.details.begin()),
        [&ignoredNodeNames](auto& keyValuePair) {
            return ignoredNodeNames.find(keyValuePair.first) == ignoredNodeNames.end();
        });
    return std::move(newMeta);
}

session_id_t NodeSessionMetadata::getSubsessionSize(const std::string& subsessionName) const {
    auto it = details.find(subsessionName);
    if (it == details.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to get non-existing subsession: {} size", subsessionName);
        throw std::logic_error("Tried to take non existing subsession size");
    }
    return std::get<1>(it->second);
}
}  // namespace ovms
