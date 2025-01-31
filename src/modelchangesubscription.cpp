//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "modelchangesubscription.hpp"

#include <exception>
#include <sstream>

#include "dags/pipelinedefinition.hpp"

namespace ovms {
void ModelChangeSubscription::subscribe(PipelineDefinition& pd) {
    SPDLOG_INFO("Subscription to {} from {}", ownerName, pd.getName());
    if (subscriptions.find(pd.getName()) != subscriptions.end()) {
        std::stringstream ss;
        ss << "Tried to subscribe pipeline:" << pd.getName() << " to:" << ownerName;
        ss << ", but this pipeline was already subscribed";
        const std::string details = ss.str();
        SPDLOG_ERROR("{}", details.c_str());
        throw std::logic_error(details);
    }
    subscriptions.insert({pd.getName(), pd});
}

void ModelChangeSubscription::unsubscribe(PipelineDefinition& pd) {
    SPDLOG_INFO("Subscription to {} from {} removed", ownerName, pd.getName());
    auto numberOfErased = subscriptions.erase(pd.getName());
    if (0 == numberOfErased) {
        std::stringstream ss;
        ss << "Tried to unsubscribe pipeline:" << pd.getName() << " to:" << ownerName;
        ss << ", but this pipeline was never subscribed";
        const std::string details = ss.str();
        SPDLOG_ERROR("{}", details.c_str());
        throw std::logic_error(details);
    }
}

void ModelChangeSubscription::notifySubscribers() {
    if (subscriptions.size() == 0) {
        return;
    }
    SPDLOG_INFO("Notified subscribers of: {}", ownerName);
    for (auto& [pipelineName, pipelineDefinition] : subscriptions) {
        pipelineDefinition.notifyUsedModelChanged(ownerName);
    }
}
}  // namespace ovms
