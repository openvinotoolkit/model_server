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
#include "nodesession.hpp"

#include "../logging.hpp"
#include "../status.hpp"
#include "../tensor_utils.hpp"
#include "../timer.hpp"
#include "gathernodeinputhandler.hpp"
#include "nodeinputhandler.hpp"

namespace ovms {
NodeSession::~NodeSession() = default;

const NodeSessionMetadata& NodeSession::getNodeSessionMetadata() const {
    return this->metadata;
}

Status NodeSession::setInput(const std::string& inputName, TensorWithSource& tensor, session_id_t shardId) {
    return inputHandler->setInput(inputName, tensor, shardId);
}

static std::unique_ptr<NodeInputHandler> createNodeInputHandler(uint32_t inputsCount, const CollapseDetails& collapsingDetails) {
    if (collapsingDetails.collapsedSessionNames.size() == 0) {
        return std::make_unique<NodeInputHandler>(inputsCount);
    } else {
        return std::make_unique<GatherNodeInputHandler>(inputsCount, collapsingDetails);
    }
}

NodeSession::NodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails) :
    metadata(metadata),
    sessionKey(metadata.getSessionKey()),
    nodeName(nodeName),
    timer(std::make_unique<Timer<TIMER_END>>()),
    inputHandler(createNodeInputHandler(inputsCount, collapsingDetails)) {}

bool NodeSession::isReady() const {
    bool isReady = inputHandler->isReady();
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "node: {} session: {} isReady: {}", getName(), getSessionKey(), isReady);
    return isReady;
}

Status NodeSession::notifyFinishedDependency() {
    return this->inputHandler->notifyFinishedDependency();
}

Timer<TIMER_END>& NodeSession::getTimer() const {
    return *this->timer;
}

ReleaseSessionGuard::ReleaseSessionGuard(NodeSession& nodeSession) :
    nodeSession(nodeSession) {}

ReleaseSessionGuard::~ReleaseSessionGuard() {
    nodeSession.release();
}
}  // namespace ovms
