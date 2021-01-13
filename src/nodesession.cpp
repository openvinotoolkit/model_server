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

#include "logging.hpp"
#include "nodeinputhandler.hpp"
#include "nodeoutputhandler.hpp"

namespace ovms {
NodeSession::~NodeSession() = default;

const NodeSessionMetadata& NodeSession::getNodeSessionMetadata() const {
    return this->metadata;
}

void NodeSession::setInput(const std::string& inputName, InferenceEngine::Blob::Ptr& blobPtr) {
    inputHandler->setInput(inputName, blobPtr);
}

NodeSession::NodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount) :
    metadata(metadata),
    sessionKey(metadata.getSessionKey()),
    nodeName(nodeName),
    inputHandler(std::make_unique<NodeInputHandler>(inputsCount)),
    outputHandler(std::make_unique<NodeOutputHandler>()) {}

NodeSession::NodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount) :
    metadata(std::move(metadata)),
    sessionKey(this->metadata.getSessionKey()),
    nodeName(nodeName),
    inputHandler(std::make_unique<NodeInputHandler>(inputsCount)),
    outputHandler(std::make_unique<NodeOutputHandler>()) {}

bool NodeSession::isReady() const {
    bool isReady = inputHandler->isReady();  // TODO gather input handler will influence result
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "node: {} session: {} isReady: {}", getName(), getSessionKey(), isReady);
    return isReady;
}
void NodeSession::notifyFinishedDependency() {
    this->inputHandler->notifyFinishedDependency();
}
}  // namespace ovms
