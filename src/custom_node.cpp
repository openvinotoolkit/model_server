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
#include "custom_node.hpp"

#include <utility>

#include "custom_node_output_allocator.hpp"
#include "customnodesession.hpp"
#include "logging.hpp"
#include "node_library.hpp"
#include "node_library_utils.hpp"

namespace ovms {

CustomNode::CustomNode(
    const std::string& nodeName,
    const NodeLibrary& library,
    const parameters_t& parameters,
    const std::unordered_map<std::string, std::string>& nodeOutputNameAlias,
    std::optional<uint32_t> demultiplyCount,
    std::set<std::string> gatherFromNode,
    void* customNodeLibraryInternalManager) :
    Node(nodeName, demultiplyCount, gatherFromNode),
    library(library),
    parameters(parameters),
    nodeOutputNameAlias(nodeOutputNameAlias),
    libraryParameters(createCustomNodeParamArray(this->parameters)),
    customNodeLibraryInternalManager(customNodeLibraryInternalManager) {
}

Status CustomNode::execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) {
    auto& nodeSession = getNodeSession(sessionKey);
    auto& customNodeSession = static_cast<CustomNodeSession&>(nodeSession);
    return customNodeSession.execute(notifyEndQueue, *this, this->library, this->libraryParameters, this->parameters.size(), customNodeLibraryInternalManager);
}

Status CustomNode::fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    auto& customNodeSession = static_cast<CustomNodeSession&>(nodeSession);
    const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
    SessionResult sessionResults{sessionMetadata, {}};
    auto it = nodeSessionOutputs.emplace(sessionMetadata.getSessionKey(), std::move(sessionResults));
    if (!it.second) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to put node: {} session: {} results in node session outputs",
            getName(), nodeSession.getSessionKey());
        customNodeSession.release();
        return StatusCode::INTERNAL_ERROR;
    }
    auto& metadataBlobResultsPair = it.first->second;
    auto& blobResults = metadataBlobResultsPair.second;
    return this->fetchResults(
        blobResults,
        nodeSession.getSessionKey());
}

Status CustomNode::fetchResults(TensorMap& outputs, session_key_t sessionKey) {
    auto& session = static_cast<CustomNodeSession&>(this->getNodeSession(sessionKey));
    session.clearInputs();

    for (const auto& node : this->next) {
        for (const auto& pair : node.get().getMappingByDependency(*this)) {
            const auto& output_name = pair.first;
            if (outputs.find(output_name) != outputs.end()) {
                continue;
            }
            const auto& realOutputName = this->getRealOutputName(output_name);
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Getting custom node output tensor with name: {}",
                getName(), sessionKey, realOutputName);

            std::shared_ptr<ov::runtime::Tensor> resultTensor;
            auto status = session.fetchResult(realOutputName, resultTensor);
            if (!status.ok()) {
                SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node: {} session: {} Custom node output with name {} is missing",
                    getName(), sessionKey, realOutputName);
                return StatusCode::NODE_LIBRARY_MISSING_OUTPUT;
            }

            outputs.emplace(std::make_pair(output_name, std::move(resultTensor)));
            SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Node: {} session: {} Blob with name {} has been prepared under alias {}",
                getName(), sessionKey, realOutputName, output_name);
        }
    }

    return StatusCode::OK;
}

std::unique_ptr<NodeSession> CustomNode::createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) {
    return std::make_unique<CustomNodeSession>(metadata, getName(), previous.size(), collapsingDetails);
}

}  // namespace ovms
