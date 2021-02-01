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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../node.hpp"
#include "../dl_node.hpp"
#include "test_utils.hpp"

using namespace ovms;

class DemultiplexerDLNode : public DLNode {
public:
    DemultiplexerDLNode(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager, std::unordered_map<std::string, std::string> nodeOutputNameAlias, std::optional<uint32_t> demultiplyCount, const NodeSessionMetadata& meta) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount.value_or(0)) {
        this->demultiplexCount = demultiplexCount;
        // createSession to have source session for fetchResults()
        std::unique_ptr<NodeSession> nodeSession = createNodeSession(meta, 1);
        auto emplacePair = nodeSessions.emplace(meta.getSessionKey(), std::move(nodeSession));
        EXPECT_TRUE(emplacePair.second);
    }
    void setFetchResult(InferenceEngine::Blob::Ptr& blobToReturn) {
        this->blobToReturn = blobToReturn;
    }
    using Node::fetchResults;
    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
    const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
    const auto sessionKey = sessionMetadata.getSessionKey();
    BlobMap blobs{{std::string("a"), blobToReturn}};
    std::pair<NodeSessionMetadata, BlobMap> metaBlobsPair{sessionMetadata, blobs};
    nodeSessionOutputs.emplace(sessionKey, std::move(metaBlobsPair));
    return StatusCode::OK;

    }
private:
    InferenceEngine::Blob::Ptr blobToReturn;
};

TEST(DemultiplexerTest, CheckDemultipliedBlobs) {
    const uint16_t demultiplyCount = 2;
    // prepare pre demultiplexer blob
    std::vector<float> blobData1{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67, 0, 8};
    std::vector<float> blobData2{4, 42, 35, -2, 13, 2, -1, 0.9, -0.3, 4.67, 100, 80};
    const std::vector<size_t> shape{1, demultiplyCount, blobData1.size()};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::CHW};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    std::vector<float> blobDataNonDemultiplexed(blobData1.size() * demultiplyCount);
    std::copy(blobData1.begin(), blobData1.end(), blobDataNonDemultiplexed.begin());
    std::copy(blobData2.begin(), blobData2.end(), blobDataNonDemultiplexed.begin() + blobData1.size());
    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(desc, blobDataNonDemultiplexed.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    DemultiplexerDLNode demultiplexerNode("node", "model", 1, manager, std::unordered_map<std::string, std::string>{{"a", "a"}}, demultiplyCount, meta);
    demultiplexerNode.setFetchResult(inputBlob);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(sessionResults.size(), 2);
    // TODO check for each output metadata key
    // TODO check for output shapes
    // TODO check for output numbers
}

