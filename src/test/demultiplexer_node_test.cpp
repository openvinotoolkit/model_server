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

#include "../dl_node.hpp"
#include "../logging.hpp"
#include "../node.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::ElementsAre;

static const std::string mockerDemutliplexerNodeOutputName = "mockedDemultiplexerOutputName";
static const std::string mockerDemutliplexerNodeOutputName2 = "mockedDemultiplexerOutputName2";

class DemultiplexerDLNode : public DLNode {
public:
    DemultiplexerDLNode(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager, std::unordered_map<std::string, std::string> nodeOutputNameAlias, std::optional<uint32_t> demultiplyCount, const NodeSessionMetadata& meta) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount.value_or(0)) {
        // createSession to have source session for fetchResults()
        CollapseDetails collapsingDetails;
        std::unique_ptr<NodeSession> nodeSession = createNodeSession(meta, collapsingDetails);
        auto emplacePair = nodeSessions.emplace(meta.getSessionKey(), std::move(nodeSession));
        EXPECT_TRUE(emplacePair.second);
    }
    void setFetchResult(InferenceEngine::Blob::Ptr& intermediateResultBlob) {
        this->intermediateResultBlob = intermediateResultBlob;
    }
    using Node::fetchResults;
    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
        const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
        const auto sessionKey = sessionMetadata.getSessionKey();
        InferenceEngine::Blob::Ptr secondOutput;
        EXPECT_EQ(blobClone(secondOutput, intermediateResultBlob), StatusCode::OK);
        BlobMap blobs{{mockerDemutliplexerNodeOutputName, intermediateResultBlob},
            {mockerDemutliplexerNodeOutputName2, secondOutput}};
        std::pair<NodeSessionMetadata, BlobMap> metaBlobsPair{sessionMetadata, std::move(blobs)};
        nodeSessionOutputs.emplace(sessionKey, std::move(metaBlobsPair));
        return StatusCode::OK;
    }

private:
    InferenceEngine::Blob::Ptr intermediateResultBlob;
};

using ::testing::AnyOf;
using ::testing::Eq;

TEST(DemultiplexerTest, CheckDemultipliedBlobsMultipleOutputs) {
    const uint16_t demultiplyCount = 2;
    // prepare pre demultiplexer blob
    std::vector<std::vector<float>> blobsData{
        {-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67, 0, 8},
        {4, 42, 35, -2, 13, 2, -1, 0.9, -0.3, 4.67, 100, 80}};
    const std::vector<size_t> shape{demultiplyCount, 1, blobsData[0].size()};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::CHW};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    std::vector<float> blobDataNonDemultiplexed(blobsData[0].size() * demultiplyCount);
    std::copy(blobsData[0].begin(), blobsData[0].end(), blobDataNonDemultiplexed.begin());
    std::copy(blobsData[1].begin(), blobsData[1].end(), blobDataNonDemultiplexed.begin() + blobsData[0].size());
    InferenceEngine::Blob::Ptr intermediateResultBlob = InferenceEngine::make_shared_blob<float>(desc, blobDataNonDemultiplexed.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);
    demultiplexerNode.setFetchResult(intermediateResultBlob);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(sessionResults.size(), demultiplyCount);
    auto demultiplexedMetadata = meta.generateSubsessions(demultiplexerNodeName, demultiplyCount);
    ASSERT_EQ(demultiplexedMetadata.size(), demultiplyCount);
    for (size_t shardId = 0; shardId < demultiplyCount; ++shardId) {
        auto& sessionResult = sessionResults[demultiplexedMetadata[shardId].getSessionKey()];
        ASSERT_EQ(sessionResult.first.getSessionKey(), demultiplexedMetadata[shardId].getSessionKey());
        for (auto& [blobName, blob] : sessionResult.second) {
            EXPECT_THAT(blobName, AnyOf(Eq(mockerDemutliplexerNodeOutputName),
                                      Eq(mockerDemutliplexerNodeOutputName2)));
            ASSERT_EQ(blobsData[shardId].size(), blob->size());
            ASSERT_THAT(blob->getTensorDesc().getDims(), ElementsAre(1, blobsData[shardId].size()));
            EXPECT_EQ(std::memcmp((char*)((const void*)InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap()), blobsData[shardId].data(), blob->byteSize()), 0) << "Failed comparison for shard: " << shardId << " blobName: " << blobName;
            EXPECT_THAT(std::vector<float>((const float*)(const void*)(InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap()), (const float*)(const void*)InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap() + blob->size()),
                ::testing::ElementsAreArray(blobsData[shardId]));
        }
    }
}

TEST(DemultiplexerTest, DemultiplyShouldReturnErrorWhenWrongOutputDimensions) {
    const uint16_t demultiplyCount = 3;
    std::vector<float> blobData{-1, 4, 5, 12, 3, 52};
    // imitate (1, 2, 3) but shoudl be (1,3,x1, ..., xN)
    const std::vector<size_t> shape{1, demultiplyCount - 1, 3};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::CHW};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    InferenceEngine::Blob::Ptr intermediateResultBlob = InferenceEngine::make_shared_blob<float>(desc, blobData.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultBlob);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::PIPELINE_WRONG_DIMENSION_SIZE_TO_DEMULTIPLY);
}

TEST(DemultiplexerTest, DemultiplyShouldReturnErrorWhenNotEnoughDimensionsInOutput) {
    std::vector<float> blobData{-1, 4, 5, 12, 3, 52};
    const uint16_t demultiplyCount = blobData.size();
    // imitate (1, 3) but should be at least (1,3,x1, ..., xN) N >= 1
    const std::vector<size_t> shape{1, demultiplyCount};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::NC};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    InferenceEngine::Blob::Ptr intermediateResultBlob = InferenceEngine::make_shared_blob<float>(desc, blobData.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultBlob);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY);
}
