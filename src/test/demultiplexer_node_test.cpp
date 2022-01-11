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
    void setFetchResult(std::shared_ptr<ov::runtime::Tensor>& intermediateResultTensor) {
        this->intermediateResultTensor = intermediateResultTensor;
    }
    using Node::fetchResults;
    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
        const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
        const auto sessionKey = sessionMetadata.getSessionKey();
        std::shared_ptr<ov::runtime::Tensor> secondOutput;
        EXPECT_EQ(tensorClone(secondOutput, *intermediateResultTensor), StatusCode::OK);
        TensorMap tensors{{mockerDemutliplexerNodeOutputName, intermediateResultTensor},
            {mockerDemutliplexerNodeOutputName2, secondOutput}};
        std::pair<NodeSessionMetadata, TensorMap> metaTensorsPair{sessionMetadata, std::move(tensors)};
        nodeSessionOutputs.emplace(sessionKey, std::move(metaTensorsPair));
        return StatusCode::OK;
    }

private:
    std::shared_ptr<ov::runtime::Tensor> intermediateResultTensor;
};

using ::testing::AnyOf;
using ::testing::Eq;

TEST(DemultiplexerTest, CheckDemultipliedTensorsMultipleOutputs) {
    // prepare pre demultiplexer tensor
    std::vector<std::vector<float>> tensorsData{
        {-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67, 0, 8},
        {4, 42, 35, -2, 13, 2, -1, 0.9, -0.3, 4.67, 100, 80}};
    const uint16_t demultiplyCount = tensorsData.size();
    const std::vector<size_t> shape{demultiplyCount, 1, tensorsData[0].size()};
    const auto precision{ov::element::Type_t::f32};
    std::vector<float> tensorDataNonDemultiplexed(tensorsData[0].size() * demultiplyCount);
    std::copy(tensorsData[0].begin(), tensorsData[0].end(), tensorDataNonDemultiplexed.begin());
    std::copy(tensorsData[1].begin(), tensorsData[1].end(), tensorDataNonDemultiplexed.begin() + tensorsData[0].size());
    std::shared_ptr<ov::runtime::Tensor> intermediateResultTensor = createSharedTensor(precision, shape, tensorDataNonDemultiplexed.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);
    demultiplexerNode.setFetchResult(intermediateResultTensor);
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
        for (auto& [tensorName, tensor] : sessionResult.second) {
            EXPECT_THAT(tensorName, AnyOf(Eq(mockerDemutliplexerNodeOutputName),
                                      Eq(mockerDemutliplexerNodeOutputName2)));
            ASSERT_EQ(tensorsData[shardId].size(), tensor->get_size());
            ASSERT_THAT(tensor->get_shape(), ElementsAre(1, tensorsData[shardId].size()));
            EXPECT_EQ(std::memcmp(tensor->data(), tensorsData[shardId].data(), tensor->get_byte_size()), 0) << "Failed comparison for shard: " << shardId << " tensorName: " << tensorName;
            EXPECT_THAT(std::vector<float>((float*)tensor->data(), (float*)tensor->data() + tensor->get_size()),
                ::testing::ElementsAreArray(tensorsData[shardId]));
        }
    }
}

TEST(DemultiplexerTest, DemultiplyShouldReturnErrorWhenWrongOutputDimensions) {
    const uint16_t demultiplyCount = 3;
    std::vector<float> tensorData{-1, 4, 5, 12, 3, 52};
    // imitate (1, 2, 3) but shoudl be (1,3,x1, ..., xN)
    const std::vector<size_t> shape{1, demultiplyCount - 1, 3};
    const auto precision{ov::element::Type_t::f32};
    std::shared_ptr<ov::runtime::Tensor> intermediateResultTensor = createSharedTensor(precision, shape, tensorData.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultTensor);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::PIPELINE_WRONG_DIMENSION_SIZE_TO_DEMULTIPLY);
}

TEST(DemultiplexerTest, DemultiplyShouldReturnErrorWhenNotEnoughDimensionsInOutput) {
    std::vector<float> tensorData{-1, 4, 5, 12, 3, 52};
    const uint16_t demultiplyCount = tensorData.size();
    // imitate (1, 3) but should be at least (1,3,x1, ..., xN) N >= 1
    const std::vector<size_t> shape{1, demultiplyCount};
    const auto precision{ov::element::Type_t::f32};
    std::shared_ptr<ov::runtime::Tensor> intermediateResultTensor = createSharedTensor(precision, shape, tensorData.data());
    // construct demultiplexer node
    NodeSessionMetadata meta;
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultTensor);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY);
}
