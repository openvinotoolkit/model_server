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

#include "../dags/dl_node.hpp"
#include "../dags/dlnodesession.hpp"
#include "../dags/gathernodeinputhandler.hpp"
#include "../dags/node.hpp"
#include "../dags/nodeinputhandler.hpp"
#include "../dags/nodestreamidguard.hpp"
#include "../logging.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../ov_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::ElementsAre;

static const std::string mockerDemultiplexerNodeOutputName = "mockedDemultiplexerOutputName";
static const std::string mockerDemultiplexerNodeOutputName2 = "mockedDemultiplexerOutputName2";

class DemultiplexerDLNode : public DLNode {
public:
    DemultiplexerDLNode(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion, ModelManager& modelManager, std::unordered_map<std::string, std::string> nodeOutputNameAlias, std::optional<int32_t> demultiplyCount, const NodeSessionMetadata& meta) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, demultiplyCount.value_or(0)) {
        // createSession to have source session for fetchResults()
        CollapseDetails collapsingDetails;
        std::unique_ptr<NodeSession> nodeSession = createNodeSession(meta, collapsingDetails);
        auto emplacePair = nodeSessions.emplace(meta.getSessionKey(), std::move(nodeSession));
        EXPECT_TRUE(emplacePair.second);
    }

    void setFetchResult(const TensorWithSourceMap& intermediateResults) {
        this->intermediateResults = intermediateResults;
    }

    using Node::fetchResults;
    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) {
        const auto& sessionMetadata = nodeSession.getNodeSessionMetadata();
        const auto sessionKey = sessionMetadata.getSessionKey();
        std::pair<NodeSessionMetadata, TensorWithSourceMap> metaTensorsPair{sessionMetadata, std::move(intermediateResults)};
        nodeSessionOutputs.emplace(sessionKey, std::move(metaTensorsPair));
        return StatusCode::OK;
    }

private:
    TensorWithSourceMap intermediateResults;
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
    std::vector<float> tensorDataNonDemultiplexed2(tensorsData[0].size() * demultiplyCount);
    std::copy(tensorsData[0].begin(), tensorsData[0].end(), tensorDataNonDemultiplexed.begin());
    std::copy(tensorsData[1].begin(), tensorsData[1].end(), tensorDataNonDemultiplexed.begin() + tensorsData[0].size());
    std::copy(tensorsData[0].begin(), tensorsData[0].end(), tensorDataNonDemultiplexed2.begin());
    std::copy(tensorsData[1].begin(), tensorsData[1].end(), tensorDataNonDemultiplexed2.begin() + tensorsData[0].size());
    TensorWithSourceMap intermediateResultTensors{
        {mockerDemultiplexerNodeOutputName, TensorWithSource(createSharedTensor(precision, shape, tensorDataNonDemultiplexed.data()))},
        {mockerDemultiplexerNodeOutputName2, TensorWithSource(createSharedTensor(precision, shape, tensorDataNonDemultiplexed2.data()))}};
    // construct demultiplexer node
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);
    demultiplexerNode.setFetchResult(intermediateResultTensors);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(sessionResults.size(), demultiplyCount);
    auto demultiplexedMetadata = meta.generateSubsessions(demultiplexerNodeName, demultiplyCount);
    ASSERT_EQ(demultiplexedMetadata.size(), demultiplyCount);
    for (size_t shardId = 0; shardId < demultiplyCount; ++shardId) {
        SessionResult& sessionResult = sessionResults.at(demultiplexedMetadata[shardId].getSessionKey());
        ASSERT_EQ(sessionResult.first.getSessionKey(), demultiplexedMetadata[shardId].getSessionKey());
        for (auto& [tensorName, tensorWithSource] : sessionResult.second) {
            auto& tensor = tensorWithSource.getActualTensor();
            EXPECT_THAT(tensorName, AnyOf(Eq(mockerDemultiplexerNodeOutputName),
                                        Eq(mockerDemultiplexerNodeOutputName2)));
            ASSERT_EQ(tensorsData[shardId].size(), tensor.get_size());
            ASSERT_THAT(tensor.get_shape(), ElementsAre(1, tensorsData[shardId].size()));
            EXPECT_EQ(std::memcmp(tensor.data(), tensorsData[shardId].data(), tensor.get_byte_size()), 0) << "Failed comparison for shard: " << shardId << " tensorName: " << tensorName;
            EXPECT_THAT(std::vector<float>((float*)tensor.data(), (float*)tensor.data() + tensor.get_size()),
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
    TensorWithSourceMap intermediateResultTensors{
        {mockerDemultiplexerNodeOutputName, TensorWithSource(createSharedTensor(precision, shape, tensorData.data()))}};
    // construct demultiplexer node
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultTensors);
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
    TensorWithSourceMap intermediateResultTensors{
        {mockerDemultiplexerNodeOutputName, TensorWithSource(createSharedTensor(precision, shape, tensorData.data()))}};
    // construct demultiplexer node
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultTensors);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY);
}

TEST(DemultiplexerTest, ShardsShareDataWithSourceTensor) {
    std::vector<float> tensorData{8, 9};
    const uint16_t demultiplyCount = tensorData.size();
    // imitate (2, 1, 1) to demultiply to 2x (1, 1)
    const std::vector<size_t> shape{demultiplyCount, 1, 1};
    const auto precision{ov::element::Type_t::f32};
    auto intermediateTensor = createSharedTensor(precision, shape, tensorData.data());
    TensorWithSourceMap intermediateResultTensors{
        {mockerDemultiplexerNodeOutputName, TensorWithSource(intermediateTensor)}};
    // construct demultiplexer node
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    ConstructorEnabledModelManager manager;
    std::string demultiplexerNodeName("node");
    DemultiplexerDLNode demultiplexerNode(demultiplexerNodeName, "model", 1, manager, std::unordered_map<std::string, std::string>{{"NOT_USED", "NOT_USED"}}, demultiplyCount, meta);  // demultiplexer expects (1, 3, x1, ..., xN);
    demultiplexerNode.setFetchResult(intermediateResultTensors);
    SessionResults sessionResults;
    session_key_t sessionKey = meta.getSessionKey();
    // perform test
    auto status = demultiplexerNode.fetchResults(sessionKey, sessionResults);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(sessionResults.size(), demultiplyCount);
    auto demultiplexedMetadata = meta.generateSubsessions(demultiplexerNodeName, demultiplyCount);
    for (size_t shardId = 0; shardId < demultiplyCount; ++shardId) {
        auto& sessionResult = sessionResults.at(demultiplexedMetadata[shardId].getSessionKey());
        ASSERT_EQ(sessionResult.first.getSessionKey(), demultiplexedMetadata[shardId].getSessionKey());
        for (auto& [tensorName, tensorWithSource] : sessionResult.second) {
            auto expectedPtr = ((char*)intermediateTensor.data()) + sizeof(float) * shardId;
            auto actualPtr = (char*)tensorWithSource.getActualTensor().data();
            // Actual sharded tensor need to share data with original tensor
            EXPECT_THAT(actualPtr, expectedPtr);
            // Source tensor ptr need to be equal to original intermediate tensor passed to demultiplexer
            EXPECT_THAT(tensorWithSource.getSourceTensor().data(), intermediateTensor.data());
        }
    }
}

// In case of demultiplexer right before gather, the input tensor object (TensorWithShape) will contain source tensor.
// This test ensures the gathering prodcues tensor with correct data (which implies it takes care of source tensor ownership).
TEST(DemultiplexerTest, GatherShardsWithExistingSourceTensors) {
    // Mock DLNode session to be possible to get inputs from its input handler.
    class MockNodeSession : public DLNodeSession {
    public:
        using DLNodeSession::DLNodeSession;

        const TensorMap& getInputs() {
            return this->inputHandler->getInputs();
        }
    };

    // Mock DLNode to inject factory method to be able to create Node with mocked NodeSession.
    class MockDLNode : public DLNode {
    protected:
        std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) override {
            return std::make_unique<MockNodeSession>(metadata, getName(), previous.size(), collapsingDetails,
                this->modelManager, this->modelName, this->modelVersion.value_or(0));
        }

    public:
        using DLNode::DLNode;

        const TensorMap& getInputs(const session_key_t& key) {
            auto& nodeSession = dynamic_cast<MockNodeSession&>(this->getNodeSession(key));
            return nodeSession.getInputs();
        }
    };

    ConstructorEnabledModelManager manager;
    int32_t demultiplyCount = 3;

    auto dl_demulti = std::make_unique<DLNode>(
        std::string("dummy_node_1"),
        std::string("dummy_model"),
        1, manager,
        std::unordered_map<std::string, std::string>{{DUMMY_MODEL_INPUT_NAME, DUMMY_MODEL_INPUT_NAME}},
        demultiplyCount);
    auto dl_gather = std::make_unique<MockDLNode>(
        std::string("dummy_node_2"),
        std::string("dummy_model"),
        1, manager,
        std::unordered_map<std::string, std::string>{{DUMMY_MODEL_INPUT_NAME, DUMMY_MODEL_INPUT_NAME}},
        std::nullopt,
        std::set<std::string>{"dummy_node_1"});

    dl_demulti->addDependant(*dl_gather);
    dl_gather->addDependency(*dl_demulti, Aliases{{DUMMY_MODEL_INPUT_NAME, DUMMY_MODEL_INPUT_NAME}});

    size_t size = 2;

    // Prepare source tensors
    std::vector<float> data{3.2, 5.9};
    std::vector<std::shared_ptr<ov::Tensor>> tensors;
    for (int i = 0; i < demultiplyCount; i++) {
        tensors.push_back(std::make_shared<ov::Tensor>(ov::element::f32, ov::Shape{1, size}));
        std::memcpy(tensors[i]->data(), data.data(), sizeof(float) * size);
    }

    // Prepare session results and pass to ::setInputs method - source and actual tensors.
    // This imitates demultiplexing (or any process in DAG that forces tensor to keep its source tensor).
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subMetas = meta.generateSubsessions("dummy_node_1", demultiplyCount);
    for (int i = 0; i < demultiplyCount; i++) {
        TensorWithSource tensorWithSource{
            ov::Tensor(tensors[i]->get_element_type(), {1, size}, tensors[i]->data()),
            *tensors[i]};
        TensorWithSourceMap tensorMap{{DUMMY_MODEL_INPUT_NAME, tensorWithSource}};
        SessionResult result{subMetas[i], tensorMap};
        SessionResults results{
            {"unused_session_key", result}};
        // Last ::setInput will trigger gathering step.
        dl_gather->setInputs(*dl_demulti, results);
        tensors[i].reset();
    }

    // Fetch session and its gathered input.
    auto sessions = dl_gather->getReadySessions();
    ASSERT_EQ(sessions.size(), 1);
    const auto& inputs = dl_gather->getInputs(sessions[0]);
    ASSERT_EQ(inputs.size(), 1);
    auto& input = inputs.begin()->second;

    // Test tensor metadata
    ASSERT_EQ(input.get_element_type(), ov::element::f32);
    ASSERT_EQ(input.get_shape(), (ov::Shape{(size_t)demultiplyCount, 1, size}));

    // Test actual data
    std::vector<float> res;
    for (int i = 0; i < demultiplyCount; i++) {
        res.insert(res.end(), data.begin(), data.end());
    }
    ASSERT_EQ(res.size(), input.get_size());
    EXPECT_EQ(
        std::memcmp(
            (char*)res.data(),
            (char*)input.data(),
            input.get_byte_size()),
        0)
        << "Failed comparison";
}
