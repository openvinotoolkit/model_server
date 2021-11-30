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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../dl_node.hpp"
#include "../dlnodesession.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../gathernodeinputhandler.hpp"
#include "../logging.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../nodeinputhandler.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "../pipelinedefinition.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::ElementsAre;
using testing::Return;

class GatherNodeInputHandlerTest : public ::testing::Test {};

TEST_F(GatherNodeInputHandlerTest, ThreePredecessorNodesWithSubsessionSize2) {
    // simulate all 3 inputs comming from different predecessor nodes
    // with session demultiplexed to 2 shards
    const uint32_t shardsCount = 2;  // subsessionSize/demultiplyCount
    std::vector<std::string> inputNames{"a", "b"};
    std::vector<std::vector<size_t>> shapes{{1, 10}, {1, 2}};
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::f32, ov::element::Type_t::f32};
    std::vector<std::vector<float>> blobsData{{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67}, {1., 3}};
    std::vector<InferenceEngine::Blob::Ptr> inputBlobs{createSharedTensor(precisions[0], shapes[0], blobsData[0].data()), createSharedTensor(precisions[1], shapes[1], blobsData[1].data())};
    NodeSessionMetadata meta;
    const std::string demultiplexerName = "NOT_IMPORTANT_NAME";
    auto newMeta = meta.generateSubsessions(demultiplexerName, shardsCount)[0];
    auto [_, collapsingDetails] = newMeta.getCollapsedSessionMetadata({demultiplexerName});
    GatherNodeInputHandler gInputHandler(inputNames.size(), collapsingDetails);
    for (session_id_t j = 0; j < shardsCount; ++j) {
        for (size_t i = 0; i < inputNames.size(); ++i) {
            EXPECT_FALSE(gInputHandler.isReady());
            gInputHandler.setInput(inputNames[i], inputBlobs[i], j);
            // each input comming from different node so we call notify each time
            ASSERT_EQ(gInputHandler.notifyFinishedDependency(), StatusCode::OK);
        }
    }
    EXPECT_TRUE(gInputHandler.isReady());
    const auto tensorMap = gInputHandler.getInputs();
    EXPECT_EQ(tensorMap.size(), inputNames.size());

    std::vector<std::vector<float>> resultBlobsData(inputNames.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
        resultBlobsData[i].reserve(blobsData[i].size() * shardsCount);
        std::copy(blobsData[i].begin(), blobsData[i].end(), resultBlobsData[i].begin());
        std::copy(blobsData[i].begin(), blobsData[i].end(), resultBlobsData[i].begin() + blobsData[i].size());
    }
    for (size_t i = 0; i < inputNames.size(); ++i) {
        const auto& tensor = tensorMap.at(inputNames[i]);
        EXPECT_EQ(tensor->get_size(), blobsData[i].size() * shardsCount);
        EXPECT_THAT(tensor->get_shape(), ElementsAre(shardsCount, 1, blobsData[i].size()));
        EXPECT_EQ(std::memcmp((char*)((const void*)(tensor.data())), resultBlobsData[i].data(), resultBlobsData[i].size() * sizeof(float)), 0);
    }
}

TEST_F(GatherNodeInputHandlerTest, GatheringOnTwoDemultiplexersAtOnce) {
    const std::string inputName{"a"};
    const size_t elementCountPerShard = 10;
    std::vector<size_t> shape{1, elementCountPerShard};
    ov::element::Type_t precision{ov::element::Type_t::f32};
    const std::vector<session_id_t> demultiplyCounts{3, 5};  // 3 for first demultiply, 5 for second
    const std::vector<std::string> demultiplexerNodeNames{"firstDemultiplexer", "secondDemultiplexer"};
    NodeSessionMetadata meta;
    auto firstLevelMetas = meta.generateSubsessions(demultiplexerNodeNames[0], demultiplyCounts[0]);
    std::vector<std::vector<NodeSessionMetadata>> metadatas(demultiplyCounts[0]);
    for (size_t i = 0; i < demultiplyCounts[0]; ++i) {
        metadatas[i] = firstLevelMetas[i].generateSubsessions(demultiplexerNodeNames[1], demultiplyCounts[1]);
    }

    const size_t numberOfShards = std::accumulate(demultiplyCounts.begin(), demultiplyCounts.end(), 1, std::multiplies<size_t>());
    const size_t numberOfElementsInGatheredBlob = elementCountPerShard * numberOfShards;
    std::vector<float> blobsData(numberOfElementsInGatheredBlob);
    std::iota(blobsData.begin(), blobsData.end(), 0.1);
    std::vector<ov::runtime::Tensor> inputBlobs(numberOfShards);
    GatherNodeInputHandler gInputHandler(1, {demultiplexerNodeNames, demultiplyCounts});
    Status status;
    for (size_t i = 0; i < demultiplyCounts[0]; ++i) {
        for (size_t j = 0; j < demultiplyCounts[1]; ++j) {
            auto index = i * demultiplyCounts[1] + j;
            auto tensor = createSharedTensor(precision, shape, (void*)(blobsData.data() + index * elementCountPerShard));
            ASSERT_FALSE(gInputHandler.isReady());
            SPDLOG_DEBUG("i: {}, j: {}, metadatas.size: {}, metadatas[i].size() :{}", i, j, metadatas.size(), metadatas[i].size());
            auto shardId = metadatas[i][j].getShardId({demultiplexerNodeNames[0], demultiplexerNodeNames[1]});
            status = gInputHandler.setInput(inputName,
                tensor,
                shardId);
            ASSERT_EQ(status, StatusCode::OK) << status.string();
            gInputHandler.notifyFinishedDependency();
        }
    }
    ASSERT_TRUE(gInputHandler.isReady());
    const auto tensorMap = gInputHandler.getInputs();
    ASSERT_EQ(tensorMap.size(), 1);
    const auto& tensor = tensorMap.at(inputName);
    EXPECT_EQ(tensor->get_size(), blobsData.size());
    EXPECT_THAT(tensor->get_shape(), ElementsAre(demultiplyCounts[0], demultiplyCounts[1], 1, elementCountPerShard));
    EXPECT_EQ(std::memcmp((char*)((const void*)(tensor.data())), blobsData.data(), blobsData.size() * sizeof(float)), 0);
}

TEST_F(GatherNodeInputHandlerTest, SetInputsWithShardsHavingDifferentShapesShouldReturnErrorWhenGathering) {
    const std::string inputNames{"a"};
    std::vector<std::vector<size_t>> shapes{{1, 10}, {1, 9}};
    ov::element::Type_t precision{ov::element::Type_t::f32};
    std::vector<float> blobsData{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67};
    std::vector<std::shared_ptr<ov::runtime::Tensor>> inputBlobs{createSharedTensor(precision, shapes[0], blobsData.data(), createSharedTensor(precision, shapes[1], blobsData.data()};
    const session_id_t shardsCount = 2;  // subsessionSize/demultiplyCount
    CollapseDetails collapsingDetails{{std::string("NOT_IMPORTANT_DEMULTIPLEXER_NAME")}, {shardsCount}};
    GatherNodeInputHandler gInputHandler(inputNames.size(), collapsingDetails);
    Status status;
    for (session_id_t j = 0; j < shardsCount; ++j) {
        EXPECT_FALSE(gInputHandler.isReady());
        status = gInputHandler.setInput(inputNames, inputBlobs[j], j);
        EXPECT_EQ(status, StatusCode::OK) << status.string();
        // each input comming from different node so we call notify each time
        status = gInputHandler.notifyFinishedDependency();
        if (!status.ok()) {
            EXPECT_EQ(status, StatusCode::PIPELINE_INCONSISTENT_SHARD_DIMENSIONS) << status.string();
            break;
        }
    }
    // The second notify should fail since the shard dimension should be different
    EXPECT_EQ(status, StatusCode::PIPELINE_INCONSISTENT_SHARD_DIMENSIONS) << status.string();
}

class GatherNodeTest : public TestWithTempDir {};

static const char* configDummy1BsDummy2Bs = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        },
        {
            "config": {
                "name": "dummy2bs",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1,
                "batch_size": 2
            }
        }
    ]
})";

class DLNodeSessionWithGetInputsExposed : public DLNodeSession {
public:
    DLNodeSessionWithGetInputsExposed(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
        DLNodeSession(metadata, nodeName, inputsCount, collapsingDetails, manager, modelName, modelVersion) {}
    DLNodeSessionWithGetInputsExposed(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
        DLNodeSession(std::move(metadata), nodeName, inputsCount, collapsingDetails, manager, modelName, modelVersion) {}

    const auto& getInputs() const {
        return this->inputHandler->getInputs();
    }
};

class DLNodeWithGetInputsExposed : public DLNode {
public:
    DLNodeWithGetInputsExposed(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager,
        std::unordered_map<std::string, std::string> nodeOutputNameAlias, const std::optional<std::set<std::string>>& gatherFrom) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias, 0, gatherFrom.value_or(std::set<std::string>())) {
    }
    const auto& getInputsFromInputHandler(session_key_t sessionId) const {
        DLNodeSessionWithGetInputsExposed& dlnodesessionWithGetInputsExposed = static_cast<DLNodeSessionWithGetInputsExposed&>(*nodeSessions.at(sessionId));
        return dlnodesessionWithGetInputsExposed.getInputs();
    }
    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, const CollapseDetails& collapsingDetails) override {
        return std::make_unique<DLNodeSessionWithGetInputsExposed>(metadata, getName(), previous.size(), collapsingDetails,
            this->modelManager, this->modelName, this->modelVersion.value_or(0));
    }
};

TEST_F(GatherNodeTest, FullFlowGatherInNonExitNode) {
    // This test simulates node with multiple subsessions connected to following node
    // that should gather it results but is not exit node
    ConstructorEnabledModelManager manager;
    const std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(configDummy1BsDummy2Bs, fileToReload);
    auto status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK) << status.string();
    const std::string node1Name = "node1";
    DLNode oneDummyNode1{node1Name, "dummy", 1, manager, {}};
    const std::string demultiplexerNodeName{"nodeDummy"};
    const std::optional<std::set<std::string>> gatherFrom{{demultiplexerNodeName}};
    DLNodeWithGetInputsExposed gather2DummyNode{"nodeGather", "dummy2bs", 1, manager, {}, gatherFrom};
    Pipeline::connect(oneDummyNode1, gather2DummyNode, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});

    // prepare blobs to be gathered
    const std::vector<size_t> shape{1, 10};
    const ov::element::Type_t precision{ov::element::Type_t::f32};
    std::vector<float> nodeRawResults1{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67};
    std::vector<float> nodeRawResults2{-13, -4.4, 15, 2, 0.3, -42, 13, 0.1, 91, 21.67};
    auto originalBlob1 = createSharedTensor(precision, shape, nodeRawResults1.data());
    auto originalBlob2 = createSharedTensor(precision, shape, nodeRawResults2.data());
    // prepare session results
    TensorMap dummy1Result{{DUMMY_MODEL_OUTPUT_NAME, originalBlob1}};
    TensorMap dummy2Result{{DUMMY_MODEL_OUTPUT_NAME, originalBlob2}};
    NodeSessionMetadata meta;
    const session_id_t shardsCount = 2;
    auto subsessions = meta.generateSubsessions(demultiplexerNodeName, shardsCount);
    ASSERT_EQ(subsessions.size(), 2);
    SessionResults oneDummyNodeSessionResults1;
    SessionResults oneDummyNodeSessionResults2;
    oneDummyNodeSessionResults1.insert({subsessions[0].getSessionKey(), {subsessions[0], dummy1Result}});
    oneDummyNodeSessionResults2.insert({subsessions[1].getSessionKey(), {subsessions[1], dummy2Result}});
    // actual test steps
    ASSERT_EQ(gather2DummyNode.setInputs(oneDummyNode1, oneDummyNodeSessionResults1), StatusCode::OK);
    ASSERT_EQ(gather2DummyNode.setInputs(oneDummyNode1, oneDummyNodeSessionResults2), StatusCode::OK);
    auto readySessions = gather2DummyNode.getReadySessions();
    ASSERT_EQ(readySessions.size(), 1);
    const auto& inputs = gather2DummyNode.getInputsFromInputHandler(subsessions[0].getSessionKey({demultiplexerNodeName}));
    EXPECT_EQ(inputs.size(), 1);
    ASSERT_NE(inputs.find(DUMMY_MODEL_INPUT_NAME), inputs.end());
    const auto& gatheredBlob = inputs.at(DUMMY_MODEL_INPUT_NAME);
    EXPECT_EQ(gatheredBlob->get_size(), nodeRawResults1.size() * shardsCount);
    std::vector<float> resultBlobData(nodeRawResults1.size() * shardsCount);
    std::copy(nodeRawResults1.begin(), nodeRawResults1.end(), resultBlobData.begin());
    std::copy(nodeRawResults2.begin(), nodeRawResults2.end(), resultBlobData.begin() + nodeRawResults1.size());
    EXPECT_EQ(memcmp((char*)((const void*)gatheredBlob.data()), resultBlobData.data(), resultBlobData.size() * sizeof(float)), 0);
}
