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
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../dl_node.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../gathernodeinputhandler.hpp"
#include "../modelconfig.hpp"
#include "../nodeinputhandler.hpp"
#include "../pipeline.hpp"
#include "../pipeline_factory.hpp"
#include "../pipelinedefinition.hpp"
#define DEBUG
#include <cstdio>
#include <cstdlib>

#include "../dlnodesession.hpp"
#include "../logging.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::_;
using testing::ElementsAre;
using testing::Return;

class GatherNodeInputHandlerTest : public ::testing::Test {};

TEST_F(GatherNodeInputHandlerTest, ThreePredecessorNodesWithSubsessionSize2) {
    // simulate all 3 inputs comming from different predecessor nodes
    // with session demultiplexed to 2 shards
    const std::vector<std::string> inputNames{"a", "b", "c"};
    const std::vector<size_t> shape{1, 10};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::NC};
    std::vector<float> blobData{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(desc, blobData.data());
    const uint32_t shardsCount = 2;  // subsessionSize/demultiplyCount
    GatherNodeInputHandler gInputHandler(inputNames.size(), shardsCount);
    for (session_id_t j = 0; j < shardsCount; ++j) {
        for (size_t i = 0; i < inputNames.size(); ++i) {
            EXPECT_FALSE(gInputHandler.isReady());
            gInputHandler.setInput(inputNames[i], inputBlob, j);
            // each input comming from different node so we call notify each time
            gInputHandler.notifyFinishedDependency();
        }
    }
    EXPECT_TRUE(gInputHandler.isReady());
    const auto blobMap = gInputHandler.getInputs();
    EXPECT_EQ(blobMap.size(), inputNames.size());

    std::vector<float> resultBlobData(blobData.size() * shardsCount);
    std::copy(blobData.begin(), blobData.end(), resultBlobData.begin());
    std::copy(blobData.begin(), blobData.end(), resultBlobData.begin() + blobData.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
        const auto& blob = blobMap.at(inputNames[i]);
        EXPECT_EQ(blob->size(), blobData.size() * shardsCount);
        EXPECT_THAT(blob->getTensorDesc().getDims(), ElementsAre(1, shardsCount, blobData.size()));
        EXPECT_EQ(std::memcmp((char*)((const void*)blob->cbuffer()), resultBlobData.data(), resultBlobData.size() * sizeof(float)), 0);
    }
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
    DLNodeSessionWithGetInputsExposed(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, session_id_t shardsCount, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
        DLNodeSession(metadata, nodeName, inputsCount, shardsCount, manager, modelName, modelVersion) {}
    DLNodeSessionWithGetInputsExposed(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, session_id_t shardsCount, ModelManager& manager, const std::string& modelName, model_version_t modelVersion) :
        DLNodeSession(std::move(metadata), nodeName, inputsCount, shardsCount, manager, modelName, modelVersion) {}

    const auto& getInputs() const {
        return this->inputHandler->getInputs();
    }
};

class DLNodeWithGetInputsExposed : public DLNode {
public:
    DLNodeWithGetInputsExposed(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager,
        std::unordered_map<std::string, std::string> nodeOutputNameAlias, const std::optional<std::set<std::string>>& gatherFrom) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias) {
        this->gatherFrom = gatherFrom;
    }
    const auto& getInputsFromInputHandler(session_key_t sessionId) const {
        DLNodeSessionWithGetInputsExposed& dlnodesessionWithGetInputsExposed = static_cast<DLNodeSessionWithGetInputsExposed&>(*nodeSessions.at(sessionId));
        return dlnodesessionWithGetInputsExposed.getInputs();
    }
    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, session_id_t shardsCount) override {
        return std::make_unique<DLNodeSessionWithGetInputsExposed>(metadata, getName(), previous.size(), shardsCount,
            this->modelManager, this->modelName, this->modelVersion.value_or(0));
    }
};

TEST_F(GatherNodeTest, FullFlowGatherInNonExitNode) {
    // This test simulates node with multiple subsessions connected to following node
    // that should gather it results but is not exit node
    ConstructorEnabledModelManager manager;
    const std::string fileToReload = directoryPath + "/ovms_config_file.json";
    createConfigFileWithContent(configDummy1BsDummy2Bs, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), StatusCode::OK);
    const std::string node1Name = "node1";
    DLNode oneDummyNode1{node1Name, "dummy", 1, manager, {}};
    const std::string demultiplexerNodeName{"nodeDummy"};
    const std::optional<std::set<std::string>> gatherFrom{{demultiplexerNodeName}};
    DLNodeWithGetInputsExposed gather2DummyNode{"nodeGather", "dummy2bs", 1, manager, {}, gatherFrom};
    Pipeline::connect(oneDummyNode1, gather2DummyNode, {{DUMMY_MODEL_OUTPUT_NAME, DUMMY_MODEL_INPUT_NAME}});

    // prepare blobs to be gathered
    const std::vector<size_t> shape{1, 10};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::NC};
    std::vector<float> nodeRawResults1{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67};
    std::vector<float> nodeRawResults2{-13, -4.4, 15, 2, 0.3, -42, 13, 0.1, 91, 21.67};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    InferenceEngine::Blob::Ptr originalBlob1 = InferenceEngine::make_shared_blob<float>(desc, nodeRawResults1.data());
    InferenceEngine::Blob::Ptr originalBlob2 = InferenceEngine::make_shared_blob<float>(desc, nodeRawResults2.data());
    // prepare session results
    BlobMap dummy1Result{{DUMMY_MODEL_OUTPUT_NAME, originalBlob1}};
    BlobMap dummy2Result{{DUMMY_MODEL_OUTPUT_NAME, originalBlob2}};
    NodeSessionMetadata meta;
    const session_id_t shardsCount = 2;
    auto subsessions = meta.generateSubsessions(demultiplexerNodeName, shardsCount);
    ASSERT_EQ(subsessions.size(), 2);
    SessionResults oneDummyNodeSessionResults1;
    SessionResults oneDummyNodeSessionResults2;
    oneDummyNodeSessionResults1.insert({subsessions[0].getSessionKey(), {subsessions[0], dummy1Result}});
    oneDummyNodeSessionResults2.insert({subsessions[1].getSessionKey(), {subsessions[1], dummy2Result}});
    // actual test steps
    gather2DummyNode.setInputs(oneDummyNode1, oneDummyNodeSessionResults1);
    gather2DummyNode.setInputs(oneDummyNode1, oneDummyNodeSessionResults2);
    auto readySessions = gather2DummyNode.getReadySessions();
    ASSERT_EQ(readySessions.size(), 1);
    const auto& inputs = gather2DummyNode.getInputsFromInputHandler(subsessions[0].getSessionKey({demultiplexerNodeName}));
    EXPECT_EQ(inputs.size(), 1);
    ASSERT_NE(inputs.find(DUMMY_MODEL_INPUT_NAME), inputs.end());
    const auto& gatheredBlob = inputs.at(DUMMY_MODEL_INPUT_NAME);
    EXPECT_EQ(gatheredBlob->size(), nodeRawResults1.size() * shardsCount);
    std::vector<float> resultBlobData(nodeRawResults1.size() * shardsCount);
    std::copy(nodeRawResults1.begin(), nodeRawResults1.end(), resultBlobData.begin());
    std::copy(nodeRawResults2.begin(), nodeRawResults2.end(), resultBlobData.begin() + nodeRawResults1.size());
    EXPECT_EQ(memcmp((char*)((const void*)gatheredBlob->cbuffer()), resultBlobData.data(), resultBlobData.size() * sizeof(float)), 0);
}
