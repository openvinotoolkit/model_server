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
#include "test_utils.hpp"

using namespace ovms;

class DemultiplexerDLNodeTest : public DLNode {
public:
    DemultiplexerDLNodeTest(const std::string& nodeName, const std::string& modelName, std::optional<model_version_t> modelVersion,
        ModelManager& modelManager,
        std::unordered_map<std::string, std::string> nodeOutputNameAlias, std::optional<uint16_t> demultiplyCount) :
        DLNode(nodeName, modelName, modelVersion, modelManager, nodeOutputNameAlias) {
        this->demultiplexCount = demultiplexCount;
    }
};

TEST(DemultiplexerNode, CheckDemultipliedBlobs) {
    const std::vector<std::string> inputNames{"a", "b", "c"};
    const std::vector<size_t> shape{3, 4};
    const InferenceEngine::Precision precision{InferenceEngine::Precision::FP32};
    const InferenceEngine::Layout layout{InferenceEngine::Layout::NC};
    std::vector<float> blobData{-1, 4, 5, 12, 3, 52, 12, 0.5, 9, 1.67, 0, 8};
    const InferenceEngine::TensorDesc desc{precision, shape, layout};
    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(desc, blobData.data());
    const size_t demultiplyCount = 3;
    //DemultiplexerDLNodeTest("node", "model", std::nullopt);
}