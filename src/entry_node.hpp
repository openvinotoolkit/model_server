//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#pragma once
#include <memory>
#include <optional>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "logging.hpp"
#include "node.hpp"
#include "tensorinfo.hpp"

namespace ovms {

const std::string ENTRY_NODE_NAME = "request";

class EntryNode : public Node {
    const tensorflow::serving::PredictRequest* request;
    const tensor_map_t inputsInfo;

public:
    EntryNode(const tensorflow::serving::PredictRequest* request,
        const tensor_map_t& inputsInfo,
        std::optional<uint32_t> demultiplyCount = std::nullopt) :
        Node(ENTRY_NODE_NAME, demultiplyCount),
        request(request),
        inputsInfo(inputsInfo) {}

    Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;

protected:
    Status fetchResults(BlobMap& outputs);
    Status createShardedBlob(InferenceEngine::Blob::Ptr& dividedBlob, const InferenceEngine::TensorDesc& dividedBlobDesc, InferenceEngine::Blob::Ptr blob, size_t i, size_t step, const NodeSessionMetadata& metadata, const std::string blobName) override;

public:
    // Entry nodes have no dependency
    void addDependency(Node&, const Aliases&) override {
        throw std::logic_error("This node cannot have dependency");
    }

    Status isInputBinary(const std::string& name, bool& isBinary) const;
};

}  // namespace ovms
