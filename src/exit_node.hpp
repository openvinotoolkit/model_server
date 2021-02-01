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
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "node.hpp"
#include "tensorinfo.hpp"

namespace ovms {

const std::string EXIT_NODE_NAME = "response";

class ExitNode : public Node {
    tensorflow::serving::PredictResponse* response;

public:
    ExitNode(tensorflow::serving::PredictResponse* response) :
        Node(EXIT_NODE_NAME),
        response(response) {
    }

    // Exit node does not have execute logic.
    // It serializes its received input blobs to proto in ::fetchResults
    Status execute(session_key_t sessionId, PipelineEventQueue& notifyEndQueue) override;

protected:
    Status fetchResults(const BlobMap& outputs);

public:
    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;

    // Exit nodes have no dependants
    void addDependant(Node& node) override {
        throw std::logic_error("This node cannot have dependant");
    }

    Status serialize(const InferenceEngine::Blob::Ptr& blob, tensorflow::TensorProto& proto);
    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, session_id_t shardsCount) override;
};

}  // namespace ovms
