//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <memory>

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>

#include "llmnoderesources.hpp"
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;

constexpr size_t BATCH_SIZE = 1;

namespace mediapipe {

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

class LLMCalculator : public CalculatorBase {
    ov::Core core;
    ov::InferRequest tokenizer, detokenizer, llm;
    std::shared_ptr<ContinuousBatchingPipeline> cbPipe = nullptr;
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    // The calculator manages timestamp for outputs to work independently of inputs
    // this way we can support timestamp continuity for more than one request in streaming scenario.
    mediapipe::Timestamp outputTimestamp;
    bool hasLoopback{false};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse>();

        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        if (it == nodeResourcesMap.end()) {
            LOG(INFO) << "Could not find initialized LLM node named: " << cc->NodeName();
            RET_CHECK(false);
        }

        nodeResources = it->second;
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe
