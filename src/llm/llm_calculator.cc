//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;

namespace mediapipe {

class LLMCalculator : public CalculatorBase {
    mediapipe::Timestamp outputTimestamp;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse>();

        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

#define RETURN_EXECUTION_FAILED_STATUS() \
    return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution")

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
       try {
            const KFSRequest *request = cc->Inputs().Tag("REQUEST").Get<const KFSRequest*>();
            // Hardcoded single input for data
            auto data = request->raw_input_contents().Get(0);
            std::string prompt = std::string(data.begin(), data.end());
            LOG(INFO) << "Received prompt: " << prompt;

            /*
            TODO: 
            
            Real work here...
            
            */
            
            std::string outputStr = "Hardcoded output";

            auto response = std::make_unique<KFSResponse>();
            auto* responseOutput = response->add_outputs();
            responseOutput->set_name("output");
            responseOutput->set_datatype("BYTES");
            responseOutput->clear_shape();
            responseOutput->add_shape(outputStr.size());
            response->add_raw_output_contents()->assign(reinterpret_cast<char*>(outputStr.data()), outputStr.size());

            cc->Outputs().Tag("RESPONSE").AddPacket(MakePacket<KFSResponse>(*response).At(cc->InputTimestamp()));


        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            RETURN_EXECUTION_FAILED_STATUS();
        }
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe
