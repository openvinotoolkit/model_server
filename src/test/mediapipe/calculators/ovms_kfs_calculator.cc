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
#include "../../../stringutils.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#include "src/kfs_frontend/kfs_grpc_inference_service.hpp"
#include "src/test/mediapipe/calculators/ovmscalculator.pb.h"

// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {

using std::endl;

class OVMSTestKFSPassCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        RET_CHECK(cc->Inputs().GetTags().size() == 1);
        RET_CHECK(cc->Outputs().GetTags().size() == 1);
        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse>();

        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        cc->SetOffset(TimestampDiff(0));
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        const KFSRequest* request = cc->Inputs().Tag("REQUEST").Get<const KFSRequest*>();
        KFSResponse response;
        for (int i = 0; i < request->inputs().size(); i++) {
            auto* output = response.add_outputs();
            output->set_datatype(request->inputs()[i].datatype());
            output->set_name("out");
            for (int j = 0; j < request->inputs()[i].shape_size(); j++) {
                output->add_shape(request->inputs()[i].shape().at(j));
            }
            *output->mutable_contents() = request->inputs()[i].contents();
        }

        for (int i = 0; i < request->raw_input_contents().size(); i++) {
            response.add_raw_output_contents()->assign(request->raw_input_contents()[i].data(), request->raw_input_contents()[i].size());
        }

        cc->Outputs().Tag("RESPONSE").AddPacket(::mediapipe::MakePacket<KFSResponse>(std::move(response)).At(cc->InputTimestamp()));

        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(OVMSTestKFSPassCalculator);
}  // namespace mediapipe
