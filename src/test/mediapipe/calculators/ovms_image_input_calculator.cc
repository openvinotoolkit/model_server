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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/kfs_frontend/kfs_grpc_inference_service.hpp"
#include "src/test/mediapipe/calculators/ovmscalculator.pb.h"
#pragma GCC diagnostic pop

namespace mediapipe {

using std::endl;

class OVMSTestImageInputPassthroughCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        RET_CHECK(cc->Inputs().GetTags().size() == 1);
        cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
        cc->Outputs().Tag("IMAGE").Set<ImageFrame>();

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
        LOG(INFO) << "Process start";
        const auto& received = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
        ImageFrame image = ImageFrame(received.Format(), received.Width(), received.Height());
        image.CopyFrom(received, 1);
        cc->Outputs().Tag("IMAGE").AddPacket(MakePacket<ImageFrame>(std::move(image)).At(Timestamp(0)));

        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(OVMSTestImageInputPassthroughCalculator);
}  // namespace mediapipe
