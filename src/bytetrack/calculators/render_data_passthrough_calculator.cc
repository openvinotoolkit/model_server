//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "mediapipe/framework/calculator_framework.h"

#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

class PassThroughRenderDataCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag("RENDER_DATA").Set<RenderData>();
        cc->Outputs().Tag("RENDER_DATA").Set<RenderData>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        const auto& render_data =
            cc->Inputs().Tag("RENDER_DATA").Get<RenderData>();

        LOG(INFO) << "RenderData: num objects = "
                  << render_data.render_annotations_size();

        // Forward unchanged
        cc->Outputs().Tag("RENDER_DATA").AddPacket(MakePacket<RenderData>(render_data).At(cc->InputTimestamp()));

        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PassThroughRenderDataCalculator);

}  // namespace mediapipe