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