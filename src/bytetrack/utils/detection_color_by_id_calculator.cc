//*****************************************************************************
// Copyright (c) 2026 Intel Corporation
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
#include "src/bytetrack/utils/detection_color_by_id_calculator.pb.h"

#include <cmath>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/util/color.pb.h"

namespace mediapipe {

class DetectionColorByIdCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
        cc->Outputs().Tag("RENDER_DATA").Set<RenderData>();
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) override {
        const auto& options =
            cc->Options<mediapipe::DetectionColorByIdCalculatorOptions>();
        thickness_ = options.has_thickness() ? options.thickness() : 4.0f;
        saturation_ = options.has_saturation() ? options.saturation() : 0.85f;
        value_ = options.has_value() ? options.value() : 0.95f;
        return absl::OkStatus();
    }
    absl::Status Process(CalculatorContext* cc) override {
        const auto& detections =
            cc->Inputs().Tag("DETECTIONS").Get<std::vector<Detection>>();

        auto render_data = std::make_unique<RenderData>();

        for (const auto& det : detections) {
            int id = det.detection_id();
            mediapipe::Color color = IdToColor(id);

            const auto& bbox = det.location_data().relative_bounding_box();

            // ── 1. Bounding box ──────────────────────────────────────────────────
            {
                auto* a = render_data->add_render_annotations();
                *a->mutable_color() = color;
                a->set_thickness(thickness_);

                auto* rect = a->mutable_rectangle();
                rect->set_left(bbox.xmin());
                rect->set_top(bbox.ymin());
                rect->set_right(bbox.xmin() + bbox.width());
                rect->set_bottom(bbox.ymin() + bbox.height());
                rect->set_normalized(true);
            }

            // ── 2. Label   ───────────────────────────────────────────────────────
            {
                auto* a = render_data->add_render_annotations();
                *a->mutable_color() = color;  // same color as box
                a->set_thickness(thickness_ - 1.0f);

                auto* text = a->mutable_text();
                std::string label = "ID:" + std::to_string(id);
                if (!det.label().empty())
                    label += " " + det.label(0);
                if (det.score_size() > 0) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), " %.2f", det.score(0));
                    label += buf;
                }

                text->set_display_text(label);
                text->set_normalized(true);
                text->set_left(bbox.xmin() + 0.005f);
                text->set_baseline(bbox.ymin() + 0.04f);
                text->set_font_height(0.035f);
            }
        }

        cc->Outputs().Tag("RENDER_DATA").Add(render_data.release(), cc->InputTimestamp());
        return absl::OkStatus();
    }

private:
    mediapipe::Color IdToColor(int id) {
        // Golden angle ensures max visual distance between consecutive IDs
        const float kGoldenAngle = 137.508f;
        float hue = std::fmod(id * kGoldenAngle, 360.0f);
        float chroma = value_ * saturation_;
        float x = chroma * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
        float m = value_ - chroma;

        float r = 0, g = 0, b = 0;
        if (hue < 60) {
            r = chroma;
            g = x;
        } else if (hue < 120) {
            r = x;
            g = chroma;
        } else if (hue < 180) {
            g = chroma;
            b = x;
        } else if (hue < 240) {
            g = x;
            b = chroma;
        } else if (hue < 300) {
            r = x;
            b = chroma;
        } else {
            r = chroma;
            b = x;
        }

        mediapipe::Color color;
        color.set_r(static_cast<int>((r + m) * 255));
        color.set_g(static_cast<int>((g + m) * 255));
        color.set_b(static_cast<int>((b + m) * 255));
        return color;
    }
    float thickness_ = 4.0f;
    float saturation_ = 0.85f;
    float value_ = 0.95f;
};

REGISTER_CALCULATOR(DetectionColorByIdCalculator);

}  // namespace mediapipe
