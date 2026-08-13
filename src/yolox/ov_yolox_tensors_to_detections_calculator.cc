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

#include <cstdint>
#include <vector>

#include <openvino/openvino.hpp>

#include "src/yolox/ov_yolox_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class OVYoloXTensorsToDetectionsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

    float confidenceThreshold_ = 0.1f;
    float inputSize_ = 416.0f;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs()
            .Tag(INPUT_TAG_NAME)
            .Set<std::vector<ov::Tensor>>();

        cc->Outputs()
            .Tag(OUTPUT_TAG_NAME)
            .Set<std::vector<Detection>>();

        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        const auto& options =
            cc->Options<
                mediapipe::OVYoloXTensorsToDetectionsCalculatorOptions>();

        confidenceThreshold_ =
            options.has_conf_thresh() ? options.conf_thresh() : 0.1f;

        inputSize_ =
            options.has_input_size() ? options.input_size() : 416.0f;

        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }

        const auto& tensors =
            cc->Inputs()
                .Tag(INPUT_TAG_NAME)
                .Get<std::vector<ov::Tensor>>();

        RET_CHECK_EQ(tensors.size(), 2u);

        const auto& boxesTensor = tensors[0];
        const auto& labelsTensor = tensors[1];

        RET_CHECK_EQ(boxesTensor.get_element_type(), ov::element::f32);
        RET_CHECK_EQ(labelsTensor.get_element_type(), ov::element::i64);

        const auto boxesShape = boxesTensor.get_shape();
        const auto labelsShape = labelsTensor.get_shape();

        RET_CHECK_EQ(boxesShape.size(), 3u);
        RET_CHECK_EQ(boxesShape[0], 1u);
        RET_CHECK_EQ(boxesShape[2], 5u);

        RET_CHECK_EQ(labelsShape.size(), 2u);
        RET_CHECK_EQ(labelsShape[0], 1u);
        RET_CHECK_EQ(labelsShape[1], boxesShape[1]);

        const size_t numBoxes = boxesShape[1];

        const float* boxes = boxesTensor.data<const float>();
        const int64_t* labels = labelsTensor.data<const int64_t>();

        RET_CHECK(boxes != nullptr);
        RET_CHECK(labels != nullptr);

        auto detections =
            absl::make_unique<std::vector<Detection>>();

        for (size_t i = 0; i < numBoxes; ++i) {
            const float x1 = boxes[i * 5 + 0];
            const float y1 = boxes[i * 5 + 1];
            const float x2 = boxes[i * 5 + 2];
            const float y2 = boxes[i * 5 + 3];
            const float confidence = boxes[i * 5 + 4];

            if (confidence < confidenceThreshold_) {
                continue;
            }

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            Detection detection;

            auto* locationData =
                detection.mutable_location_data();

            locationData->set_format(
                LocationData::RELATIVE_BOUNDING_BOX);

            auto* boundingBox =
                locationData->mutable_relative_bounding_box();

            boundingBox->set_xmin(x1 / inputSize_);
            boundingBox->set_ymin(y1 / inputSize_);
            boundingBox->set_width((x2 - x1) / inputSize_);
            boundingBox->set_height((y2 - y1) / inputSize_);

            detection.add_score(confidence);
            detection.add_label_id(
                static_cast<int>(labels[i]));

            detections->emplace_back(std::move(detection));
        }

        cc->Outputs()
            .Tag(OUTPUT_TAG_NAME)
            .Add(detections.release(), cc->InputTimestamp());

        return absl::OkStatus();
    }
};

const std::string OVYoloXTensorsToDetectionsCalculator::INPUT_TAG_NAME{
    "TENSORS"};

const std::string OVYoloXTensorsToDetectionsCalculator::OUTPUT_TAG_NAME{
    "DETECTIONS"};

REGISTER_CALCULATOR(OVYoloXTensorsToDetectionsCalculator);

}  // namespace mediapipe