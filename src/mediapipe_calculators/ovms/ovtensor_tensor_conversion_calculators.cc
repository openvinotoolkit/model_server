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

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

ov::element::Type ToOVElementType(Tensor::ElementType type) {
    switch (type) {
    case Tensor::ElementType::kFloat32:
        return ov::element::f32;
    case Tensor::ElementType::kFloat16:
        return ov::element::f16;
    case Tensor::ElementType::kInt32:
        return ov::element::i32;
    case Tensor::ElementType::kInt8:
        return ov::element::i8;
    case Tensor::ElementType::kUInt8:
        return ov::element::u8;
    case Tensor::ElementType::kBool:
        return ov::element::boolean;
    default:
        return ov::element::dynamic;
    }
}

Tensor::ElementType ToMPTensorElementType(const ov::element::Type& type) {
    if (type == ov::element::f32) return Tensor::ElementType::kFloat32;
    if (type == ov::element::f16) return Tensor::ElementType::kFloat16;
    if (type == ov::element::i32) return Tensor::ElementType::kInt32;
    if (type == ov::element::i8) return Tensor::ElementType::kInt8;
    if (type == ov::element::u8) return Tensor::ElementType::kUInt8;
    if (type == ov::element::boolean) return Tensor::ElementType::kBool;
    return Tensor::ElementType::kNone;
}

ov::Tensor ConvertMPTensor(const Tensor& input) {
    const ov::element::Type elementType = ToOVElementType(input.element_type());
    if (elementType == ov::element::dynamic) {
        throw std::invalid_argument("Unsupported MediaPipe tensor element type");
    }
    ov::Shape shape;
    for (const int dimension : input.shape().dims) {
        if (dimension < 0) {
            throw std::invalid_argument("MediaPipe tensor dimensions must be nonnegative");
        }
        shape.push_back(static_cast<size_t>(dimension));
    }
    ov::Tensor output(elementType, shape);
    const auto readView = input.GetCpuReadView();
    std::memcpy(output.data(), readView.buffer<void>(), output.get_byte_size());
    return output;
}

class MPTensorToOVTensorCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag("TENSOR").Set<Tensor>();
        cc->Outputs().Tag("OVTENSOR").Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (cc->Inputs().Tag("TENSOR").IsEmpty()) return absl::OkStatus();
        const Tensor& input = cc->Inputs().Tag("TENSOR").Get<Tensor>();
        cc->Outputs().Tag("OVTENSOR").AddPacket(
            MakePacket<ov::Tensor>(ConvertMPTensor(input)).At(cc->InputTimestamp()));
        return absl::OkStatus();
    }
};

class OVTensorToMPTensorCalculator : public CalculatorBase {
 public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Tag("OVTENSOR").Set<ov::Tensor>();
        cc->Outputs().Tag("TENSOR").Set<Tensor>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        if (cc->Inputs().Tag("OVTENSOR").IsEmpty()) return absl::OkStatus();
        const ov::Tensor& input = cc->Inputs().Tag("OVTENSOR").Get<ov::Tensor>();
        const Tensor::ElementType elementType = ToMPTensorElementType(input.get_element_type());
        RET_CHECK(elementType != Tensor::ElementType::kNone)
            << "Unsupported OpenVINO tensor element type: " << input.get_element_type();
        std::vector<int> dimensions;
        dimensions.reserve(input.get_shape().size());
        for (const size_t dimension : input.get_shape()) {
            RET_CHECK_LE(dimension, static_cast<size_t>(std::numeric_limits<int>::max()));
            dimensions.push_back(static_cast<int>(dimension));
        }
        Tensor output(elementType, Tensor::Shape{dimensions});
        auto writeView = output.GetCpuWriteView();
        std::memcpy(writeView.buffer<void>(), input.data(), input.get_byte_size());
        cc->Outputs().Tag("TENSOR").AddPacket(
            MakePacket<Tensor>(std::move(output)).At(cc->InputTimestamp()));
        return absl::OkStatus();
    }
};

}  // namespace

REGISTER_CALCULATOR(MPTensorToOVTensorCalculator);
REGISTER_CALCULATOR(OVTensorToMPTensorCalculator);

}  // namespace mediapipe