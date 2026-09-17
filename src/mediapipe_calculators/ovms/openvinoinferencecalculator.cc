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
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <adapters/inference_adapter.h>  // model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "src/mediapipe_calculators/ovms/openvinoinferencecalculator.h"
#include "src/mediapipe_calculators/ovms/openvinoinferencecalculatoroptions.h"
#include "src/mediapipe_calculators/ovms/openvinoinferenceutils.h"
#include "src/mediapipe_calculators/ovms/openvinoinferencecalculator.pb.h"
#if (OVMS_DUMP_TO_FILE == 1)
#include "src/mediapipe_calculators/ovms/openvinoinferencedumputils.h"
#endif
#pragma GCC diagnostic pop
namespace mediapipe {
using std::endl;

static Tensor::ElementType OVType2MPType(ov::element::Type_t precision) {
    static std::unordered_map<ov::element::Type_t, Tensor::ElementType> precisionMap{
        {ov::element::Type_t::f32, Tensor::ElementType::kFloat32},
        {ov::element::Type_t::f16, Tensor::ElementType::kFloat16},
        {ov::element::Type_t::i32, Tensor::ElementType::kInt32},
        {ov::element::Type_t::i8, Tensor::ElementType::kInt8},
        {ov::element::Type_t::u8, Tensor::ElementType::kUInt8},
        {ov::element::Type_t::boolean, Tensor::ElementType::kBool},
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return Tensor::ElementType::kNone;
    }
    return it->second;
}

static ov::element::Type_t MPType2OVType(Tensor::ElementType precision) {
    static std::unordered_map<Tensor::ElementType, ov::element::Type_t> precisionMap{
        {Tensor::ElementType::kFloat32, ov::element::Type_t::f32},
        {Tensor::ElementType::kFloat16, ov::element::Type_t::f16},
        {Tensor::ElementType::kInt32, ov::element::Type_t::i32},
        {Tensor::ElementType::kInt8, ov::element::Type_t::i8},
        {Tensor::ElementType::kUInt8, ov::element::Type_t::u8},
        {Tensor::ElementType::kBool, ov::element::Type_t::boolean},
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return ov::element::Type_t::dynamic;
    }
    return it->second;
}

static ov::Tensor convertMPTensor2OVTensor(const Tensor& inputTensor) {
    void* data;
    switch (inputTensor.element_type()) {
    case Tensor::ElementType::kFloat32:
    case Tensor::ElementType::kFloat16:
        data = reinterpret_cast<void*>(const_cast<float*>(inputTensor.GetCpuReadView().buffer<float>()));
        break;
    case Tensor::ElementType::kUInt8:
        data = reinterpret_cast<void*>(const_cast<uint8_t*>(inputTensor.GetCpuReadView().buffer<uint8_t>()));
        break;
    case Tensor::ElementType::kInt8:
        data = reinterpret_cast<void*>(const_cast<int8_t*>(inputTensor.GetCpuReadView().buffer<int8_t>()));
        break;
    case Tensor::ElementType::kInt32:
        data = reinterpret_cast<void*>(const_cast<int32_t*>(inputTensor.GetCpuReadView().buffer<int32_t>()));
        break;
    case Tensor::ElementType::kBool:
        data = reinterpret_cast<void*>(const_cast<bool*>(inputTensor.GetCpuReadView().buffer<bool>()));
        break;
    default:
        data = reinterpret_cast<void*>(const_cast<void*>(inputTensor.GetCpuReadView().buffer<void>()));
        break;
    }
    auto datatype = MPType2OVType(inputTensor.element_type());
    if (datatype == ov::element::Type_t::dynamic) {
        LOG(INFO) << "Not supported precision for Mediapipe tensor deserialization";
        throw std::runtime_error("Not supported precision for Mediapipe tensor deserialization");
    }
    ov::Shape shape;
    for (const auto& dim : inputTensor.shape().dims) {
        shape.emplace_back(dim);
    }
    ov::Tensor result(datatype, shape, data);
    return result;
}

static Tensor convertOVTensor2MPTensor(const ov::Tensor& inputTensor) {
    std::vector<int> rawShape;
    for (size_t i = 0; i < inputTensor.get_shape().size(); i++) {
        rawShape.emplace_back(inputTensor.get_shape()[i]);
    }
    Tensor::Shape shape{rawShape};
    auto datatype = OVType2MPType(inputTensor.get_element_type());
    if (datatype == mediapipe::Tensor::ElementType::kNone) {
        LOG(INFO) << "Not supported precision for Mediapipe tensor serialization: " << inputTensor.get_element_type();
        throw std::runtime_error("Not supported precision for Mediapipe tensor serialization");
    }
    Tensor outputTensor(datatype, shape);
    void* data;
    switch (inputTensor.get_element_type()) {
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f16:
        data = reinterpret_cast<void*>(const_cast<float*>(outputTensor.GetCpuWriteView().buffer<float>()));
        break;
    case ov::element::Type_t::u8:
        data = reinterpret_cast<void*>(const_cast<uint8_t*>(outputTensor.GetCpuWriteView().buffer<uint8_t>()));
        break;
    case ov::element::Type_t::i8:
        data = reinterpret_cast<void*>(const_cast<int8_t*>(outputTensor.GetCpuWriteView().buffer<int8_t>()));
        break;
    case ov::element::Type_t::i32:
        data = reinterpret_cast<void*>(const_cast<int32_t*>(outputTensor.GetCpuWriteView().buffer<int32_t>()));
        break;
    case ov::element::Type_t::boolean:
        data = reinterpret_cast<void*>(const_cast<bool*>(outputTensor.GetCpuWriteView().buffer<bool>()));
        break;
    default:
        data = reinterpret_cast<void*>(const_cast<void*>(outputTensor.GetCpuWriteView().buffer<void>()));
        break;
    }
    std::memcpy(data, inputTensor.data(), inputTensor.get_byte_size());
    return outputTensor;
}

absl::Status OpenVINOInferenceCalculator::GetContract(CalculatorContract* cc) {
    LOG(INFO) << "OpenVINOInferenceCalculator GetContract start";
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());
    RET_CHECK(cc->InputSidePackets().HasTag(SESSION_TAG));
    RET_CHECK(ValidateCalculatorSettings(cc));

    for (const std::string& tag : cc->Inputs().GetTags()) {
        if (startsWith(tag, OVTENSORS_TAG)) {
            LOG(INFO) << "setting input tag:" << tag << " to OVTensors";
            cc->Inputs().Tag(tag).Set<std::vector<ov::Tensor>>();
        } else if (startsWith(tag, OVTENSOR_TAG)) {
            LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        } else if (startsWith(tag, MPTENSORS_TAG)) {
            LOG(INFO) << "setting input tag:" << tag << " to MPTensors";
            cc->Inputs().Tag(tag).Set<std::vector<Tensor>>();
        } else if (startsWith(tag, MPTENSOR_TAG)) {
            LOG(INFO) << "setting input tag:" << tag << " to MPTensor";
            cc->Inputs().Tag(tag).Set<Tensor>();
        } else {
            LOG(INFO) << "setting input tag:" << tag << " to OVTensor";
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        }
    }
    for (const std::string& tag : cc->Outputs().GetTags()) {
        if (startsWith(tag, OVTENSORS_TAG)) {
            LOG(INFO) << "setting output tag:" << tag << " to std::vector<ov::Tensor>";
            cc->Outputs().Tag(tag).Set<std::vector<ov::Tensor>>();
        } else if (startsWith(tag, OVTENSOR_TAG)) {
            LOG(INFO) << "setting output tag:" << tag << " to OVTensor";
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        } else if (startsWith(tag, MPTENSORS_TAG)) {
            LOG(INFO) << "setting output tag:" << tag << " to std::vector<Tensor>";
            cc->Outputs().Tag(tag).Set<std::vector<Tensor>>();
        } else if (startsWith(tag, MPTENSOR_TAG)) {
            LOG(INFO) << "setting output tag:" << tag << " to MPTensor";
            cc->Outputs().Tag(tag).Set<Tensor>();
        } else {
            LOG(INFO) << "setting output tag:" << tag << " to OVTensor";
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        }
    }
    cc->InputSidePackets().Tag(SESSION_TAG.c_str()).Set<std::shared_ptr<::InferenceAdapter>>();
    LOG(INFO) << "OpenVINOInferenceCalculator GetContract end";
    return absl::OkStatus();
}

absl::Status OpenVINOInferenceCalculator::Close(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOInferenceCalculator Close";
    return absl::OkStatus();
}

absl::Status OpenVINOInferenceCalculator::Open(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOInferenceCalculator Open start";
    session = cc->InputSidePackets()
                  .Tag(SESSION_TAG.c_str())
                  .Get<std::shared_ptr<::InferenceAdapter>>();
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
        if (!cc->Inputs().Get(id).Header().IsEmpty()) {
            cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
        }
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
        for (CollectionItemId id = cc->InputSidePackets().BeginId();
             id < cc->InputSidePackets().EndId(); ++id) {
            cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
        }
    }
    const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();
    for (const auto& [key, value] : options.tag_to_output_tensor_names()) {
        outputNameToTag[value] = key;
    }

    auto& input_list = options.input_order_list();
    input_order_list.clear();
    for (int i = 0; i < input_list.size(); i++) {
        input_order_list.push_back(input_list[i]);
    }
    auto& output_list = options.output_order_list();
    output_order_list.clear();
    for (int i = 0; i < output_list.size(); i++) {
        output_order_list.push_back(output_list[i]);
    }

    cc->SetOffset(TimestampDiff(0));
    LOG(INFO) << "OpenVINOInferenceCalculator Open end";
    return absl::OkStatus();
}

absl::Status OpenVINOInferenceCalculator::Process(CalculatorContext* cc) {
    LOG(INFO) << "OpenVINOInferenceCalculator process start";
    if (cc->Inputs().NumEntries() == 0) {
        return tool::StatusStop();
    }

    const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();
    const auto& inputTagInputMap = options.tag_to_input_tensor_names();
    ::InferenceInput input;
    ::InferenceOutput output;
    for (const std::string& tag : cc->Inputs().GetTags()) {
        if (cc->Inputs().Tag(tag).IsEmpty()) {
            LOG(INFO) << "OpenVINOInferenceCalculator expects all input packets at the same time for each call to Process().";
            RET_CHECK(false);
        }
        const char* realInputName{nullptr};
        auto it = inputTagInputMap.find(tag);
        if (it == inputTagInputMap.end()) {
            realInputName = tag.c_str();
        } else {
            realInputName = it->second.c_str();
        }
#define DESERIALIZE_TENSORS(TYPE, DESERIALIZE_FUN)                                  \
    auto& packet = cc->Inputs().Tag(tag).Get<std::vector<TYPE>>();                  \
    if (packet.size() != this->input_order_list.size()) {                           \
        LOG(INFO) << "input_order_list size does not match the input vector size."; \
        RET_CHECK(false);                                                           \
    }                                                                               \
    for (size_t i = 0; i < this->input_order_list.size(); i++) {                    \
        auto& tensor = packet[i];                                                   \
        input[this->input_order_list[i]] = DESERIALIZE_FUN(tensor);                 \
    }

        try {
            if (startsWith(tag, OVTENSORS_TAG)) {
                DESERIALIZE_TENSORS(ov::Tensor, );  // NOLINT
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                DESERIALIZE_TENSORS(Tensor, convertMPTensor2OVTensor);
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                auto& packet = cc->Inputs().Tag(tag).Get<Tensor>();
                input[realInputName] = convertMPTensor2OVTensor(packet);
            } else {
                auto& packet = cc->Inputs().Tag(tag).Get<ov::Tensor>();
                input[realInputName] = packet;
            }
        } catch (const std::runtime_error& e) {
            LOG(INFO) << "Failed to deserialize tensor error:" << e.what();
            RET_CHECK(false);
        }
    }
#if (OVMS_DUMP_TO_FILE == 1)
    dumpOvTensorInput(input, "input");
#endif
    try {
        output = session->infer(input);
#if (OVMS_DUMP_TO_FILE == 1)
        dumpOvTensorInput(output, "output");
#endif
    } catch (const std::exception& e) {
        LOG(INFO) << "Caught exception from session infer():" << e.what();
        RET_CHECK(false);
    } catch (...) {
        LOG(INFO) << "Caught unknown exception from session infer()";
        RET_CHECK(false);
    }
    auto outputsCount = output.size();
    RET_CHECK(outputsCount >= cc->Outputs().GetTags().size());
    LOG(INFO) << "output tags size: " << cc->Outputs().GetTags().size();
    for (const auto& tag : cc->Outputs().GetTags()) {
        LOG(INFO) << "Processing tag: " << tag;
        std::string tensorName;
        auto tensorIt = output.begin();

        if (!IsVectorTag(tag)) {
            auto it = options.tag_to_output_tensor_names().find(tag);
            if (it == options.tag_to_output_tensor_names().end()) {
                tensorName = tag;
            } else {
                tensorName = it->second;
            }
            tensorIt = output.find(tensorName);
            if (tensorIt == output.end()) {
                LOG(INFO) << "Could not find: " << tensorName << " in inference output";
                RET_CHECK(false);
            }
        }

        try {
#define SERIALIZE_TENSORS(TYPE, SESERIALIZE_FUN)                                                 \
    auto tensors = std::make_unique<std::vector<TYPE>>();                                        \
    if (output.size() > 1 && this->output_order_list.size() != this->output_order_list.size()) { \
        LOG(INFO) << "output_order_list not set properly in options for multiple outputs.";      \
        RET_CHECK(false);                                                                        \
    }                                                                                            \
    if (this->output_order_list.size() > 0) {                                                    \
        for (const auto& tensorName : this->output_order_list) {                                 \
            tensorIt = output.find(tensorName);                                                  \
            if (tensorIt == output.end()) {                                                      \
                LOG(INFO) << "Could not find: " << tensorName << " in inference output";         \
                RET_CHECK(false);                                                                \
            }                                                                                    \
            tensors->emplace_back(SESERIALIZE_FUN(tensorIt->second));                            \
        }                                                                                        \
    } else {                                                                                     \
        for (auto& [name, tensor] : output) {                                                    \
            tensors->emplace_back(SESERIALIZE_FUN(tensor));                                      \
        }                                                                                        \
    }                                                                                            \
    cc->Outputs().Tag(tag).Add(                                                                  \
        tensors.release(),                                                                       \
        cc->InputTimestamp());

            if (startsWith(tag, OVTENSORS_TAG)) {
                LOG(INFO) << "OVMS calculator will process vector<ov::Tensor>";
                SERIALIZE_TENSORS(ov::Tensor, )
            } else if (startsWith(tag, MPTENSORS_TAG)) {
                LOG(INFO) << "OVMS calculator will process vector<Tensor>";
                SERIALIZE_TENSORS(Tensor, convertOVTensor2MPTensor)
            } else if (startsWith(tag, OVTENSOR_TAG)) {
                LOG(INFO) << "OVMS calculator will process ov::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            } else if (startsWith(tag, MPTENSOR_TAG)) {
                LOG(INFO) << "OVMS calculator will process mediapipe::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new Tensor(convertOVTensor2MPTensor(tensorIt->second)),
                    cc->InputTimestamp());
            } else {
                LOG(INFO) << "OVMS calculator will process ov::Tensor";
                cc->Outputs().Tag(tag).Add(
                    new ov::Tensor(tensorIt->second),
                    cc->InputTimestamp());
            }
        } catch (const std::runtime_error& e) {
            LOG(INFO) << "Failed to deserialize tensor error:" << e.what();
            RET_CHECK(false);
        }
    }
    LOG(INFO) << "OpenVINOInferenceCalculator process end";
    return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenVINOInferenceCalculator);
}  // namespace mediapipe
