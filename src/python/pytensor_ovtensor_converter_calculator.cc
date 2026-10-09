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
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#pragma warning(push)
#pragma warning(disable : 4005 4018 4309 4018 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../precision.hpp"
#include "py_object_handle.hpp"
#include "python_calculators_plugin_api.hpp"
#include "python_calculators_plugin_loader.hpp"
#include "src/python/pytensor_ovtensor_converter_calculator.pb.h"

using namespace ovms;

namespace mediapipe {

const std::string& toKfsString(Precision precision) {
    static std::unordered_map<Precision, std::string> precisionMap{
        {Precision::BF16, "BF16"},
        {Precision::FP64, "FP64"},
        {Precision::FP32, "FP32"},
        {Precision::FP16, "FP16"},
        {Precision::I64, "INT64"},
        {Precision::I32, "INT32"},
        {Precision::I16, "INT16"},
        {Precision::I8, "INT8"},
        {Precision::U64, "UINT64"},
        {Precision::U32, "UINT32"},
        {Precision::U16, "UINT16"},
        {Precision::U8, "UINT8"},
        {Precision::BOOL, "BOOL"},
        // {Precision::STRING, "???"},
        {Precision::UNDEFINED, "UNDEFINED"}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        static const std::string UNDEFINED{"UNDEFINED"};
        return UNDEFINED;
    }
    return it->second;
}

Precision fromKfsString(const std::string& s) {
    static std::unordered_map<std::string, Precision> precisionMap{
        {"BF16", Precision::BF16},
        {"FP64", Precision::FP64},
        {"FP32", Precision::FP32},
        {"FP16", Precision::FP16},
        {"INT64", Precision::I64},
        {"INT32", Precision::I32},
        {"INT16", Precision::I16},
        {"INT8", Precision::I8},
        {"UINT64", Precision::U64},
        {"UINT32", Precision::U32},
        {"UINT16", Precision::U16},
        {"UINT8", Precision::U8},
        {"BOOL", Precision::BOOL},
        // {"???", Precision::STRING},
        {"UNDEFINED", Precision::UNDEFINED}};
    auto it = precisionMap.find(s);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

class PyTensorOvTensorConverterCalculator : public CalculatorBase {
    static const std::string OV_TENSOR_TAG_NAME;
    static const std::string OVMS_PY_TENSOR_TAG_NAME;
    const PythonCalculatorsPluginApi* api = nullptr;

    static absl::Status pluginFailed(CalculatorContext* cc, PythonPluginResult result, const std::string& message) {
        LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << message;
        const auto code = result == PythonPluginResult::PYTHON_EXCEPTION ? absl::StatusCode::kInternal : absl::StatusCode::kUnknown;
        return absl::Status(code, "Error occurred during graph execution");
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(cc->Inputs().GetTags().size() == 1);
        RET_CHECK(cc->Outputs().GetTags().size() == 1);
        RET_CHECK((*(cc->Inputs().GetTags().begin()) == OV_TENSOR_TAG_NAME && *(cc->Outputs().GetTags().begin()) == OVMS_PY_TENSOR_TAG_NAME) || (*(cc->Inputs().GetTags().begin()) == OVMS_PY_TENSOR_TAG_NAME && *(cc->Outputs().GetTags().begin()) == OV_TENSOR_TAG_NAME));
        if (*(cc->Inputs().GetTags().begin()) == OV_TENSOR_TAG_NAME) {
            RET_CHECK(cc->Options<PyTensorOvTensorConverterCalculatorOptions>().tag_to_output_tensor_names().count(OVMS_PY_TENSOR_TAG_NAME) > 0);
            if (cc->Options<PyTensorOvTensorConverterCalculatorOptions>().tag_to_output_tensor_names().count(OVMS_PY_TENSOR_TAG_NAME) > 1)
                LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->GetNodeName() << "] tag_to_output_tensor_names map contains some keys that will be ignored";
            cc->Inputs().Tag(OV_TENSOR_TAG_NAME).Set<ov::Tensor>();
            cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Set<PyObjectHandle>();
        } else {
            if (cc->Options<PyTensorOvTensorConverterCalculatorOptions>().tag_to_output_tensor_names().count(OVMS_PY_TENSOR_TAG_NAME) > 0)
                LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->GetNodeName() << "] tag_to_output_tensor_names map contains some keys that will be ignored";
            cc->Inputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Set<PyObjectHandle>();
            cc->Outputs().Tag(OV_TENSOR_TAG_NAME).Set<ov::Tensor>();
        }

        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Open start";
        api = getPythonCalculatorsPluginApi();
        if (api == nullptr) {
            return absl::Status(absl::StatusCode::kFailedPrecondition, "Python calculators plugin is not loaded");
        }
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Process start";
        for (const std::string& tag : cc->Inputs().GetTags()) {
            if (cc->Inputs().Tag(tag).IsEmpty()) {
                LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Error occurred during reading inputs. Unexpected empty packet received on input: " << tag;
                RET_CHECK(false);
            }
        }

        std::string message;
        if (*(cc->Inputs().GetTags().begin()) == OV_TENSOR_TAG_NAME) {
            const auto& inputTensor = cc->Inputs().Tag(OV_TENSOR_TAG_NAME).Get<ov::Tensor>();
            std::vector<int64_t> shape;
            for (const auto& dim : inputTensor.get_shape()) {
                if (dim > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                           << "dimension exceeded during conversion: " << dim;
                }
                shape.push_back(static_cast<int64_t>(dim));
            }
            // Existence of the key validated in GetContract
            const auto& outputName = cc->Options<PyTensorOvTensorConverterCalculatorOptions>().tag_to_output_tensor_names().at(OVMS_PY_TENSOR_TAG_NAME);
            const std::string& datatype = toKfsString(ovElementTypeToOvmsPrecision(inputTensor.get_element_type()));
            if (datatype == "UNDEFINED") {
                return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                       << "Undefined precision in input tensor: " << inputTensor.get_element_type();
            }
            void* pyTensor = nullptr;
            const auto result = api->createTensor(outputName, inputTensor.data(), shape, datatype, inputTensor.get_byte_size(), pyTensor, message);
            if (result != PythonPluginResult::OK) {
                return pluginFailed(cc, result, message);
            }
            cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Add(new PyObjectHandle(pyTensor, api->releaseObject), cc->InputTimestamp());
        } else {
            const auto& inputTensor = cc->Inputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Get<PyObjectHandle>();
            std::string datatype;
            std::vector<int64_t> dims;
            const void* data = nullptr;
            size_t bufferSize = 0;
            const auto result = api->getTensorInfo(inputTensor.get(), datatype, dims, data, bufferSize, message);
            if (result != PythonPluginResult::OK) {
                return pluginFailed(cc, result, message);
            }
            const auto precision = ovmsPrecisionToIE2Precision(fromKfsString(datatype));
            if (precision == ov::element::Type_t::dynamic) {
                return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                       << "Undefined precision in input python tensor: " << datatype;
            }
            ov::Shape shape;
            for (const auto& dim : dims) {
                if (dim < 0) {
                    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                           << "dimension negative during conversion: " << dim;
                }
                shape.push_back(dim);
            }
            auto output = std::make_unique<ov::Tensor>(precision, shape);
            if (bufferSize != output->get_byte_size()) {
                return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                       << "python buffer size: " << bufferSize << "; OV tensor size: " << output->get_byte_size() << "; mismatch";
            }
            std::memcpy(output->data(), data, output->get_byte_size());
            cc->Outputs().Tag(OV_TENSOR_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        }

        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

const std::string PyTensorOvTensorConverterCalculator::OV_TENSOR_TAG_NAME{"OVTENSOR"};
const std::string PyTensorOvTensorConverterCalculator::OVMS_PY_TENSOR_TAG_NAME{"OVMS_PY_TENSOR"};

REGISTER_CALCULATOR(PyTensorOvTensorConverterCalculator);
}  // namespace mediapipe
