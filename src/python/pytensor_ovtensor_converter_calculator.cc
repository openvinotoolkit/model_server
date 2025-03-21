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
#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>

#pragma warning(push)
#pragma warning(disable : 4005 4018 4309 4018 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

#include "../precision.hpp"
#include "python_backend.hpp"
#include "src/python/ovms_py_tensor.hpp"
#include "src/python/pytensor_ovtensor_converter_calculator.pb.h"

namespace py = pybind11;
using namespace py::literals;
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
    mediapipe::Timestamp outputTimestamp;
    static const std::string OV_TENSOR_TAG_NAME;
    static const std::string OVMS_PY_TENSOR_TAG_NAME;

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
            cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Set<PyObjectWrapper<py::object>>();
        } else {
            if (cc->Options<PyTensorOvTensorConverterCalculatorOptions>().tag_to_output_tensor_names().count(OVMS_PY_TENSOR_TAG_NAME) > 0)
                LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->GetNodeName() << "] tag_to_output_tensor_names map contains some keys that will be ignored";
            cc->Inputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Set<PyObjectWrapper<py::object>>();
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
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Process start";
        py::gil_scoped_acquire acquire;
        try {
            PythonBackend pythonBackend;

            for (const std::string& tag : cc->Inputs().GetTags()) {
                if (cc->Inputs().Tag(tag).IsEmpty()) {
                    LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Error occurred during reading inputs. Unexpected empty packet received on input: " << tag;
                    RET_CHECK(false);
                }
            }

            if (*(cc->Inputs().GetTags().begin()) == OV_TENSOR_TAG_NAME) {
                auto& inputTensor = cc->Inputs().Tag(OV_TENSOR_TAG_NAME).Get<ov::Tensor>();

                std::unique_ptr<PyObjectWrapper<py::object>> outputPyTensor;
                std::vector<py::ssize_t> shape;
#pragma warning(push)
#pragma warning(disable : 4018)
                for (const auto& dim : inputTensor.get_shape()) {
                    if (dim > std::numeric_limits<py::ssize_t>::max()) {
                        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                               << "dimension exceeded during conversion: " << dim;
                    }
                    shape.push_back(dim);
                }
#pragma warning(pop)
                const auto& options = cc->Options<PyTensorOvTensorConverterCalculatorOptions>();
                const auto& tagOutputNameMap = options.tag_to_output_tensor_names();
                const auto& outputName = tagOutputNameMap.at(OVMS_PY_TENSOR_TAG_NAME);  // Existence of the key validated in GetContract
                const std::string datatype = toKfsString(ovElementTypeToOvmsPrecision(inputTensor.get_element_type()));
                if (datatype == "UNDEFINED") {
                    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                           << "Undefined precision in input tensor: " << inputTensor.get_element_type();
                }

                pythonBackend.createOvmsPyTensor(
                    outputName,
                    const_cast<void*>((const void*)inputTensor.data()),
                    shape,
                    datatype,
                    inputTensor.get_byte_size(),
                    outputPyTensor,
                    true);
                cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Add(outputPyTensor.release(), cc->InputTimestamp());
            } else {
                if (*(cc->Inputs().GetTags().begin()) == OVMS_PY_TENSOR_TAG_NAME) {
                    auto& inputTensor = cc->Inputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Get<PyObjectWrapper<py::object>>();
                    pythonBackend.validateOvmsPyTensor(inputTensor.getObject());
                    const auto precision = ovmsPrecisionToIE2Precision(fromKfsString(inputTensor.getProperty<std::string>("datatype")));
                    if (precision == ov::element::Type_t::dynamic) {
                        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                               << "Undefined precision in input python tensor: " << inputTensor.getProperty<std::string>("datatype");
                    }
                    ov::Shape shape;
                    for (const auto& dim : inputTensor.getProperty<std::vector<py::ssize_t>>("shape")) {
                        if (dim < 0) {
                            return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                                   << "dimension negative during conversion: " << dim;
                        }
                        shape.push_back(dim);
                    }
                    const void* data = reinterpret_cast<const void*>(inputTensor.getProperty<void*>("ptr"));
                    size_t bufferSize = inputTensor.getProperty<size_t>("size");
                    std::unique_ptr<ov::Tensor> output = std::make_unique<ov::Tensor>(precision, shape);
                    if (bufferSize != output->get_byte_size()) {
                        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                               << "python buffer size: " << bufferSize << "; OV tensor size: " << output->get_byte_size() << "; mismatch";
                    }
                    memcpy((*output).data(), const_cast<void*>(data), output->get_byte_size());
                    cc->Outputs().Tag(OV_TENSOR_TAG_NAME).Add(output.release(), cc->InputTimestamp());
                }
            }
        } catch (const pybind11::error_already_set& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution");
        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            return absl::Status(absl::StatusCode::kUnknown, "Error occurred during graph execution");
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            return absl::Status(absl::StatusCode::kUnknown, "Error occurred during graph execution");
        }

        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

const std::string PyTensorOvTensorConverterCalculator::OV_TENSOR_TAG_NAME{"OVTENSOR"};
const std::string PyTensorOvTensorConverterCalculator::OVMS_PY_TENSOR_TAG_NAME{"OVMS_PY_TENSOR"};

REGISTER_CALCULATOR(PyTensorOvTensorConverterCalculator);
}  // namespace mediapipe
