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
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../precision.hpp"
#include "../python/ovms_py_tensor.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "../python/python_backend.hpp"
#include "src/mediapipe_calculators/pytensor_ovtensor_converter.pb.h"

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

namespace mediapipe {

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
            cc->Inputs().Tag(OV_TENSOR_TAG_NAME).Set<ov::Tensor>();
            cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Set<PyObjectWrapper<py::object>>();
        } else {
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
        if (*(cc->Inputs().GetTags().begin()) == OV_TENSOR_TAG_NAME) {
            auto& inputTensor = cc->Inputs().Tag(OV_TENSOR_TAG_NAME).Get<ov::Tensor>();
            py::object outTensor;
            std::unique_ptr<PyObjectWrapper<py::object>> outputPyTensor = std::make_unique<PyObjectWrapper<py::object>>(outTensor);
            PythonBackend pythonBackend;
            std::vector<py::ssize_t> shape;
            for (const auto& dim : inputTensor.get_shape()) {
                if (dim > std::numeric_limits<py::ssize_t>::max()) {
                    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                           << "dimension exceeded during conversion: " << dim;
                }
                shape.push_back(dim);
            }
            const auto& options = cc->Options<PyTensorOvTensorConverterCalculatorOptions>();
            const auto tagOutputNameMap = options.tag_to_output_tensor_names();
            auto outputName = tagOutputNameMap.at(OVMS_PY_TENSOR_TAG_NAME).c_str();  // Existence of the key validated in GetContract
            pythonBackend.createOvmsPyTensor(
                outputName,
                const_cast<void*>((const void*)inputTensor.data()),
                shape,
                toString(ovElementTypeToOvmsPrecision(inputTensor.get_element_type())),
                inputTensor.get_byte_size(),
                outputPyTensor,
                true);
            cc->Outputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Add(outputPyTensor.release(), cc->InputTimestamp());
        } else {
            auto& inputTensor = cc->Inputs().Tag(OVMS_PY_TENSOR_TAG_NAME).Get<PyObjectWrapper<py::object>>();
            auto precision = ovmsPrecisionToIE2Precision(fromString(inputTensor.getProperty<std::string>("datatype")));
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
        LOG(INFO) << "PyTensorOvTensorConverterCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

const std::string PyTensorOvTensorConverterCalculator::OV_TENSOR_TAG_NAME{"OVTENSOR"};
const std::string PyTensorOvTensorConverterCalculator::OVMS_PY_TENSOR_TAG_NAME{"OVMS_PY_TENSOR"};

REGISTER_CALCULATOR(PyTensorOvTensorConverterCalculator);
}  // namespace mediapipe
