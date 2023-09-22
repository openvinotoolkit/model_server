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
#include <openvino/openvino.hpp>
#include <pybind11/embed.h>  // everything needed for embedding

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#include "pyobject.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace mediapipe {

class PythonBackendCalculator : public CalculatorBase {
    py::object pyobjectClass;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(ERROR) << "PythonBackendCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        cc->InputSidePackets().Tag("PYOBJECT").Set<py::object>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Open";
        LOG(ERROR) << "Python node name:" << cc->NodeName();
        pyobjectClass = cc->InputSidePackets().Tag("PYOBJECT").Get<py::object>();
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Process";

        py::gil_scoped_acquire acquire;
        py::print("PYTHON: Acquired GIL");
        //pyobjectClass.attr("execute")();
        // Read input from the stream
        PYOBJECT inputs = cc->Inputs().Index(0).Get<PYOBJECT>();

        //py::iterator it = pyobjectClass.attr("execute")(inputs);
        //while (it != py::iterator::sentinel()) {
        //    auto result = cast<PYOBJECT>(*it);
        //    cc->Outputs().Index(0).Add(result, myTimestamp);
        //    ++it;
        //}

        ov::Tensor in_tensor = cc->Inputs().Index(0).Get<ov::Tensor>();

        auto* out_tensor = new ov::Tensor(in_tensor.get_element_type(), in_tensor.get_shape());
        std::memcpy(out_tensor->data(), in_tensor.data(), in_tensor.get_byte_size());

        for (int i = 0; i < 10; i++) {
            float* pV = ((float*)out_tensor->data()) + i;
            *pV += 1.0f;
        }

        py::print("PYTHON: Released GIL");
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonBackendCalculator);
}  // namespace mediapipe

/*
class PythonNodeState;
class PythonSession
class PYOBJECT;

class PythonExecutorCalculator : public CalculatorBase {
    PythonNodeState pythonNodeState;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        ...
        cc->Inputs().Index(0).Set<PYOBJECT>();
        cc->Outputs().Index(0).Set<PYOBJECT>();
        cc->InputSidePackets().Tag("PYSESSION").Set<PythonSession>();
        ...
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        ...
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        ...
        auto pythonSession = cc->InputSidePackets().Tag("PYSESSION").Get<PythonSession>();
        this->pythonNodeState = pythonSession[cc->NodeName()];
        ...
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        ...
        py::gil_scoped_acquire acquire;
        ...
        // Read input from the stream
        PYOBJECT inputs = cc->Inputs().Index(0).Get<PYOBJECT>();
        // Checks, translations etc.
        ...
        // Execute user code
        outputs_raw = pythonNodeState.attr("execute")(inputs);
        // Convert results to PYOBJECT
        PYOBJECT outputs = convertToPYOBJECT(outputs_raw)
        // Push outputs down the graph
        cc->Outputs().Index(0).Add(outputs, cc->InputTimestamp());
        ...
        return absl::OkStatus();
    }
};
*/