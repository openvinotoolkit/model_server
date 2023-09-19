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
#include "pyobject.hpp"

#include "mediapipe/framework/calculator_framework.h"

#include <pybind11/embed.h> // everything needed for embedding

namespace py = pybind11;
using namespace py::literals;

namespace mediapipe {

class PythonSessionCalculator : public CalculatorBase {
    std::shared_ptr<PYOBJECT> pyobjectClass;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(ERROR) << "PythonSessionCalculator::GetContract";
        RET_CHECK(cc->Inputs().GetTags().empty());
        RET_CHECK(cc->Outputs().GetTags().empty());
        cc->OutputSidePackets().Tag("PYOBJECT").Set<std::shared_ptr<PYOBJECT>>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonSessionCalculator::Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonSessionCalculator::Open";
        auto session = std::make_shared<PYOBJECT>();
        py::gil_scoped_acquire acquire;
        py::print("PYTHON SESSION: Acquired GIL");
        py::exec(R"(
            import time
            time.sleep(2)
            print('slept for 2s')
        )");
        cc->OutputSidePackets().Tag("PYOBJECT").Set(MakePacket<std::shared_ptr<PYOBJECT>>(session));
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonSessionCalculator::Process";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonSessionCalculator);

}  // namespace mediapipe