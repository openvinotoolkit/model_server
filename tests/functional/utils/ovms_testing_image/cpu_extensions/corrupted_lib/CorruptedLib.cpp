//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <csignal>
#include <memory>

#include <openvino/op/op.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

//! [op:header]
namespace TemplateExtension {

class CorruptedLib : public ov::op::Op {

public:
    OPENVINO_OP("Multiply", "opset1");
    CorruptedLib() = default;
    CorruptedLib(const ov::Output<ov::Node>& arg) :
        Op({arg}) {
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        std::cout << "Executing CorruptedLib evaluate()" << std::endl;
        return true;
    }
}; // class CorruptedLib
}  // namespace TemplateExtension
