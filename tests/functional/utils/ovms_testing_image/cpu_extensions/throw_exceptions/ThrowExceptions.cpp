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

class ThrowExceptions : public ov::op::Op {

public:
    OPENVINO_OP("Multiply", "opset1");
    /**
        NOTE:
        This extension was written against Resnet50-Binary model.
        Intention is to "hijack" all layers with type="Multiply" (Our method 'evaluate(...)' will be called instead).
        We use "Multiply" because it is first layer type that use input tensor values applied to model.
        Whole purpose of this extension is to use insted "<layer id="2" ... >" from resnet50-binary-0001.xml:
            <layer id="2" name="Mul_/Fused_Mul_" type="Multiply" version="opset1">
                <input>
                        <port id="0"> ... </port>   // id="0" - input parameter layer.
                        ...
                </input>
                ...
            </layer>
    **/
    ThrowExceptions() = default;
    ThrowExceptions(const ov::Output<ov::Node>& arg) :
        Op({arg}) {
        constructor_validate_and_infer_types();
        std::cout << "ThrowExceptions() constructor call";
    }

    void validate_and_infer_types() {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<ThrowExceptions>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override  {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        std::cout << "Executing ThrowExceptions evaluate()" << std::endl;
        auto in = inputs[0];
        const float* in_data = in.data<float>();
        switch(int(in_data[0])) {
            case SIGFPE:   // DIV by ZERO
                auto foo = 0;
                std::cout << "SIGFPE " << std::endl;
                auto x = 0xDEADF00D / foo;  // Goodbye cruel world.
        }
        return true;
    }
    bool has_evaluate() const override {
        return true;
    }
};
//! [op:header]

}  // namespace TemplateExtension

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({std::make_shared<ov::OpExtension<TemplateExtension::ThrowExceptions>>()}));
