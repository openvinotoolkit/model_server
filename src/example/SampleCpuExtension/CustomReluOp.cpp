// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "CustomReluOp.hpp"

using namespace TemplateExtension;

//! [op:ctor]
CustomReluOp::CustomReluOp(const ov::Output<ov::Node>& arg) :
    Op({arg}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomReluOp::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomReluOp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() != 2, "Incorrect number of new arguments");

    return std::make_shared<CustomReluOp>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomReluOp::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomReluOp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto in = inputs[0];
    auto out = outputs[0];
    static bool wasCalled = false;
    if (!wasCalled) {
        std::cout << "Running Relu custom kernel for the first time (next messages won't be printed)"
                  << std::endl;
        wasCalled = true;
    }
    // std::cout << "executing custom layer" << std::endl;
    const float* in_data = in.data<float>();
    float* out_data = out.data<float>();
    for (size_t j = 0; j < in.get_size(); j++) {
        out_data[j] = in_data[j] < 0 ? 0 : in_data[j];
    }

    return true;
}

bool CustomReluOp::has_evaluate() const {
    return true;
}
//! [op:evaluate]
