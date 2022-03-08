// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <openvino/op/op.hpp>

//! [op:header]
namespace TemplateExtension {

class CustomReluOp : public ov::op::Op {
public:
    OPENVINO_OP("CustomReLU", "opset1");

    CustomReluOp() = default;
    CustomReluOp(const ov::Output<ov::Node>& arg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension
