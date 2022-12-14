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
#include "deserialization.hpp"

namespace ovms {

template <>
Status InputSink<ov::InferRequest&>::give(const std::string& name, ov::Tensor& tensor) {
    Status status;
    try {
        requester.set_tensor(name, tensor);
        // OV implementation the ov::Exception is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const ov::Exception& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }

    return status;
}

ov::Tensor makeTensor(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo) {
    ov::Shape shape;
    for (size_t i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        shape.push_back(requestInput.tensor_shape().dim(i).size());
    }
    ov::element::Type precision = tensorInfo->getOvPrecision();
    return ov::Tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(requestInput.tensor_content().data())));
}

ov::Tensor makeTensor(const ::inference::ModelInferRequest::InferInputTensor& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo,
    const std::string& buffer) {
    ov::Shape shape;
    for (size_t i = 0; i < requestInput.shape_size(); i++) {
        shape.push_back(requestInput.shape().at(i));
    }
    ov::element::Type precision = tensorInfo->getOvPrecision();
    ov::Tensor tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(buffer.data())));
    return tensor;
}
ov::Tensor makeTensor(const ::inference::ModelInferRequest::InferInputTensor& requestInput,
    const std::shared_ptr<TensorInfo>& tensorInfo) {
    ov::Shape shape;
    for (size_t i = 0; i < requestInput.shape_size(); i++) {
        shape.push_back(requestInput.shape().at(i));
    }
    ov::element::Type precision = tensorInfo->getOvPrecision();
    ov::Tensor tensor(precision, shape);
    return tensor;
}

}  // namespace ovms
