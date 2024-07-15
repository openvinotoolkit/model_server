//*****************************************************************************
// Copyright 2021-2022 Intel Corporation
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

#include "capi_frontend/buffer.hpp"
#include "itensorfactory.hpp"
#include "logging.hpp"

namespace ovms {

template <>
Status InputSink<ov::InferRequest&>::give(const std::string& name, ov::Tensor& tensor) {
    OVMS_PROFILE_FUNCTION();
    Status status;
    try {
        OV_LOGGER("ov::InferRequest: {}, request.set_tensor({}, tensor: {})", reinterpret_cast<void*>(&requester), name, reinterpret_cast<void*>(&tensor));
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
ov::Tensor makeTensor(const InferenceTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo, IOVTensorFactory* factory) {
    OVMS_PROFILE_FUNCTION();
    ov::Shape shape;
    OV_LOGGER("ov::Shape(): {}", (void*)&shape);
    for (const auto& dim : requestInput.getShape()) {
        OV_LOGGER("ov::Shape::push_back({})", dim);
        shape.push_back(dim);
    }
    ov::element::Type_t precision = tensorInfo->getOvPrecision();
    if (!requestInput.getBuffer()->getByteSize()) {
        OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
        return ov::Tensor(precision, shape);
    }
    if (requestInput.getBuffer()->getBufferType() == OVMS_BUFFERTYPE_CPU) {
        OV_LOGGER("ov::Tensor({}, shape, data:{})", toString(ovms::ovElementTypeToOvmsPrecision(precision)), const_cast<void*>(reinterpret_cast<const void*>(requestInput.getBuffer()->data())));
        return ov::Tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(requestInput.getBuffer()->data())));
    } else {
        // TODO FIXME check ptr
        //       return factory->create(precision, shape, requestInput.getBuffer()->data());
        if (!factory) {
            SPDLOG_DEBUG("Tried to use empty tensor factory");
            return ov::Tensor();
        }
        SPDLOG_ERROR("ER");
        auto t = factory->create(precision, shape, requestInput.getBuffer()->data());
        SPDLOG_ERROR("ER:{}", (void*)&t);
        return t;
        // TODO instead of context pass in factory of tensors
    }
}

ov::Tensor makeTensor(const tensorflow::TensorProto& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    OV_LOGGER("ov::Shape()");
    ov::Shape shape;
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        OV_LOGGER("ov::Shape::push_back({})", requestInput.tensor_shape().dim(i).size());
        shape.push_back(requestInput.tensor_shape().dim(i).size());
    }
    ov::element::Type_t precision = tensorInfo->getOvPrecision();
    if (!requestInput.tensor_content().size()) {
        OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
        return ov::Tensor(precision, shape);
    }
    OV_LOGGER("ov::Tensor({}, shape, data)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
    return ov::Tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(requestInput.tensor_content().data())));
}

ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo,
    const std::string& buffer) {
    OVMS_PROFILE_FUNCTION();
    OV_LOGGER("ov::Shape()");
    ov::Shape shape;
    for (int i = 0; i < requestInput.shape_size(); i++) {
        OV_LOGGER("ov::Shape::push_back({})", requestInput.shape().at(i));
        shape.push_back(requestInput.shape().at(i));
    }
    ov::element::Type precision = tensorInfo->getOvPrecision();
    if (!buffer.size()) {
        OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
        return ov::Tensor(precision, shape);
    }
    OV_LOGGER("ov::Tensor({}, shape, data)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
    return ov::Tensor(precision, shape, const_cast<void*>(reinterpret_cast<const void*>(buffer.data())));
}
ov::Tensor makeTensor(const ::KFSRequest::InferInputTensor& requestInput,
    const std::shared_ptr<const TensorInfo>& tensorInfo) {
    OVMS_PROFILE_FUNCTION();
    OV_LOGGER("ov::Shape()");
    ov::Shape shape;
    for (int i = 0; i < requestInput.shape_size(); i++) {
        OV_LOGGER("ov::Shape::push_back({})", requestInput.shape().at(i));
        shape.push_back(requestInput.shape().at(i));
    }

    ov::element::Type precision = tensorInfo->getOvPrecision();
    OV_LOGGER("ov::Tensor({}, shape)", toString(ovms::ovElementTypeToOvmsPrecision(precision)));
    ov::Tensor tensor(precision, shape);
    return tensor;
}

template <>
bool specifiesOutputs(const InferenceRequest& request) {
    return true;  // TODO FIXME
}

template <>
Status getTensor(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor) {
    return request.getInput(name.c_str(), tensor);
}

template <>
class RequestTensorExtractor<InferenceRequest, const InferenceTensor**, true> {
public:
    static Status extract(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor);
};
// TODO why we need this defined outside class? Linking issues otherwise
Status RequestTensorExtractor<InferenceRequest, const InferenceTensor**, true>::extract(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor) {
    SPDLOG_TRACE("Extracting output: {}", name);
    return request.getOutput(name.c_str(), tensor);
}

template <>
class RequestTensorExtractor<InferenceRequest, const InferenceTensor**, false> {
public:
    static Status extract(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor) {
        SPDLOG_TRACE("Extracting input", name);
        return request.getInput(name.c_str(), tensor);
    };
};

template class RequestTensorExtractor<InferenceRequest, const InferenceTensor**, false>;
template class RequestTensorExtractor<InferenceRequest, const InferenceTensor**, true>;
}  // namespace ovms
