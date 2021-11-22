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
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "logging.hpp"
#include "tensorinfo.hpp"

namespace ovms {

TensorInfo::TensorInfo(const std::string& name,
    const InferenceEngine::Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(InferenceEngine::Layout::ANY) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const InferenceEngine::Precision& precision,
    const shape_t& shape,
    const InferenceEngine::Layout& layout) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(layout) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const InferenceEngine::TensorDesc& tensorDesc) :
    name(name),
    mapping(""),
    precision(tensorDesc.getPrecision()),
    shape(tensorDesc.getDims()),
    layout(tensorDesc.getLayout()) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const InferenceEngine::Precision& precision,
    const shape_t& shape,
    const InferenceEngine::Layout& layout) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape(shape),
    layout(layout) {
    this->updateEffectiveShape();
}

const std::string& TensorInfo::getName() const {
    return name;
}

const std::string& TensorInfo::getMappedName() const {
    return mapping.size() == 0 ? name : mapping;
}

void TensorInfo::setMappedName(const std::string& mappedName) {
    mapping = mappedName;
}

const InferenceEngine::Precision TensorInfo::getPrecision() const {
    return precision;
}

void TensorInfo::setPrecision(const InferenceEngine::Precision& requestedPrecision) {
    precision = requestedPrecision;
}

const tensorflow::DataType TensorInfo::getPrecisionAsDataType() const {
    return getPrecisionAsDataType(precision);
}

const tensorflow::DataType TensorInfo::getPrecisionAsDataType(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return tensorflow::DataType::DT_FLOAT;
    case InferenceEngine::Precision::I32:
        return tensorflow::DataType::DT_INT32;
    case InferenceEngine::Precision::I8:
        return tensorflow::DataType::DT_INT8;
    case InferenceEngine::Precision::U8:
        return tensorflow::DataType::DT_UINT8;
    case InferenceEngine::Precision::FP16:
        return tensorflow::DataType::DT_HALF;
    // case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
    case InferenceEngine::Precision::I16:
        return tensorflow::DataType::DT_INT16;
    case InferenceEngine::Precision::U16:
        return tensorflow::DataType::DT_UINT16;
    case InferenceEngine::Precision::U64:
        return tensorflow::DataType::DT_UINT64;
    case InferenceEngine::Precision::I64:
        return tensorflow::DataType::DT_INT64;
    // case InferenceEngine::Precision::BIN:   return tensorflow::DataType::
    case InferenceEngine::Precision::BOOL:
        return tensorflow::DataType::DT_BOOL;
    default:
        return tensorflow::DataType::DT_INVALID;
    }
}

ov::element::Type TensorInfo::getPrecisionFromDataType(tensorflow::DataType dataType) {
    switch (dataType) {
    case tensorflow::DataType::DT_FLOAT:
        return ov::element::f32;
    case tensorflow::DataType::DT_INT32:
        return ov::element::i32;
    default:
        return ov::element::undefined;
    }
}

const std::string TensorInfo::getPrecisionAsString() const {
    return getPrecisionAsString(precision);
}

const std::string TensorInfo::getPrecisionAsString(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return "FP32";
    case InferenceEngine::Precision::I32:
        return "I32";
    case InferenceEngine::Precision::I8:
        return "I8";
    case InferenceEngine::Precision::U8:
        return "U8";
    case InferenceEngine::Precision::FP16:
        return "FP16";
        // case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
    case InferenceEngine::Precision::I16:
        return "I16";
    case InferenceEngine::Precision::U16:
        return "U16";
    case InferenceEngine::Precision::I64:
        return "I64";
    case InferenceEngine::Precision::BOOL:
        return "BOOL";
    default:
        return "DT_INVALID";
    }
}

const std::string TensorInfo::getDataTypeAsString(tensorflow::DataType dataType) {
    switch (dataType) {
    case tensorflow::DataType::DT_FLOAT:
        return "FP32";
    case tensorflow::DataType::DT_INT32:
        return "I32";
    case tensorflow::DataType::DT_INT8:
        return "I8";
    case tensorflow::DataType::DT_UINT8:
        return "U8";
    case tensorflow::DataType::DT_HALF:
        return "FP16";
    case tensorflow::DataType::DT_INT16:
        return "I16";
    case tensorflow::DataType::DT_UINT16:
        return "U16";
    case tensorflow::DataType::DT_UINT64:
        return "U64";
    case tensorflow::DataType::DT_INT64:
        return "I64";
    case tensorflow::DataType::DT_BOOL:
        return "BOOL";
    case tensorflow::DataType::DT_STRING:
        return "STRING";
    default:
        return "DT_INVALID";
    }
}

InferenceEngine::Layout TensorInfo::getLayoutFromString(const std::string& layout) {
    if (layout == "ANY")
        return InferenceEngine::Layout::ANY;
    if (layout == "NCHW")
        return InferenceEngine::Layout::NCHW;
    if (layout == "NHWC")
        return InferenceEngine::Layout::NHWC;
    if (layout == "NCDHW")
        return InferenceEngine::Layout::NCDHW;
    if (layout == "NDHWC")
        return InferenceEngine::Layout::NDHWC;
    if (layout == "OIHW")
        return InferenceEngine::Layout::OIHW;
    if (layout == "GOIHW")
        return InferenceEngine::Layout::GOIHW;
    if (layout == "OIDHW")
        return InferenceEngine::Layout::OIDHW;
    if (layout == "GOIDHW")
        return InferenceEngine::Layout::GOIDHW;
    if (layout == "SCALAR")
        return InferenceEngine::Layout::SCALAR;
    if (layout == "C")
        return InferenceEngine::Layout::C;
    if (layout == "CHW")
        return InferenceEngine::Layout::CHW;
    if (layout == "HW")
        return InferenceEngine::Layout::HW;
    if (layout == "HWC")
        return InferenceEngine::Layout::HWC;
    if (layout == "NC")
        return InferenceEngine::Layout::NC;
    if (layout == "CN")
        return InferenceEngine::Layout::CN;
    if (layout == "BLOCKED")
        return InferenceEngine::Layout::BLOCKED;

    return InferenceEngine::Layout::ANY;
}

std::string TensorInfo::getStringFromLayout(InferenceEngine::Layout layout) {
    switch (layout) {
    case InferenceEngine::Layout::ANY:
        return "ANY";
    case InferenceEngine::Layout::NCHW:
        return "NCHW";
    case InferenceEngine::Layout::NHWC:
        return "NHWC";
    case InferenceEngine::Layout::NCDHW:
        return "NCDHW";
    case InferenceEngine::Layout::NDHWC:
        return "NDHWC";
    case InferenceEngine::Layout::OIHW:
        return "OIHW";
    case InferenceEngine::Layout::GOIHW:
        return "GOIHW";
    case InferenceEngine::Layout::OIDHW:
        return "OIDHW";
    case InferenceEngine::Layout::GOIDHW:
        return "GOIDHW";
    case InferenceEngine::Layout::SCALAR:
        return "SCALAR";
    case InferenceEngine::Layout::C:
        return "C";
    case InferenceEngine::Layout::CHW:
        return "CHW";
    case InferenceEngine::Layout::HW:
        return "HW";
    case InferenceEngine::Layout::HWC:
        return "HWC";
    case InferenceEngine::Layout::NC:
        return "NC";
    case InferenceEngine::Layout::CN:
        return "CN";
    case InferenceEngine::Layout::BLOCKED:
        return "BLOCKED";
    }
    return "";
}

const InferenceEngine::Layout& TensorInfo::getLayout() const {
    return layout;
}

const shape_t& TensorInfo::getShape() const {
    return shape;
}

bool TensorInfo::isInfluencedByDemultiplexer() const {
    return influencedByDemultiplexer;
}

const shape_t& TensorInfo::getEffectiveShape() const {
    return effectiveShape.size() > 0 ? effectiveShape : shape;
}

void TensorInfo::setShape(const shape_t& shape) {
    this->shape = shape;
    this->updateEffectiveShape();
}

void TensorInfo::setLayout(InferenceEngine::Layout layout) {
    this->layout = layout;
    this->updateEffectiveShape();
}

void TensorInfo::updateEffectiveShape() {
    this->effectiveShape = this->getTensorDesc().getBlockingDesc().getBlockDims();
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithNewShape(const shape_t& shape) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->shape = shape;
    copy->layout = InferenceEngine::Layout::ANY;
    copy->updateEffectiveShape();
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithEffectiveDimensionPrefix(size_t dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->effectiveShape = this->getEffectiveShape();
    copy->effectiveShape.insert(copy->effectiveShape.begin(), dim);
    return copy;
}

const InferenceEngine::TensorDesc TensorInfo::getTensorDesc() const {
    return InferenceEngine::TensorDesc{precision, shape, layout};
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return this->getEffectiveShape() == other.getEffectiveShape() &&
           this->getPrecision() == other.getPrecision();
}

bool TensorInfo::isTensorUnspecified() const {
    return this->getPrecision() == InferenceEngine::Precision::UNSPECIFIED;
}

std::string TensorInfo::shapeToString(const shape_t& shape) {
    std::ostringstream oss;
    oss << "(";
    size_t i = 0;
    if (shape.size() > 0) {
        for (; i < shape.size() - 1; i++) {
            oss << shape[i] << ",";
        }
        oss << shape[i];
    }
    oss << ")";

    return oss.str();
}

std::string TensorInfo::tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape) {
    std::ostringstream oss;
    oss << "(";
    int i = 0;
    if (tensorShape.dim_size() > 0) {
        for (; i < tensorShape.dim_size() - 1; i++) {
            oss << tensorShape.dim(i).size() << ",";
        }
        oss << tensorShape.dim(i).size();
    }
    oss << ")";

    return oss.str();
}

std::shared_ptr<TensorInfo> TensorInfo::getUnspecifiedTensorInfo() {
    std::shared_ptr<TensorInfo> info = std::make_shared<TensorInfo>("", InferenceEngine::Precision::UNSPECIFIED, shape_t{});
    return info;
}

std::string TensorInfo::tensorDescToString(const InferenceEngine::TensorDesc& desc) {
    std::stringstream ss;
    ss << "shape: " << shapeToString(desc.getDims())
       << " effective shape: " << shapeToString(desc.getBlockingDesc().getBlockDims())
       << " precision: " << getPrecisionAsString(desc.getPrecision())
       << " layout: " << getStringFromLayout(desc.getLayout());
    return ss.str();
}

const size_t TensorInfo::getBatchSize() const {
    return getEffectiveShape()[0];
}

}  // namespace ovms
