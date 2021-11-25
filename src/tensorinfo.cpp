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
#include <unordered_map>

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
    TensorInfo(name, IE1PrecisionToOvmsPrecision(precision), shape) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision_2(precision),
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
    precision_2(IE1PrecisionToOvmsPrecision(precision)),
    shape(shape),
    layout(layout) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const InferenceEngine::TensorDesc& tensorDesc) :
    name(name),
    mapping(""),
    precision_2(IE1PrecisionToOvmsPrecision(tensorDesc.getPrecision())),
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
    precision_2(IE1PrecisionToOvmsPrecision(precision)),
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
    return ovmsPrecisionToIE1Precision(precision_2);
}
const Precision TensorInfo::getPrecision_2() const {
    return precision_2;
}

void TensorInfo::setPrecision(const InferenceEngine::Precision& requestedPrecision) {
    precision_2 = IE1PrecisionToOvmsPrecision(requestedPrecision);
}

tensorflow::DataType TensorInfo::getPrecisionAsDataType() const {
    return getPrecisionAsDataType(precision_2);
}

tensorflow::DataType TensorInfo::getPrecisionAsDataType(InferenceEngine::Precision precision) {
    return getPrecisionAsDataType(IE1PrecisionToOvmsPrecision(precision));
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

tensorflow::DataType TensorInfo::getPrecisionAsDataType(Precision precision) {
    static std::unordered_map<Precision, tensorflow::DataType> precisionMap{
        {Precision::FP32, tensorflow::DataType::DT_FLOAT},
        {Precision::FP16, tensorflow::DataType::DT_HALF},
        {Precision::I64, tensorflow::DataType::DT_INT64},
        {Precision::I32, tensorflow::DataType::DT_INT32},
        {Precision::I16, tensorflow::DataType::DT_INT16},
        {Precision::I8, tensorflow::DataType::DT_INT8},
        {Precision::U64, tensorflow::DataType::DT_UINT64},
        {Precision::U16, tensorflow::DataType::DT_UINT16},
        {Precision::U8, tensorflow::DataType::DT_UINT8},
        //    {Precision::MIXED, tensorflow::DataType::DT_INVALID},
        //    {Precision::Q78, tensorflow::DataType::DT_INVALID},
        //    {Precision::BIN, tensorflow::DataType::DT_INVALID},
        {Precision::BOOL, tensorflow::DataType::DT_BOOL}
        //    {Precision::CUSTOM, tensorflow::DataType::DT_INVALID}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        // TODO missing precisions
        return tensorflow::DataType::DT_INVALID;
    }
    return it->second;
}

std::string TensorInfo::getPrecisionAsString(Precision precision) {
    static std::unordered_map<Precision, const char*> precisionMap{
        {Precision::BF16, "BF16"},
        {Precision::FP64, "FP64"},
        {Precision::FP32, "FP32"},
        {Precision::FP16, "FP16"},
        {Precision::I64, "I64"},
        {Precision::I32, "I32"},
        {Precision::I16, "I16"},
        {Precision::I8, "I8"},
        {Precision::I4, "I4"},
        {Precision::U64, "U64"},
        {Precision::U32, "U32"},
        {Precision::U16, "U16"},
        {Precision::U8, "U8"},
        {Precision::U4, "U4"},
        {Precision::U1, "U1"},
        {Precision::MIXED, "MIXED"},
        {Precision::Q78, "Q78"},
        {Precision::BIN, "BIN"},
        {Precision::BOOL, "BOOL"},
        {Precision::UNDEFINED, "UNDEFINED"},
        {Precision::CUSTOM, "CUSTOM"}};
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        return "DT_INVALID";  // TODO other way? why translate it to TF equivalent maybe UNDEFINED?
    }
    return it->second;
}

ov::element::Type TensorInfo::getOvPrecision() const {
    return ovmsPrecisionToIE2Precision(precision_2);
}

std::string TensorInfo::getPrecisionAsString() const {
    return getPrecisionAsString(precision_2);
}

std::string TensorInfo::getPrecisionAsString(InferenceEngine::Precision precision) {
    return getPrecisionAsString(IE1PrecisionToOvmsPrecision(precision));
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

const shape_t& TensorInfo::getShape_2() const {
    return shape_2;
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
    this->shape_2 = effectiveShape;
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
    copy->shape_2 = copy->effectiveShape;
    return copy;
}

const InferenceEngine::TensorDesc TensorInfo::getTensorDesc() const {
    // TODO how to
    return InferenceEngine::TensorDesc{ovmsPrecisionToIE1Precision(precision_2), shape, layout};
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return this->getShape_2() == other.getShape_2() &&
           this->getPrecision_2() == other.getPrecision_2();
}

bool TensorInfo::isTensorUnspecified() const {
    return this->getPrecision_2() == Precision::UNDEFINED;
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
