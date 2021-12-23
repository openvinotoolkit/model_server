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
    const Shape& shape) :
    name(name),
    mapping(""),
    precision_2(precision),
    shape_3(shape),
    layout(InferenceEngine::Layout::ANY) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision_2(precision),
    shape(shape),
    shape_3(shape),
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
    shape_3(shape),
    layout(layout) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const shape_t& shape,
    const InferenceEngine::Layout& layout) :
    name(name),
    mapping(""),
    precision_2(precision),
    shape(shape),
    shape_3(shape),
    layout(layout) {
    this->updateEffectiveShape();
}

TensorInfo::TensorInfo(const std::string& name,
    const InferenceEngine::TensorDesc& tensorDesc) :
    name(name),
    mapping(""),
    precision_2(IE1PrecisionToOvmsPrecision(tensorDesc.getPrecision())),
    shape(tensorDesc.getDims()),
    shape_3(shape),
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
    shape_3(shape),
    layout(layout) {
    this->updateEffectiveShape();
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const shape_t& shape,
    const InferenceEngine::Layout& layout) :
    name(name),
    mapping(mapping),
    precision_2(precision),
    shape(shape),
    shape_3(shape),
    layout(layout) {
    this->updateEffectiveShape();
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const Shape& shape,
    const InferenceEngine::Layout& layout) :
    name(name),
    mapping(mapping),
    precision_2(precision),
    shape(shape.getFlatShape()),
    shape_3(shape),
    layout(layout) {
    this->updateEffectiveShape();
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(mapping),
    precision_2(precision),
    shape(shape),
    shape_3(shape),
    layout(InferenceEngine::Layout::ANY) {
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
void TensorInfo::setPrecision(const ovms::Precision& requestedPrecision) {
    precision_2 = requestedPrecision;
}

tensorflow::DataType TensorInfo::getPrecisionAsDataType() const {
    return getPrecisionAsDataType(precision_2);
}

tensorflow::DataType TensorInfo::getPrecisionAsDataType(InferenceEngine::Precision precision) {
    return getPrecisionAsDataType(IE1PrecisionToOvmsPrecision(precision));
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
    return toString(precision);
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

void TensorInfo::setShape_2(const shape_t& shape) {
    this->shape = shape;
    this->updateEffectiveShape();
}

void TensorInfo::setShape(const Shape& shape) {
    this->shape_3 = shape;
}

const Shape& TensorInfo::getShape_3() const {
    return this->shape_3;
}

void TensorInfo::setLayout(InferenceEngine::Layout layout) {
    this->layout = layout;
    this->updateEffectiveShape();
}

void TensorInfo::updateEffectiveShape() {
    // TODO: Get rid of this.

    // this->effectiveShape = this->getTensorDesc().getBlockingDesc().getBlockDims();
    // if (effectiveShape.size() == 0) {
    //     this->shape_2 = this->shape;
    // } else {
    //     this->shape_2 = effectiveShape;
    // }
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithNewShape(const Shape& shape) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->shape = shape.getFlatShape();
    copy->shape_3 = shape;
    copy->layout = InferenceEngine::Layout::ANY;
    copy->updateEffectiveShape();
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithEffectiveDimensionPrefix(dimension_value_t dim) const {
    return createCopyWithEffectiveDimensionPrefix(dim ? dim : DYNAMIC_DIMENSION);
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithEffectiveDimensionPrefix(const Dimension& dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->shape.insert(copy->shape.begin(), dim.getAnyValue());
    copy->shape_3.emplace(copy->shape_3.begin(), dim);  // TODO check together with pipeline definiton apply demultiplexer to shape
    return copy;
}

const InferenceEngine::TensorDesc TensorInfo::getTensorDesc() const {
    // TODO how to
    return InferenceEngine::TensorDesc{ovmsPrecisionToIE1Precision(precision_2), shape, layout};
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return (this->getShape_3() == other.getShape_3()) &&
           (this->getPrecision_2() == other.getPrecision_2());
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
    return getShape_3().getFlatShape()[0];  // TODO use layout
}

std::string TensorInfo::asString() const {
    std::stringstream ss;
    ss
        << "name: " << getName() << "; "
        << "mapping_name: " << getMappedName() << "; "
        << "shape: " << getShape_3().toString() << "; "
        << "precision: " << getPrecisionAsString() << "; "
        << "layout: " << getStringFromLayout(getLayout());
    return ss.str();
}

}  // namespace ovms
