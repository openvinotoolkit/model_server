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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "logging.hpp"
#include "tensorinfo.hpp"

namespace ovms {

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const Shape& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape_3(shape),
    layout(getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape_3(shape),
    layout(getDefaultLayout()) {
}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const shape_t& shape,
    const layout_t& layout) :
    name(name),
    mapping(""),
    precision(precision),
    shape_3(shape),
    layout(layout) {
}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const Shape& shape,
    const layout_t& layout) :
    name(name),
    mapping(""),
    precision(precision),
    shape_3(shape),
    layout(layout) {
}

TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const shape_t& shape,
    const layout_t& layout) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape_3(shape),
    layout(layout) {
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const Shape& shape,
    const layout_t& layout) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape_3(shape),
    layout(layout) {
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape_3(shape),
    layout(getDefaultLayout()) {
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

const Precision TensorInfo::getPrecision() const {
    return precision;
}

void TensorInfo::setPrecision(const ovms::Precision& requestedPrecision) {
    precision = requestedPrecision;
}

tensorflow::DataType TensorInfo::getPrecisionAsDataType() const {
    return getPrecisionAsDataType(precision);
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
    return ovmsPrecisionToIE2Precision(precision);
}

std::string TensorInfo::getPrecisionAsString() const {
    return getPrecisionAsString(precision);
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

std::string TensorInfo::getStringFromLayout(const layout_t& layout) {
    return layout;
}

const layout_t& TensorInfo::getLayout() const {
    return layout;
}

bool TensorInfo::isInfluencedByDemultiplexer() const {
    return influencedByDemultiplexer;
}

void TensorInfo::setShape(const Shape& shape) {
    this->shape_3 = shape;
}

const Shape& TensorInfo::getShape_3() const {
    return this->shape_3;
}

void TensorInfo::setLayout(const layout_t& layout) {
    this->layout = layout;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithNewShape(const Shape& shape) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->shape_3 = shape;
    copy->layout = getDefaultLayout();
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithEffectiveDimensionPrefix(const Dimension& dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->shape_3.emplace(copy->shape_3.begin(), dim);  // TODO check together with pipeline definiton apply demultiplexer to shape
    return copy;
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return (this->getShape_3() == other.getShape_3()) &&
           (this->getPrecision() == other.getPrecision());
}

bool TensorInfo::isTensorUnspecified() const {
    return this->getPrecision() == Precision::UNDEFINED;
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
    return std::make_shared<TensorInfo>("", Precision::UNDEFINED, Shape{});
}

const Dimension& TensorInfo::getBatchSize() const {
    return getShape_3()[0];  // TODO use layout
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

const layout_t& TensorInfo::getDefaultLayout() {
    static const layout_t defaultLayout{"N..."};
    return defaultLayout;
}

}  // namespace ovms
