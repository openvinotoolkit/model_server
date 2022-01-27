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

// in case we change behaviour for this constructor we may need to write additionall tests for TensorInfo intersection / DAGs
TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const Shape& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(getDefaultLayout()) {
}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const shape_t& shape,
    const Layout& layout) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(layout) {
}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const Shape& shape,
    const Layout& layout) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(layout) {
}

TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const shape_t& shape,
    const Layout& layout) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape(shape),
    layout(layout) {
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const Shape& shape,
    const Layout& layout) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape(shape),
    layout(layout) {
}
TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(mapping),
    precision(precision),
    shape(shape),
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

std::string TensorInfo::getStringFromLayout(const Layout& layout) {
    return layout;
}

const Layout& TensorInfo::getLayout() const {
    return layout;
}

bool TensorInfo::isInfluencedByDemultiplexer() const {
    return influencedByDemultiplexer;
}

void TensorInfo::setShape(const Shape& shape) {
    this->shape = shape;
}

const Shape& TensorInfo::getShape() const {
    return this->shape;
}

void TensorInfo::setLayout(const Layout& layout) {
    this->layout = layout;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithNewShape(const Shape& shape) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->shape = shape;
    copy->layout = getDefaultLayout();
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithEffectiveDimensionPrefix(const Dimension& dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->shape.emplace(copy->shape.begin(), dim);  // TODO check together with pipeline definiton apply demultiplexer to shape
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createIntersection(const TensorInfo& other) {
    if (this->isTensorUnspecified())
        return std::make_shared<TensorInfo>(other);
    if (other.isTensorUnspecified())
        return std::make_shared<TensorInfo>(*this);
    if ((this->getName() != other.getName()) ||
        (this->getMappedName() != other.getMappedName())) {
        return nullptr;
    }
    Precision precision;
    if (this->getPrecision() != other.getPrecision()) {
        if (this->getPrecision() == Precision::UNDEFINED) {
            precision = other.getPrecision();
        } else if (other.getPrecision() == Precision::UNDEFINED) {
            precision = this->getPrecision();
        } else {
            return nullptr;
        }
    } else {
        precision = this->getPrecision();
    }
    Layout layout;
    if (this->getLayout() != other.getLayout()) {
        if ((this->getLayout() != TensorInfo::getDefaultLayout()) &&
            (other.getLayout() == TensorInfo::getDefaultLayout())) {
                layout = this->getLayout();
        } else if (this->getLayout() == TensorInfo::getDefaultLayout()) {
            layout = other.getLayout();
        } else {
            return nullptr;
        }
    }
    if (this->influencedByDemultiplexer != other.influencedByDemultiplexer)
        return nullptr;
    auto newShape = this->getShape().createIntersection(other.getShape());
    if (newShape == std::nullopt)
        return nullptr;
    return std::make_shared<TensorInfo>(this->getName(),
        this->getMappedName(),
        precision,
        std::move(newShape.value()),
        this->getLayout());
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return (this->getShape() == other.getShape()) &&
           (this->getPrecision() == other.getPrecision());
}

bool TensorInfo::isTensorUnspecified() const {
    return (this->getPrecision() == Precision::UNDEFINED) &&
           (this->getName() == "") &&
           (this->getShape() == Shape());
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

const std::optional<Dimension> TensorInfo::getBatchSize() const {
    const auto batchIndex = this->layout.getBatchIndex();
    if (!batchIndex.has_value()) {
        return std::nullopt;
    }
    if (getShape().size() < batchIndex.value() + 1) {
        throw std::logic_error("batch outside of shape range");
    }
    return getShape()[batchIndex.value()];
}

std::string TensorInfo::asString() const {
    std::stringstream ss;
    ss
        << "name: " << getName() << "; "
        << "mapping_name: " << getMappedName() << "; "
        << "shape: " << getShape().toString() << "; "
        << "precision: " << getPrecisionAsString() << "; "
        << "layout: " << getStringFromLayout(getLayout());
    return ss.str();
}

// in case we change behaviour for this constructor we may need to write additionall tests for TensorInfo intersection / DAGs
const Layout& TensorInfo::getDefaultLayout() {
    static const Layout defaultLayout{"N..."};
    return defaultLayout;
}

}  // namespace ovms
