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
#include "tensorinfo.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "kfs_grpc_inference_service.hpp"
#include "logging.hpp"

namespace ovms {

// in case we change behaviour for this constructor we may need to write additional tests for TensorInfo intersection / DAGs
TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const Shape& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(Layout::getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    name(name),
    mapping(""),
    precision(precision),
    shape(shape),
    layout(Layout::getDefaultLayout()) {
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
    layout(Layout::getDefaultLayout()) {
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

std::string TensorInfo::getPrecisionAsString(Precision precision) {
    return toString(precision);
}

ov::element::Type TensorInfo::getOvPrecision() const {
    return ovmsPrecisionToIE2Precision(precision);
}

std::string TensorInfo::getPrecisionAsString() const {
    return getPrecisionAsString(precision);
}

std::string TensorInfo::getPrecisionAsKFSPrecision(Precision precision) {
    return ovmsPrecisionToKFSPrecision(precision);
}

std::string TensorInfo::getPrecisionAsKFSPrecision() const {
    return getPrecisionAsKFSPrecision(precision);
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
    copy->layout = Layout::getUnspecifiedLayout();
    return copy;
}

std::shared_ptr<TensorInfo> TensorInfo::createCopyWithDemultiplexerDimensionPrefix(const Dimension& dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->shape.emplace(copy->shape.begin(), dim);
    copy->layout = this->getLayout();
    auto batchPosition = copy->layout.find(BATCH_DIMENSION_LETTER);
    if (batchPosition != std::string::npos) {
        copy->layout.replace(batchPosition, 1, std::string(1, UNDEFINED_DIMENSION_CHAR));
    }
    copy->layout = std::string(1, BATCH_DIMENSION_LETTER[0]) + copy->layout;
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
    auto newShape = this->getShape().createIntersection(other.getShape());
    if (!newShape.has_value())
        return nullptr;
    auto layout = this->getLayout().createIntersection(other.getLayout(), newShape.value().size());
    if (!layout.has_value())
        return nullptr;
    return std::make_shared<TensorInfo>(this->getName(),
        this->getMappedName(),
        precision,
        std::move(newShape.value()),
        layout.value());
}

bool TensorInfo::isTensorSpecEqual(const TensorInfo& other) const {
    return (this->getShape() == other.getShape()) &&
           (this->getPrecision() == other.getPrecision()) &&
           (this->getLayout() == other.getLayout());
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

std::string tensorShapeToString(const google::protobuf::RepeatedField<int64_t>& shape) {
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
}  // namespace ovms
