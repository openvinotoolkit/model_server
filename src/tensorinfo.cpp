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

#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#include "logging.hpp"
#include "stringutils.hpp"

namespace ovms {

const std::string STRING_SERIALIZATION_HINT_NAME_SUFFIX = "_string";

// in case we change behaviour for this constructor we may need to write additional tests for TensorInfo intersection / DAGs
TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const Shape& shape) :
    TensorInfo(name, "", precision, shape, Layout::getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const Precision& precision,
    const shape_t& shape) :
    TensorInfo(name, "", precision, shape, Layout::getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const shape_t& shape,
    const Layout& layout) :
    TensorInfo(name, "", precision, shape, layout) {}

TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const Precision& precision,
    const shape_t& shape) :
    TensorInfo(name, mapping, precision, shape, Layout::getDefaultLayout()) {}

TensorInfo::TensorInfo(const std::string& name,
    const ovms::Precision& precision,
    const Shape& shape,
    const Layout& layout) :
    TensorInfo(name, "", precision, shape, layout) {}

TensorInfo::ProcessingHint TensorInfo::getPreProcessingHint() const {
    return this->preProcessingHint;
}

TensorInfo::ProcessingHint TensorInfo::getPostProcessingHint() const {
    return this->postProcessingHint;
}

void TensorInfo::createProcessingHints() {
    // Pre
    size_t expectedDimsForImage = this->influencedByDemultiplexer ? 5 : 4;
    if (this->shape.size() == 2 && this->precision == ovms::Precision::U8 && !this->influencedByDemultiplexer) {
        this->preProcessingHint = TensorInfo::ProcessingHint::STRING_2D_U8;
    } else if (this->shape.size() == 1 && this->precision == ovms::Precision::U8 && this->shape.at(0).isDynamic() && !this->influencedByDemultiplexer) {
        this->preProcessingHint = TensorInfo::ProcessingHint::STRING_1D_U8;
    } else if (this->shape.size() == expectedDimsForImage) {
        this->preProcessingHint = TensorInfo::ProcessingHint::IMAGE;
    } else {
        this->preProcessingHint = TensorInfo::ProcessingHint::NO_PROCESSING;
    }

    // Post
    if (this->precision == ovms::Precision::U8 && this->shape.size() == 2 && endsWith(this->getMappedName(), STRING_SERIALIZATION_HINT_NAME_SUFFIX)) {
        this->postProcessingHint = TensorInfo::ProcessingHint::STRING_2D_U8;
    } else {
        this->postProcessingHint = TensorInfo::ProcessingHint::NO_PROCESSING;
    }
}

TensorInfo::TensorInfo(const std::string& name,
    const std::string& mapping,
    const ovms::Precision& precision,
    const shape_t& shape,
    const Layout& layout) :
    TensorInfo(name, mapping, precision, Shape(shape), layout) {}

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
    createProcessingHints();
}

const std::string& TensorInfo::getName() const {
    return name;
}

const std::string& TensorInfo::getMappedName() const {
    return mapping.size() == 0 ? name : mapping;
}

const Precision TensorInfo::getPrecision() const {
    return precision;
}

const std::string& TensorInfo::getPrecisionAsString(Precision precision) {
    return toString(precision);
}

ov::element::Type TensorInfo::getOvPrecision() const {
    return ovmsPrecisionToIE2Precision(precision);
}

const std::string& TensorInfo::getPrecisionAsString() const {
    return getPrecisionAsString(precision);
}

const std::string& TensorInfo::getStringFromLayout(const Layout& layout) {
    return layout;
}

const Layout& TensorInfo::getLayout() const {
    return layout;
}

bool TensorInfo::isInfluencedByDemultiplexer() const {
    return influencedByDemultiplexer;
}

const Shape& TensorInfo::getShape() const {
    return this->shape;
}

std::shared_ptr<const TensorInfo> TensorInfo::createCopyWithNewShape(const Shape& shape) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->shape = shape;
    copy->layout = Layout::getUnspecifiedLayout();
    copy->createProcessingHints();
    return copy;
}

std::shared_ptr<const TensorInfo> TensorInfo::createCopyWithNewMappedName(const std::string& mappedName) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->mapping = mappedName;
    copy->createProcessingHints();
    return copy;
}

std::shared_ptr<const TensorInfo> TensorInfo::createCopyWithDemultiplexerDimensionPrefix(const Dimension& dim) const {
    auto copy = std::make_shared<TensorInfo>(*this);
    copy->influencedByDemultiplexer = true;
    copy->shape.emplace(copy->shape.begin(), dim);
    copy->layout = this->getLayout();
    auto batchPosition = copy->layout.find(Layout::BATCH_DIMENSION_LETTER);
    if (batchPosition != std::string::npos) {
        copy->layout.replace(batchPosition, 1, std::string(1, Layout::UNDEFINED_DIMENSION_CHAR));
    }
    copy->layout = std::string(1, Layout::BATCH_DIMENSION_LETTER[0]) + copy->layout;
    copy->createProcessingHints();
    return copy;
}

std::shared_ptr<const TensorInfo> TensorInfo::createIntersection(const TensorInfo& other) const {
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

std::shared_ptr<const TensorInfo> TensorInfo::getUnspecifiedTensorInfo() {
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
