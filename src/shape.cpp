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

#include "shape.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

#include "logging.hpp"
#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

Dimension::Dimension() :
    Dimension(DYNAMIC_DIMENSION) {
}

Dimension::Dimension(const ov::Dimension& dim) {
    if (dim.is_static()) {
        this->minimum = dim.get_length();
        this->maximum = dim.get_length();
    } else if (!dim.get_interval().has_upper_bound()) {
        this->minimum = DYNAMIC_DIMENSION;
        this->maximum = DYNAMIC_DIMENSION;
    } else {
        this->minimum = dim.get_min_length();
        this->maximum = dim.get_max_length();
    }
}

Dimension::Dimension(dimension_value_t dim) :
    Dimension(dim, dim) {
}

Dimension::Dimension(dimension_value_t minimum, dimension_value_t maximum) {
    if (minimum == -1 && maximum != -1) {
        throw std::invalid_argument("Invalid range");
    }
    if (minimum < -1 || maximum < -1) {
        throw std::invalid_argument("Range must not be lower than -1");
    }
    if (minimum > maximum) {
        throw std::invalid_argument("Range maximum must be higher or equal to minimum");
    }

    this->minimum = minimum;
    this->maximum = maximum;
}

bool Dimension::isDynamic() const {
    if (this->minimum != this->maximum)
        return true;
    if (this->minimum == DYNAMIC_DIMENSION)
        return true;
    return false;
}

bool Dimension::isStatic() const {
    return !this->isDynamic();
}

ov::Dimension Dimension::createPartialDimension() const {
    if (this->isStatic()) {
        OV_LOGGER("ov::Dimension({})", this->getStaticValue());
        return ov::Dimension(this->getStaticValue());
    }
    if (this->minimum == DYNAMIC_DIMENSION) {
        OV_LOGGER("ov::Dimension::dynamic()");
        return ov::Dimension::dynamic();
    }
    OV_LOGGER("ov::Dimension({},{})", this->minimum, this->maximum);
    return ov::Dimension(this->minimum, this->maximum);
}

dimension_value_t Dimension::getStaticValue() const {
    if (this->isDynamic())
        throw std::invalid_argument("getStaticValue but dimension dynamic");
    return this->maximum;
}

dimension_value_t Dimension::getMinValue() const {
    if (this->isStatic())
        throw std::invalid_argument("getMinValue but dimension static");
    if (this->isAny())
        throw std::invalid_argument("getMinValue but dimension any");
    return this->minimum;
}

dimension_value_t Dimension::getMaxValue() const {
    if (this->isStatic())
        throw std::invalid_argument("getMaxValue but dimension static");
    if (this->isAny())
        throw std::invalid_argument("getMinValue but dimension any");
    return this->maximum;
}

bool Dimension::operator==(const Dimension& rhs) const {
    return this->minimum == rhs.minimum && this->maximum == rhs.maximum;
}

bool Dimension::operator!=(const Dimension& rhs) const {
    return !(this->operator==(rhs));
}

Status Dimension::fromString(const std::string& str, Dimension& dimOut) {
    Dimension dim;

    std::string strCopy = str;
    erase_spaces(strCopy);
    if (strCopy.find(DIMENSION_RANGE_DELIMETER) != std::string::npos) {
        // Range
        if (strCopy.find_first_not_of("0123456789:") != std::string::npos) {
            SPDLOG_ERROR("Parsing dimension string not a range: {}", strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        }
        size_t delimCount = std::count(strCopy.begin(), strCopy.end(), DIMENSION_RANGE_DELIMETER);
        if (delimCount != 1) {
            SPDLOG_ERROR("Parsing dimension string, wrong amount of '{}' - {}; {}", DIMENSION_RANGE_DELIMETER, delimCount, strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        } else {
            std::vector<std::string> tokens = tokenize(strCopy, DIMENSION_RANGE_DELIMETER);
            if (tokens.size() == 2) {
                try {
                    int dimNumberMin = std::stoi(tokens[0]);
                    int dimNumberMax = std::stoi(tokens[1]);
                    if (dimNumberMin > 0 && dimNumberMax > 0) {
                        dim = Dimension(dimNumberMin, dimNumberMax);
                    } else if (dimNumberMin >= dimNumberMax) {
                        SPDLOG_ERROR("Parsing dimension string range max must be higher than min: {}", strCopy);
                        return StatusCode::DIM_WRONG_FORMAT;
                    } else {
                        SPDLOG_ERROR("Parsing dimension string range must be lager than 0: {}", strCopy);
                        return StatusCode::DIM_WRONG_FORMAT;
                    }
                } catch (const std::out_of_range& e) {
                    SPDLOG_ERROR("Parsing dimension string out of range: {}, error: {}", strCopy, e.what());
                    return StatusCode::DIM_WRONG_FORMAT;
                } catch (...) {
                    SPDLOG_ERROR("Parsing dimension string: {}", strCopy);
                    return StatusCode::DIM_WRONG_FORMAT;
                }
            } else {
                SPDLOG_ERROR("Parsing dimension string, not a number between '{}' - {}", DIMENSION_RANGE_DELIMETER, strCopy);
                return StatusCode::DIM_WRONG_FORMAT;
            }
        }
    } else {
        size_t count = std::count(strCopy.begin(), strCopy.end(), '-');
        if (count > 1) {
            SPDLOG_ERROR("Parsing dimension string: {}; too many '-' characters", strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        } else if (count == 1 && *strCopy.begin() != '-') {
            SPDLOG_ERROR("Parsing dimension string: {}; invalid '-' position", strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        }
        // Number
        if (strCopy.find_first_not_of("0123456789-") != std::string::npos) {
            SPDLOG_ERROR("Parsing dimension string not a number: {}", strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        }
        try {
            int dimNumber = std::stoi(strCopy);
            if (dimNumber == DYNAMIC_DIMENSION) {
                dim = Dimension::any();
            } else if (dimNumber >= 0) {
                dim = Dimension(dimNumber);
            } else {
                SPDLOG_ERROR("Parsing dimension string out of range: {}", strCopy);
                return StatusCode::DIM_WRONG_FORMAT;
            }
        } catch (const std::out_of_range& e) {
            SPDLOG_ERROR("Parsing dimension string out of range: {}, error: {}", strCopy, e.what());
            return StatusCode::DIM_WRONG_FORMAT;
        } catch (...) {
            SPDLOG_ERROR("Parsing dimension string: {}", strCopy);
            return StatusCode::DIM_WRONG_FORMAT;
        }
    }

    dimOut = dim;
    return StatusCode::OK;
}

bool Dimension::isAny() const {
    return (this->maximum == DYNAMIC_DIMENSION) &&
           (this->minimum == DYNAMIC_DIMENSION);
}

bool Dimension::partiallyFitsInto(const Dimension& next) const {
    if (next.isAny() || isAny()) {
        return true;
    }
    if (isStatic()) {
        return next.match(getStaticValue());
    }
    if (next.isStatic()) {
        return this->match(next.getStaticValue());
    }
    // both are dynamic
    if (next.getMinValue() > getMaxValue()) {
        return false;
    }
    if (next.getMaxValue() < getMinValue()) {
        return false;
    }
    return true;
}

bool Dimension::match(dimension_value_t value) const {
    if (value < -1)
        return false;
    if (isAny()) {
        return true;
    }
    if (isStatic()) {
        return getStaticValue() == value;
    }
    if (value < getMinValue()) {
        return false;
    }
    if (value > getMaxValue()) {
        return false;
    }
    return true;
}

std::string Dimension::toString() const {
    std::stringstream ss;

    if (this->isStatic()) {
        ss << this->minimum;
    } else {
        if (this->maximum == DYNAMIC_DIMENSION) {
            ss << DYNAMIC_DIMENSION;
        } else {
            ss << "[" << this->minimum << "~" << this->maximum << "]";
        }
    }

    return ss.str();
}

Dimension Dimension::any() {
    return Dimension();
}

dimension_value_t Dimension::getLowerBound() const {
    return isStatic() ? getStaticValue() : getMinValue();
}

dimension_value_t Dimension::getUpperBound() const {
    return isStatic() ? getStaticValue() : getMaxValue();
}

std::optional<Dimension> Dimension::createIntersection(const Dimension& other) const {
    if (*this == Dimension::any())
        return other;
    if (other == Dimension::any())
        return *this;
    auto start = std::max(this->getLowerBound(), other.getLowerBound());
    auto end = std::min(this->getUpperBound(), other.getUpperBound());
    if (end < start)
        return std::nullopt;
    return Dimension{start, end};
}

Shape::Shape() {
}

Shape::Shape(std::initializer_list<Dimension> list) :
    std::vector<Dimension>(list) {}

Shape::Shape(const shape_t& s) {
    auto status = fromFlatShape(s, *this);
    if (!status.ok()) {
        throw std::invalid_argument("Could not convert from flat shape");
    }
}

Status Shape::fromFlatShape(const shape_t& shapeIn, Shape& shapeOut) {
    Shape shape;
    for (size_t dim : shapeIn) {
        if (dim > static_cast<size_t>(std::numeric_limits<dimension_value_t>::max())) {
            return StatusCode::CANNOT_CONVERT_FLAT_SHAPE;
        } else {
            shape.add(Dimension{static_cast<dimension_value_t>(dim)});
        }
    }
    shapeOut = std::move(shape);
    return StatusCode::OK;
}

Shape::Shape(const ov::PartialShape& shape) {
    this->reserve(shape.size());
    OV_LOGGER("const ov::Dimension& dim : shape");
    for (const ov::Dimension& dim : shape) {
        OV_LOGGER("dim.is_static()");
        if (dim.is_static()) {
            OV_LOGGER("dim.get_length()");
            this->emplace_back(Dimension{dim.get_length()});
        } else if (!dim.get_interval().has_upper_bound()) {
            OV_LOGGER("dim.get_interval().has_upper_bound()");
            this->emplace_back(Dimension::any());
        } else {
            OV_LOGGER("dim.get_min_length()");
            OV_LOGGER("dim.get_max_length()");
            this->emplace_back(Dimension{dim.get_min_length(), dim.get_max_length()});
        }
    }
}

Shape& Shape::add(const Dimension& dim) {
    return this->add(dim, this->size());
}
Shape& Shape::add(const Dimension& dim, size_t pos) {
    this->emplace(this->begin() + pos, dim);
    return *this;
}

ov::PartialShape Shape::createPartialShape() const {
    OV_LOGGER("shape = ov::PartialShape()");
    ov::PartialShape shape;
    OV_LOGGER("shape.reserve({})", this->size());
    shape.reserve(this->size());
    for (const Dimension& dim : *this) {
        if (dim.isStatic()) {
            OV_LOGGER("shape.push_back(ov::Dimension({}))", dim.getStaticValue());
            shape.push_back(ov::Dimension(dim.getStaticValue()));
        } else if (dim.isAny()) {
            OV_LOGGER("shape.push_back(ov::Dimension::dynamic())");
            shape.push_back(ov::Dimension::dynamic());
        } else {
            OV_LOGGER("shape.push_back(ov::Dimension({}, {}))", dim.getMinValue(), dim.getMaxValue());
            shape.push_back(ov::Dimension{dim.getMinValue(), dim.getMaxValue()});
        }
    }
    return shape;
}

bool Shape::operator==(const Shape& rhs) const {
    if (this->size() != rhs.size())
        return false;

    for (size_t i = 0; i < this->size(); i++) {
        if ((*this)[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool Shape::operator!=(const Shape& rhs) const {
    return !(this->operator==(rhs));
}

bool Shape::match(const ov::Shape& ovShape) const {
    if (this->size() != ovShape.size()) {
        return false;
    }
    for (size_t i = 0; i < this->size(); i++) {
        if (!(*this)[i].match(ovShape[i])) {
            return false;
        }
    }
    return true;
}

bool Shape::match(const ov::Shape& ovShape, const size_t skipPosition) const {
    if (this->size() != ovShape.size()) {
        return false;
    }
    for (size_t i = 0; i < skipPosition; i++) {
        if (!(*this)[i].match(ovShape[i])) {
            return false;
        }
    }
    for (size_t i = skipPosition + 1; i < this->size(); i++) {
        if (!(*this)[i].match(ovShape[i])) {
            return false;
        }
    }
    return true;
}

std::optional<Shape> Shape::createIntersection(const Shape& other) const {
    if (this->size() != other.size())
        return std::nullopt;
    Shape intersected;
    intersected.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        auto intersectedDim = (*this)[i].createIntersection(other[i]);
        if (!intersectedDim.has_value()) {
            return std::nullopt;
        }
        intersected.emplace_back(std::move(intersectedDim.value()));
    }
    return intersected;
}

std::string Shape::toString() const {
    std::stringstream ss;

    ss << "(";

    size_t dimensionCount = this->size();
    if (dimensionCount > 0) {
        for (size_t i = 0; i < dimensionCount - 1; i++) {
            ss << (*this)[i].toString() << ",";
        }
        ss << (*this)[dimensionCount - 1].toString();
    }

    ss << ")";

    return ss.str();
}

Status Shape::fromString(const std::string& strIn, Shape& shapeOut) {
    Shape shape;
    std::string str = strIn;

    erase_spaces(str);
    if (str.find_first_not_of("0123456789(),-:") != std::string::npos)
        return StatusCode::SHAPE_WRONG_FORMAT;

    if (std::count(str.begin(), str.end(), '(') != 1)
        return StatusCode::SHAPE_WRONG_FORMAT;

    if (std::count(str.begin(), str.end(), ')') != 1)
        return StatusCode::SHAPE_WRONG_FORMAT;

    if (str.size() <= 2)
        return StatusCode::SHAPE_WRONG_FORMAT;

    if (str.front() != '(' || str.back() != ')')
        return StatusCode::SHAPE_WRONG_FORMAT;

    str.pop_back();
    str.erase(str.begin());

    std::vector<std::string> tokens = tokenize(str, ',');

    for (const std::string& token : tokens) {
        size_t count = std::count(token.begin(), token.end(), '-');
        if (count > 1) {
            SPDLOG_ERROR("Parsing model shape string: {}; too many '-' characters", token);
            return StatusCode::SHAPE_WRONG_FORMAT;
        } else if (count == 1 && !token.empty() && *token.begin() != '-') {
            SPDLOG_ERROR("Parsing model shape string: {}; invalid '-' position", token);
            return StatusCode::SHAPE_WRONG_FORMAT;
        }

        count = std::count(token.begin(), token.end(), DIMENSION_RANGE_DELIMETER);
        if (count > 1) {
            SPDLOG_ERROR("Parsing model shape string: {}; too many '{}' characters", DIMENSION_RANGE_DELIMETER, token);
            return StatusCode::SHAPE_WRONG_FORMAT;
        }
        try {
            if (count == 0) {
                int dimValue = std::stoi(token);
                if (dimValue == DYNAMIC_DIMENSION || dimValue >= 0) {
                    shape.add(Dimension(dimValue));
                } else {
                    SPDLOG_ERROR("Parsing model shape string: {}; must be {} (any) or >= 0", token, DYNAMIC_DIMENSION);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
            } else {
                std::vector<std::string> subTokens = tokenize(token, DIMENSION_RANGE_DELIMETER);
                if (subTokens.size() != 2 || subTokens[0].empty() || subTokens[1].empty()) {
                    SPDLOG_ERROR("Parsing model shape string: {}; range must have min and max", strIn);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
                int dimMin = std::stoi(subTokens[0]);
                int dimMax = std::stoi(subTokens[1]);
                if (dimMin < 0 || dimMax < 0) {
                    SPDLOG_ERROR("Parsing model shape string: {}; range must be higher than or equal 0", token);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
                if (dimMin >= dimMax) {
                    SPDLOG_ERROR("Parsing model shape string: {}; range max must be higher than min", token);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
                shape.add(Dimension(dimMin, dimMax));
            }
        } catch (const std::out_of_range& e) {
            SPDLOG_ERROR("Parsing model shape string out of range: {}, error: {}", str, e.what());
            return StatusCode::SHAPE_WRONG_FORMAT;
        } catch (...) {
            SPDLOG_ERROR("Parsing model shape string: {}", strIn);
            return StatusCode::SHAPE_WRONG_FORMAT;
        }
    }

    shapeOut = std::move(shape);
    return StatusCode::OK;
}

ShapeInfo::operator std::string() const {
    std::stringstream ss;
    ss << this->shape.toString() << " (" << (this->shapeMode == Mode::FIXED ? "fixed" : "auto") << ")";
    return ss.str();
}

}  // namespace ovms
