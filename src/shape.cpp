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
#include <sstream>

#include "logging.hpp"
#include "stringutils.hpp"

namespace ovms {

Dimension::Dimension() :
    Dimension(DYNAMIC_DIMENSION) {
}

Dimension::Dimension(dimension_value_t dim) :
    Dimension(dim, dim) {
}

Dimension::Dimension(dimension_value_t minimum, dimension_value_t maximum) {
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

dimension_value_t Dimension::getAnyValue() const {
    return this->maximum;
}

dimension_value_t Dimension::getStaticValue() const {
    if (this->isDynamic())
        throw std::invalid_argument("getStaticValue but dimension dynamic");
    return this->maximum;
}

dimension_value_t Dimension::getMinValue() const {
    if (this->isStatic())
        throw std::invalid_argument("getMinValue but dimension static");
    return this->minimum;
}

dimension_value_t Dimension::getMaxValue() const {
    if (this->isStatic())
        throw std::invalid_argument("getMaxValue but dimension static");
    return this->maximum;
}

bool Dimension::operator==(const Dimension& rhs) const {
    return this->minimum == rhs.minimum && this->maximum == rhs.maximum;
}

bool Dimension::operator!=(const Dimension& rhs) const {
    return !(this->operator==(rhs));
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

Shape::Shape() {
}

Shape::Shape(std::initializer_list<Dimension> list) {
    this->dimensions.reserve(list.size());
    for (const Dimension& dim : list) {
        this->dimensions.emplace_back(dim);
    }
}

Status Shape::fromFlatShape(const shape_t& shapeIn, Shape& shapeOut) {
    Shape shape;
    for (size_t dim : shapeIn) {
        if (dim > std::numeric_limits<dimension_value_t>::max()) {
            return StatusCode::CANNOT_CONVERT_FLAT_SHAPE;
        } else {
            shape.add(Dimension{static_cast<dimension_value_t>(dim)});
        }
    }
    shapeOut = shape;
    return StatusCode::OK;
}

Shape::Shape(const ov::PartialShape& shape) {
    this->dimensions.reserve(shape.size());
    for (const ov::Dimension& dim : shape) {
        if (dim.is_static()) {
            this->dimensions.emplace_back(Dimension{dim.get_length()});
        } else if (!dim.get_interval().has_upper_bound()) {
            this->dimensions.emplace_back(Dimension::any());
        } else {
            this->dimensions.emplace_back(Dimension{dim.get_min_length(), dim.get_max_length()});
        }
    }
}

Shape& Shape::add(const Dimension& dim) {
    this->dimensions.emplace_back(dim);
    return *this;
}

size_t Shape::getSize() const {
    return this->dimensions.size();
}

ov::PartialShape Shape::createPartialShape() const {
    ov::PartialShape shape;

    shape.reserve(this->dimensions.size());
    for (const Dimension& dim : this->dimensions) {
        if (dim.isStatic()) {
            shape.push_back(ov::Dimension(dim.getStaticValue()));
        } else {
            shape.push_back(ov::Dimension{dim.getMinValue(), dim.getMaxValue()});
        }
    }

    return shape;
}

shape_t Shape::getFlatShape() const {
    shape_t shape;
    shape.reserve(this->dimensions.size());

    for (const Dimension& dim : this->dimensions) {
        if (dim.getAnyValue() <= 0)
            shape.emplace_back(0);
        else
            shape.emplace_back(dim.getAnyValue());
    }

    return shape;
}

bool Shape::operator==(const Shape& rhs) const {
    if (this->dimensions.size() != rhs.dimensions.size())
        return false;

    for (size_t i = 0; i < this->dimensions.size(); i++) {
        if (this->dimensions[i] != rhs.dimensions[i]) {
            return false;
        }
    }
    return true;
}

bool Shape::operator!=(const Shape& rhs) const {
    return !(this->operator==(rhs));
}

std::string Shape::toString() const {
    std::stringstream ss;

    ss << "(";

    size_t dimensionCount = this->dimensions.size();
    if (dimensionCount > 0) {
        for (size_t i = 0; i < dimensionCount - 1; i++) {
            ss << this->dimensions[i].toString() << ",";
        }
        ss << this->dimensions[dimensionCount - 1].toString();
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

        count = std::count(token.begin(), token.end(), ':');
        if (count > 1) {
            SPDLOG_ERROR("Parsing model shape string: {}; too many ':' characters", token);
            return StatusCode::SHAPE_WRONG_FORMAT;
        }
        try {
            if (count == 0) {
                int dimValue = std::stoi(token);
                if (dimValue == DYNAMIC_DIMENSION || dimValue > 0) {
                    shape.add(Dimension(dimValue));
                } else {
                    SPDLOG_ERROR("Parsing model shape string: {}; must be {} (any) or higher than 0", token, DYNAMIC_DIMENSION);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
            } else {
                std::vector<std::string> subTokens = tokenize(token, ':');
                if (subTokens.size() != 2 || subTokens[0].empty() || subTokens[1].empty()) {
                    SPDLOG_ERROR("Parsing model shape string: {}; range must have min and max", strIn);
                    return StatusCode::SHAPE_WRONG_FORMAT;
                }
                int dimMin = std::stoi(subTokens[0]);
                int dimMax = std::stoi(subTokens[1]);
                if (dimMin <= 0 || dimMax <= 0) {
                    SPDLOG_ERROR("Parsing model shape string: {}; range must be higher than 0", token);
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

    shapeOut = shape;
    return StatusCode::OK;
}

ShapeInfo_2::operator std::string() const {
    std::stringstream ss;
    ss << this->shape.toString() << " (" << (this->shapeMode == Mode::FIXED ? "fixed" : "auto") << ")";
    return ss.str();
}

}  // namespace ovms
