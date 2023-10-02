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
#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace ovms {
class Status;
using dimension_value_t = std::int64_t;

constexpr dimension_value_t DYNAMIC_DIMENSION = -1;

constexpr char DIMENSION_RANGE_DELIMETER = ':';

enum Mode { FIXED,
    AUTO };
using shape_t = std::vector<size_t>;
using signed_shape_t = std::vector<dimension_value_t>;

template <typename T>
std::string shapeToString(const T& shape) {
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

class Dimension {
    dimension_value_t minimum, maximum;

public:
    Dimension();

    Dimension(const ov::Dimension& dim);

    Dimension(dimension_value_t dim);

    Dimension(dimension_value_t minimum, dimension_value_t maximum);

    bool isStatic() const;
    bool isDynamic() const;

    ov::Dimension createPartialDimension() const;

    dimension_value_t getStaticValue() const;
    dimension_value_t getMinValue() const;
    dimension_value_t getMaxValue() const;

    bool operator==(const Dimension& rhs) const;
    bool operator!=(const Dimension& rhs) const;

    static Status fromString(const std::string& str, Dimension& dimOut);
    std::string toString() const;

    static Dimension any();

    bool match(dimension_value_t value) const;
    bool partiallyFitsInto(const Dimension& value) const;
    bool isAny() const;
    std::optional<Dimension> createIntersection(const Dimension& other) const;

    dimension_value_t getLowerBound() const;
    dimension_value_t getUpperBound() const;
};

class Shape : public std::vector<Dimension> {
public:
    Shape();
    Shape(const shape_t& s);
    // Create shape out of ovms::Shape{1, 5, 100, 100}
    Shape(std::initializer_list<Dimension> list);

    // Create ovms::Shape out of oridnary vector of dimensions.
    static Status fromFlatShape(const shape_t& shapeIn, Shape& shapeOut);

    // Create ovms::Shape out of ov::PartialShape.
    Shape(const ov::PartialShape& shape);

    Shape& add(const Dimension& dim, size_t pos);
    Shape& add(const Dimension& dim);

    bool isStatic() const;
    bool isDynamic() const;

    ov::PartialShape createPartialShape() const;

    bool operator==(const Shape& rhs) const;
    bool operator!=(const Shape& rhs) const;

    bool match(const ov::Shape& rhs) const;
    bool match(const ov::Shape& rhs, const size_t skipPosition) const;
    std::optional<Shape> createIntersection(const Shape& other) const;

    std::string toString() const;
    static Status fromString(const std::string& strIn, Shape& shapeOut);
};

using shapes_map_t = std::unordered_map<std::string, Shape>;

struct ShapeInfo {
    Mode shapeMode = FIXED;
    Shape shape;

    operator std::string() const;

    bool operator==(const ShapeInfo& rhs) const {
        return this->shapeMode == rhs.shapeMode && this->shape == rhs.shape;
    }

    bool operator!=(const ShapeInfo& rhs) const {
        return !(*this == rhs);
    }
};

using shapes_info_map_t = std::unordered_map<std::string, ShapeInfo>;

}  // namespace ovms
