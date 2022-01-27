//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "layout.hpp"
#include "precision.hpp"
#include "shape.hpp"

namespace ovms {

class TensorInfo;

using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;

/**
     * @brief Class containing information about the tensor
     */
class TensorInfo {
protected:
    /**
         * @brief Input name
         */
    std::string name;

    /**
         * @brief Mapping name
         */
    std::string mapping;

    Precision precision;

    /**
         * @brief Model input shape
         */
    Shape shape;

    /**
         * @brief Tensor layout
         */
    Layout layout;

    /**
         * @brief Information if influenced by demultiplexer
         */
    bool influencedByDemultiplexer = false;

public:
    /**
         * @brief Construct a new Tensor Info object
         * 
         */
    TensorInfo() = default;
    TensorInfo(const TensorInfo&) = default;

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         */
    TensorInfo(const std::string& name,
        const Precision& precision,
        const shape_t& shape);
    TensorInfo(const std::string& name,
        const Precision& precision,
        const Shape& shape);

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         * @param layout 
         */
    TensorInfo(const std::string& name,
        const Precision& precision,
        const shape_t& shape,
        const Layout& layout);
    TensorInfo(const std::string& name,
        const Precision& precision,
        const Shape& shape,
        const Layout& layout);

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         * @param layout 
         */
    TensorInfo(const std::string& name,
        const std::string& mapping,
        const Precision& precision,
        const shape_t& shape,
        const Layout& layout);
    TensorInfo(const std::string& name,
        const std::string& mapping,
        const Precision& precision,
        const Shape& shape,
        const Layout& layout);
    TensorInfo(const std::string& name,
        const std::string& mapping,
        const Precision& precision,
        const shape_t& shape);

    /**
         * @brief Get the Name object
         * 
         * @return const std::string& 
         */
    const std::string& getName() const;

    /**
         * @brief Get the tensor name - as in model or mapped name
         * 
         * @return const std::string& 
         */
    const std::string& getMappedName() const;
    void setMappedName(const std::string& mappedName);

    /**
         * @brief Get the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    const Precision getPrecision() const;

    /**
         * @brief Set the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    void setPrecision(const ovms::Precision& requestedPrecision);

    /**
         * @brief Set the Layout object
         */
    void setLayout(const Layout& layout);

    /**
         * @brief Get the Precision As DataType object
         * 
         * @return const tensorflow::DataType
         */
    tensorflow::DataType getPrecisionAsDataType() const;

    static tensorflow::DataType getPrecisionAsDataType(Precision precision);
    ov::element::Type getOvPrecision() const;

    /**
        * @brief Get the Precision As String object
        *
        * @return const std::string
        */
    std::string getPrecisionAsString() const;

    /**
        * @brief Get the string representation of TensorInfo object
        *
        * @return String representation
        */
    std::string asString() const;

    static std::string getPrecisionAsString(Precision precision);

    static const std::string getDataTypeAsString(tensorflow::DataType dataType);

    /**
         * @brief Get the layout name from Layout
         *
         * @param Layout
         * @return std::string
         */
    static std::string getStringFromLayout(const Layout& layout);

    /**
         * @brief Get the Layout string
         *
         * @return const Layout&
         */
    const Layout& getLayout() const;

    /**
         * @brief Gets input shape
         *
         * @return shape
         */
    const Shape& getShape() const;
    void setShape(const Shape& shape);

    bool isInfluencedByDemultiplexer() const;

    std::shared_ptr<TensorInfo> createCopyWithNewShape(const Shape& shape) const;

    std::shared_ptr<TensorInfo> createCopyWithEffectiveDimensionPrefix(const Dimension& dim) const;
    std::shared_ptr<TensorInfo> createIntersection(const TensorInfo& other);

    bool isTensorUnspecified() const;

    bool isTensorSpecEqual(const TensorInfo& other) const;

    static std::string shapeToString(const shape_t& shape);

    static std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape);

    static std::shared_ptr<TensorInfo> getUnspecifiedTensorInfo();

    const std::optional<Dimension> getBatchSize() const;
};
}  // namespace ovms
