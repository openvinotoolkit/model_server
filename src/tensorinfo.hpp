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

#include "layout.hpp"
#include "precision.hpp"
#include "shape.hpp"

namespace google::protobuf {
template <typename T>
class RepeatedField;
}

namespace ovms {

class TensorInfo;

using tensor_map_t = std::map<std::string, std::shared_ptr<const TensorInfo>>;

/**
     * @brief Class containing information about the tensor
     */
class TensorInfo {
public:
    enum class ProcessingHint {
        IMAGE,
        STRING_1D_U8,
        STRING_2D_U8,
        NO_PROCESSING
    };

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

    /**
         * @brief Get the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    const Precision getPrecision() const;

    ov::element::Type getOvPrecision() const;

    /**
        * @brief Get the Precision As String object
        *
        * @return const std::string
        */
    const std::string& getPrecisionAsString() const;

    /**
        * @brief Get the string representation of TensorInfo object
        *
        * @return String representation
        */
    std::string asString() const;

    static const std::string& getPrecisionAsString(Precision precision);

    /**
         * @brief Get the layout name from Layout
         *
         * @param Layout
         * @return std::string
         */
    static const std::string& getStringFromLayout(const Layout& layout);

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

    ProcessingHint getPreProcessingHint() const;
    ProcessingHint getPostProcessingHint() const;

    bool isInfluencedByDemultiplexer() const;

    std::shared_ptr<const TensorInfo> createCopyWithNewShape(const Shape& shape) const;
    std::shared_ptr<const TensorInfo> createCopyWithNewMappedName(const std::string& mappedName) const;

    std::shared_ptr<const TensorInfo> createCopyWithDemultiplexerDimensionPrefix(const Dimension& dim) const;
    std::shared_ptr<const TensorInfo> createIntersection(const TensorInfo& other) const;

    bool isTensorUnspecified() const;

    bool isTensorSpecEqual(const TensorInfo& other) const;

    static std::shared_ptr<const TensorInfo> getUnspecifiedTensorInfo();

    const std::optional<Dimension> getBatchSize() const;

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

    void createProcessingHints();
    TensorInfo::ProcessingHint preProcessingHint = TensorInfo::ProcessingHint::NO_PROCESSING;
    TensorInfo::ProcessingHint postProcessingHint = TensorInfo::ProcessingHint::NO_PROCESSING;
};

}  // namespace ovms
