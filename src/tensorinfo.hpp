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
#include <vector>

#include <inference_engine.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "precision.hpp"
#include "shapeinfo.hpp"

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

    Precision precision_2;

    /**
         * @brief Model input shape
         */
    shape_t shape;
    shape_t shape_2;

    /**
        * @brief Model input effective shape
        */
    shape_t effectiveShape;

    /**
         * @brief Tensor layout
         */
    InferenceEngine::Layout layout;

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
        const InferenceEngine::Precision& precision,
        const shape_t& shape);
    TensorInfo(const std::string& name,
        const Precision& precision,
        const shape_t& shape);

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         * @param layout 
         */
    TensorInfo(const std::string& name,
        const InferenceEngine::Precision& precision,
        const shape_t& shape,
        const InferenceEngine::Layout& layout);
    TensorInfo(const std::string& name,
        const Precision& precision,
        const shape_t& shape,
        const InferenceEngine::Layout& layout);

    TensorInfo(const std::string& name,
        const InferenceEngine::TensorDesc& tensorDesc);

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
        const InferenceEngine::Precision& precision,
        const shape_t& shape,
        const InferenceEngine::Layout& layout);
    TensorInfo(const std::string& name,
        const std::string& mapping,
        const Precision& precision,
        const shape_t& shape,
        const InferenceEngine::Layout& layout);

    /**
         * @brief Get the Name object
         * 
         * @return const std::string& 
         */
    const std::string& getName() const;

    /**
         * @brief Get the tensor name - as in network model or mapped name
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
    const InferenceEngine::Precision getPrecision() const;
    const Precision getPrecision_2() const;

    /**
         * @brief Set the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    void setPrecision(const InferenceEngine::Precision& requestedPrecision);

    /**
         * @brief Set the Layout object
         * 
         * @return const InferenceEngine::Layout
         */
    void setLayout(InferenceEngine::Layout layout);

    /**
         * @brief Get the Precision As DataType object
         * 
         * @return const tensorflow::DataType
         */
    tensorflow::DataType getPrecisionAsDataType() const;

    static tensorflow::DataType getPrecisionAsDataType(InferenceEngine::Precision precision);
    static tensorflow::DataType getPrecisionAsDataType(Precision precision);
    static ov::element::Type getPrecisionFromDataType(tensorflow::DataType dataType);

    /**
        * @brief Get the Precision As String object
        *
        * @return const std::string
        */
    std::string getPrecisionAsString() const;

    static std::string getPrecisionAsString(const InferenceEngine::Precision precision);
    static std::string getPrecisionAsString(Precision precision);

    static const std::string getDataTypeAsString(tensorflow::DataType dataType);

    /**
         * @brief Get the InferenceEngine Layout From String 
         * 
         * @param layout 
         * @return InferenceEngine::Layout 
         */
    static InferenceEngine::Layout getLayoutFromString(const std::string& layout);

    /**
         * @brief Get the layout name from InferenceEngine Layout
         *
         * @param InferenceEngine::Layout
         * @return std::string
         */
    static std::string getStringFromLayout(const InferenceEngine::Layout layout);

    /**
         * @brief Get the Layout enum
         *
         * @return const InferenceEngine::Layout
         */
    const InferenceEngine::Layout& getLayout() const;

    /**
         * @brief Gets input shape
         *
         * @return shape
         */
    const shape_t& getShape() const;
    const shape_t& getShape_2() const;

    /**
         * @brief Gets input effective shape
         *
         * @return shape
         */
    const shape_t& getEffectiveShape() const;

    void setShape(const shape_t& shape);

    bool isInfluencedByDemultiplexer() const;

    std::shared_ptr<TensorInfo> createCopyWithNewShape(const shape_t& shape) const;

    std::shared_ptr<TensorInfo> createCopyWithEffectiveDimensionPrefix(size_t dim) const;

    /**
         * @brief Get the Tensor Desc object
         *
         * @return const InferenceEngine::TensorDesc&
         */
    const InferenceEngine::TensorDesc getTensorDesc() const;

    bool isTensorUnspecified() const;

    bool isTensorSpecEqual(const TensorInfo& other) const;

    static std::string shapeToString(const shape_t& shape);

    static std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape);

    static std::shared_ptr<TensorInfo> getUnspecifiedTensorInfo();

    static std::string tensorDescToString(const InferenceEngine::TensorDesc& desc);

    const size_t getBatchSize() const;

private:
    void updateEffectiveShape();
};
}  // namespace ovms
