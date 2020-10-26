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

#include <inference_engine.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wno-implicit-function-declaration"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

#include "modelconfig.hpp"

namespace ovms {

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

    /**
         * @brief Tensor precision data type
         */
    InferenceEngine::Precision precision;

    /**
         * @brief Model input
         */
    shape_t shape;

    /**
         * @brief Tensor layout
         */
    InferenceEngine::Layout layout;

    /**
         * @brief TensorDesc
         */
    InferenceEngine::TensorDesc tensorDesc;

public:
    /**
         * @brief Construct a new Tensor Info object
         * 
         */
    TensorInfo() = default;

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         */
    TensorInfo(const std::string& name,
        const InferenceEngine::Precision& precision,
        const shape_t& shape) :
        name(name),
        mapping(""),
        precision(precision),
        shape(shape) {}

    /**
         * @brief Construct a new Tensor Info object
         * 
         * @param name 
         * @param precision 
         * @param shape
         * @param layout 
         * @param tensorDesc 
         */
    TensorInfo(const std::string& name,
        const InferenceEngine::Precision& precision,
        const shape_t& shape,
        const InferenceEngine::Layout& layout) :
        name(name),
        mapping(""),
        precision(precision),
        shape(shape),
        layout(layout) {}

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
        const InferenceEngine::Layout& layout) :
        name(name),
        mapping(mapping),
        precision(precision),
        shape(shape),
        layout(layout) {}

    /**
         * @brief Get the Name object
         * 
         * @return const std::string& 
         */
    const std::string& getName() const {
        return name;
    }

    /**
         * @brief Get the tensor name - as in network model or mapped name
         * 
         * @return const std::string& 
         */
    const std::string& getMappedName() const {
        return mapping.size() == 0 ? name : mapping;
    }

    /**
         * @brief Get the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    const InferenceEngine::Precision getPrecision() const {
        return precision;
    }

    /**
         * @brief Set the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
    void setPrecision(const InferenceEngine::Precision& requestedPrecision) {
        precision = requestedPrecision;
    }

    /**
         * @brief Get the Precision As DataType object
         * 
         * @return const tensorflow::DataType
         */
    const tensorflow::DataType getPrecisionAsDataType() const {
        return getPrecisionAsDataType(precision);
    }

    static const tensorflow::DataType getPrecisionAsDataType(InferenceEngine::Precision precision) {
        switch (precision) {
        case InferenceEngine::Precision::FP32:
            return tensorflow::DataType::DT_FLOAT;
        case InferenceEngine::Precision::FP16:
            return tensorflow::DataType::DT_HALF;
        // case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
        case InferenceEngine::Precision::I16:
            return tensorflow::DataType::DT_INT16;
        case InferenceEngine::Precision::U8:
            return tensorflow::DataType::DT_UINT8;
        case InferenceEngine::Precision::I8:
            return tensorflow::DataType::DT_INT8;
        case InferenceEngine::Precision::U16:
            return tensorflow::DataType::DT_UINT16;
        case InferenceEngine::Precision::I32:
            return tensorflow::DataType::DT_INT32;
        case InferenceEngine::Precision::U64:
            return tensorflow::DataType::DT_UINT64;
        case InferenceEngine::Precision::I64:
            return tensorflow::DataType::DT_INT64;
        // case InferenceEngine::Precision::BIN:   return tensorflow::DataType::
        case InferenceEngine::Precision::BOOL:
            return tensorflow::DataType::DT_BOOL;
        default:
            return tensorflow::DataType::DT_INVALID;
        }
    }

    /**
        * @brief Get the Precision As String object
        *
        * @return const std::string
        */
    const std::string getPrecisionAsString() const {
        return getPrecisionAsString(precision);
    }

    static const std::string getPrecisionAsString(InferenceEngine::Precision precision) {
        switch (precision) {
        case InferenceEngine::Precision::FP32:
            return "FP32";
        case InferenceEngine::Precision::FP16:
            return "FP16";
            // case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
        case InferenceEngine::Precision::I16:
            return "I16";
        case InferenceEngine::Precision::U8:
            return "U8";
        case InferenceEngine::Precision::I8:
            return "I8";
        case InferenceEngine::Precision::U16:
            return "U16";
        case InferenceEngine::Precision::I32:
            return "I32";
        case InferenceEngine::Precision::I64:
            return "I64";
        case InferenceEngine::Precision::BOOL:
            return "BOOL";
        default:
            return "DT_INVALID";
        }
    }

    static const std::string getDataTypeAsString(tensorflow::DataType dataType) {
        switch (dataType) {
        case tensorflow::DataType::DT_FLOAT:
            return "FP32";
        case tensorflow::DataType::DT_HALF:
            return "FP16";
        case tensorflow::DataType::DT_INT16:
            return "I16";
        case tensorflow::DataType::DT_UINT8:
            return "U8";
        case tensorflow::DataType::DT_INT8:
            return "I8";
        case tensorflow::DataType::DT_UINT16:
            return "U16";
        case tensorflow::DataType::DT_INT32:
            return "I32";
        case tensorflow::DataType::DT_UINT64:
            return "U64";
        case tensorflow::DataType::DT_INT64:
            return "I64";
        case tensorflow::DataType::DT_BOOL:
            return "BOOL";
        default:
            return "DT_INVALID";
        }
    }

    /**
         * @brief Get the InferenceEngine Layout From String 
         * 
         * @param layout 
         * @return InferenceEngine::Layout 
         */
    static InferenceEngine::Layout getLayoutFromString(const std::string& layout) {
        if (layout == "ANY")
            return InferenceEngine::Layout::ANY;
        if (layout == "NCHW")
            return InferenceEngine::Layout::NCHW;
        if (layout == "NHWC")
            return InferenceEngine::Layout::NHWC;
        if (layout == "NCDHW")
            return InferenceEngine::Layout::NCDHW;
        if (layout == "NDHWC")
            return InferenceEngine::Layout::NDHWC;
        if (layout == "OIHW")
            return InferenceEngine::Layout::OIHW;
        if (layout == "GOIHW")
            return InferenceEngine::Layout::GOIHW;
        if (layout == "OIDHW")
            return InferenceEngine::Layout::OIDHW;
        if (layout == "GOIDHW")
            return InferenceEngine::Layout::GOIDHW;
        if (layout == "SCALAR")
            return InferenceEngine::Layout::SCALAR;
        if (layout == "C")
            return InferenceEngine::Layout::C;
        if (layout == "CHW")
            return InferenceEngine::Layout::CHW;
        if (layout == "HW")
            return InferenceEngine::Layout::HW;
        if (layout == "NC")
            return InferenceEngine::Layout::NC;
        if (layout == "CN")
            return InferenceEngine::Layout::CN;
        if (layout == "BLOCKED")
            return InferenceEngine::Layout::BLOCKED;

        return InferenceEngine::Layout::ANY;
    }

    /**
         * @brief Get the layout name from InferenceEngine Layout
         *
         * @param InferenceEngine::Layout
         * @return std::string
         */
    static std::string getStringFromLayout(InferenceEngine::Layout layout) {
        switch (layout) {
        case InferenceEngine::Layout::ANY:
            return "ANY";
        case InferenceEngine::Layout::NCHW:
            return "NCHW";
        case InferenceEngine::Layout::NHWC:
            return "NHWC";
        case InferenceEngine::Layout::NCDHW:
            return "NCDHW";
        case InferenceEngine::Layout::NDHWC:
            return "NDHWC";
        case InferenceEngine::Layout::OIHW:
            return "OIHW";
        case InferenceEngine::Layout::GOIHW:
            return "GOIHW";
        case InferenceEngine::Layout::OIDHW:
            return "OIDHW";
        case InferenceEngine::Layout::GOIDHW:
            return "GOIDHW";
        case InferenceEngine::Layout::SCALAR:
            return "SCALAR";
        case InferenceEngine::Layout::C:
            return "C";
        case InferenceEngine::Layout::CHW:
            return "CHW";
        case InferenceEngine::Layout::HW:
            return "HW";
        case InferenceEngine::Layout::NC:
            return "NC";
        case InferenceEngine::Layout::CN:
            return "CN";
        case InferenceEngine::Layout::BLOCKED:
            return "BLOCKED";
        }
        return "";
    }

    /**
         * @brief Get the Layout enum
         * 
         * @return const InferenceEngine::Layout
         */
    const InferenceEngine::Layout& getLayout() const {
        return layout;
    }

    /**
         * @brief Gets input shape
         *
         * @return shape
         */
    const shape_t& getShape() const {
        return shape;
    }

    /**
         * @brief Get the Tensor Desc object
         * 
         * @return const InferenceEngine::TensorDesc& 
         */
    const InferenceEngine::TensorDesc getTensorDesc() const {
        return InferenceEngine::TensorDesc{precision, shape, layout};
    }

    static std::string shapeToString(const shape_t& shape) {
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

    static std::string tensorShapeToString(const tensorflow::TensorShapeProto& tensorShape) {
        std::ostringstream oss;
        oss << "(";
        size_t i = 0;
        if (tensorShape.dim_size() > 0) {
            for (; i < tensorShape.dim_size() - 1; i++) {
                oss << tensorShape.dim(i).size() << ",";
            }
            oss << tensorShape.dim(i).size();
        }
        oss << ")";

        return oss.str();
    }
};

using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;
}  // namespace ovms
