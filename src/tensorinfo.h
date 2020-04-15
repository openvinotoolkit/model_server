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

#include <inference_engine.hpp>
#include "tensorflow/core/framework/tensor.h"

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
         * @brief Tensor precision data type
         */
        InferenceEngine::Precision precision;

        /**
         * @brief Model input
         */
        std::vector<size_t> shape;

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
         * @param layout 
         * @param tensorDesc 
         */
        TensorInfo( const std::string& name,
                    const InferenceEngine::Precision& precision,
                    const std::vector<size_t>& shape,
                    const InferenceEngine::Layout& layout,
                    const InferenceEngine::TensorDesc& tensorDesc) :
            name(name),
            precision(precision),
            shape(shape),
            layout(layout),
            tensorDesc(tensorDesc) {}

        /**
         * @brief Get the Name object
         * 
         * @return const std::string& 
         */
        const std::string& getName() {
            return name;
        }

        /**
         * @brief Get the Precision object
         * 
         * @return const InferenceEngine::Precision
         */
        const InferenceEngine::Precision getPrecision() {
            return precision;
        }

        /**
         * @brief Get the Precision As DataType object
         * 
         * @return const tensorflow::DataType
         */
        const tensorflow::DataType getPrecisionAsDataType() {
            switch (precision)
            {
                case InferenceEngine::Precision::FP32:  return tensorflow::DataType::DT_FLOAT;
                case InferenceEngine::Precision::FP16:  return tensorflow::DataType::DT_HALF;
                //case InferenceEngine::Precision::Q78:   return tensorflow::DataType::
                case InferenceEngine::Precision::I16:   return tensorflow::DataType::DT_INT16;
                case InferenceEngine::Precision::U8:    return tensorflow::DataType::DT_UINT8;
                case InferenceEngine::Precision::U16:   return tensorflow::DataType::DT_UINT16;
                case InferenceEngine::Precision::I32:   return tensorflow::DataType::DT_INT32;
                case InferenceEngine::Precision::I64:   return tensorflow::DataType::DT_INT64;
                //case InferenceEngine::Precision::BIN:   return tensorflow::DataType::
                case InferenceEngine::Precision::BOOL:  return tensorflow::DataType::DT_BOOL;
                default:                                return tensorflow::DataType::DT_INVALID;
            }
        }

        /**
         * @brief Get the InferenceEngine Layout From String 
         * 
         * @param layout 
         * @return InferenceEngine::Layout 
         */
        static InferenceEngine::Layout getLayoutFromString(const std::string& layout) {
            if (layout == "ANY")     return InferenceEngine::Layout::ANY;
            if (layout == "NCHW")    return InferenceEngine::Layout::NCHW;
            if (layout == "NHWC")    return InferenceEngine::Layout::NHWC;
            if (layout == "NCDHW")   return InferenceEngine::Layout::NCDHW;
            if (layout == "NDHWC")   return InferenceEngine::Layout::NDHWC;
            if (layout == "OIHW")    return InferenceEngine::Layout::OIHW;
            if (layout == "GOIHW")   return InferenceEngine::Layout::GOIHW;
            if (layout == "OIDHW")   return InferenceEngine::Layout::OIDHW;
            if (layout == "GOIDHW")  return InferenceEngine::Layout::GOIDHW;
            if (layout == "SCALAR")  return InferenceEngine::Layout::SCALAR;
            if (layout == "C")       return InferenceEngine::Layout::C;
            if (layout == "CHW")     return InferenceEngine::Layout::CHW;
            if (layout == "HW")      return InferenceEngine::Layout::HW;
            if (layout == "NC")      return InferenceEngine::Layout::NC;
            if (layout == "CN")      return InferenceEngine::Layout::CN;
            if (layout == "BLOCKED") return InferenceEngine::Layout::BLOCKED;

            return InferenceEngine::Layout::ANY;
        }

        /**
         * @brief Get the Layout enum
         * 
         * @return const InferenceEngine::Layout
         */
        const InferenceEngine::Layout& getLayout() {
            return layout;
        }

        /**
         * @brief Gets input shape
         *
         * @return shape
         */
        const std::vector<size_t>& getShape() {
            return shape;
        }

        /**
         * @brief Get the Tensor Desc object
         * 
         * @return const InferenceEngine::TensorDesc& 
         */
        const InferenceEngine::TensorDesc& getTensorDesc() {
            return tensorDesc;
        }
    };
}  // namespace ovms
