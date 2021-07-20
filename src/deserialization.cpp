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
#include "deserialization.hpp"

namespace ovms {

template <>
Status InputSink<InferenceEngine::InferRequest&>::give(const std::string& name, InferenceEngine::Blob::Ptr blob) {
    Status status;
    try {
        requester.SetBlob(name, blob);
        // OV implementation the InferenceEngine::Exception is not
        // a base class for all other exceptions thrown from OV.
        // OV can throw exceptions derived from std::logic_error.
    } catch (const InferenceEngine::Exception& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_DEBUG("{}: {}", status.string(), e.what());
        return status;
    }

    return status;
}

std::shared_ptr<ovms::TensorInfo> getFinalShapedTensorInfo(const ovms::TensorInfo& servableInfo, const tensorflow::TensorProto& requestInput, bool isPipeline) {
    auto potentiallyDynamicShape = servableInfo.getShape();
    if (isPipeline) {
        potentiallyDynamicShape = servableInfo.getEffectiveShape();
    }
    shape_t newShape;
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); ++i) {
        size_t servableDimensionSize = potentiallyDynamicShape[i];
        size_t tensorProtoDimensionSize = requestInput.tensor_shape().dim(i).size();
        if (servableDimensionSize == 0) {
            newShape.emplace_back(tensorProtoDimensionSize);
        } else {
            newShape.emplace_back(servableDimensionSize);
        }
    }
    return servableInfo.createCopyWithNewShape(newShape);
}
}  // namespace ovms
