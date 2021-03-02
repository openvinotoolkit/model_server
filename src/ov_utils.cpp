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
#include "ov_utils.hpp"

#include <memory>

#include <inference_engine.hpp>
#include <spdlog/spdlog.h>

namespace ovms {

Status createSharedBlob(InferenceEngine::Blob::Ptr& destinationBlob, InferenceEngine::TensorDesc tensorDesc) {
    try {
        switch (tensorDesc.getPrecision()) {
        case InferenceEngine::Precision::FP32:
            destinationBlob = InferenceEngine::make_shared_blob<float>(tensorDesc);
            break;
        case InferenceEngine::Precision::FP16:
            destinationBlob = InferenceEngine::make_shared_blob<uint16_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::U8:
            destinationBlob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::I8:
            destinationBlob = InferenceEngine::make_shared_blob<int8_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::I16:
            destinationBlob = InferenceEngine::make_shared_blob<int16_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::U16:
            destinationBlob = InferenceEngine::make_shared_blob<uint16_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::I32:
            destinationBlob = InferenceEngine::make_shared_blob<int32_t>(tensorDesc);
            break;
        case InferenceEngine::Precision::I64:
        case InferenceEngine::Precision::MIXED:
        case InferenceEngine::Precision::Q78:
        case InferenceEngine::Precision::BIN:
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::CUSTOM:
        default: {
            SPDLOG_ERROR("Blob clone failed, unsupported precision");
            return StatusCode::INVALID_PRECISION;
        }
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        SPDLOG_DEBUG("Blob clone failed; exception message: {}", e.what());
        return StatusCode::OV_CLONE_BLOB_ERROR;
    } catch (std::logic_error& e) {
        SPDLOG_DEBUG("Blob clone failed; exception message: {}", e.what());
        return StatusCode::OV_CLONE_BLOB_ERROR;
    }
    destinationBlob->allocate();
    return StatusCode::OK;
}
}  // namespace ovms
