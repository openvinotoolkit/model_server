//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "../extractchoice.hpp"
#include "../requesttensorextractor.hpp"
#include "../statefulrequestprocessor.hpp"
#include "../logging.hpp"
#include "../profiler.hpp"
#include "../shape.hpp"
#include "../status.hpp"

namespace ovms {
class InferenceRequest;

std::optional<Dimension> getRequestBatchSize(const tensorflow::serving::PredictRequest* request, const size_t batchSizeIndex);
std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request);

template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_OUTPUT> {
public:
    static Status extract(const tensorflow::serving::PredictRequest& request, const std::string& name, const tensorflow::TensorProto** tensor, size_t* bufferId) {
        return StatusCode::NOT_IMPLEMENTED;
    }
};

template <>
class RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ExtractChoice::EXTRACT_INPUT> {
public:
    static Status extract(const tensorflow::serving::PredictRequest& request, const std::string& name, const tensorflow::TensorProto** tensor, size_t* bufferId) {
        if (bufferId == nullptr) {
            return StatusCode::INTERNAL_ERROR;
        }
        auto it = request.inputs().find(name);
        if (it == request.inputs().end()) {
            return StatusCode::NONEXISTENT_TENSOR;
        }
        *tensor = &it->second;
        return StatusCode::OK;
    }
};
/**
 * This is specific check required for passing KFS API related info
 * which informs how response should be formatted. Therefore return value should not have an impact for
 * any other frontend.
 */
bool useSharedOutputContentFn(const tensorflow::serving::PredictRequest* request);
// TODO @atobisze split
static const Status extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId) {
    if (!proto.tensor_shape().dim_size()) {
        SPDLOG_DEBUG("Sequence id tensor proto does not contain tensor shape information");
        return StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE;
    } else if (proto.tensor_shape().dim_size() != 1) {
        SPDLOG_DEBUG("Sequence id tensor proto shape has invalid number of dimensions. Expecting shape with one dimension");
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Required shape for sequence_id is: (1)");
    }

    if (proto.tensor_shape().dim(0).size() != 1) {
        SPDLOG_DEBUG("Sequence id tensor proto shape has invalid shape. Expecting shape: (1)");
        return Status(StatusCode::INVALID_SHAPE, "Required shape for sequence_id is: (1)");
    }

    if (proto.uint64_val_size() == 1) {
        sequenceId = proto.uint64_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_ID_BAD_TYPE;
}

static const Status extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput) {
    if (proto.tensor_shape().dim_size() == 0) {
        SPDLOG_DEBUG("Sequence control tensor proto does not contain tensor shape information");
        return StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE;
    } else if (proto.tensor_shape().dim_size() != 1) {
        SPDLOG_DEBUG("Sequence control tensor proto shape has invalid number of dimensions. Expecting shape with one dimension.");
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Required shape for sequence_control_input is: (1)");
    }

    if (proto.tensor_shape().dim(0).size() != 1) {
        SPDLOG_DEBUG("Sequence control tensor proto shape has invalid shape. Expecting shape: (1)");
        return Status(StatusCode::INVALID_SHAPE, "Required shape for sequence_control_input is: (1)");
    }

    if (proto.uint32_val_size() == 1) {
        sequenceControlInput = proto.uint32_val(0);
        return StatusCode::OK;
    }
    return StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE;
}
// TODO @atobisze split to cpp
static const Status extractSpecialKeys(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& sequenceProcessingSpec) {
    uint64_t sequenceId = 0;
    uint32_t sequenceControlInput = 0;
    Status status;
    auto it = request->inputs().find("sequence_id");
    if (it != request->inputs().end()) {
        status = extractSequenceId(it->second, sequenceId);
        if (!status.ok())
            return status;
    }
    it = request->inputs().find("sequence_control_input");
    if (it != request->inputs().end()) {
        status = extractSequenceControlInput(it->second, sequenceControlInput);
        if (!status.ok())
            return status;
    }

    if (sequenceControlInput != SEQUENCE_END && sequenceControlInput != NO_CONTROL_INPUT && sequenceControlInput != SEQUENCE_START) {
        return StatusCode::INVALID_SEQUENCE_CONTROL_INPUT;
    }
    if ((sequenceControlInput == SEQUENCE_END || sequenceControlInput == NO_CONTROL_INPUT) && sequenceId == 0) {
        return StatusCode::SEQUENCE_ID_NOT_PROVIDED;
    }
    sequenceProcessingSpec.setSequenceId(sequenceId);
    sequenceProcessingSpec.setSequenceControlInput(sequenceControlInput);
    return StatusCode::OK;
}
template <>
Status StatefulRequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>::extractRequestParameters(const tensorflow::serving::PredictRequest* request) {
    OVMS_PROFILE_FUNCTION();
    auto status = extractSpecialKeys(request, sequenceProcessingSpec);
    return status;
}
template <>
Status StatefulRequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>::postInferenceProcessing(tensorflow::serving::PredictResponse* response, ov::InferRequest& inferRequest) {
    // Reset inferRequest states on SEQUENCE_END
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_END) {
        SPDLOG_DEBUG("Received SEQUENCE_END signal. Resetting model state");
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    } else {
        auto modelState = inferRequest.query_state();
        if (!sequence) {
            SPDLOG_DEBUG("sequence is not set");
            return StatusCode::INTERNAL_ERROR;
        }
        sequence->updateMemoryState(modelState);
    }
    // Include sequence_id in server response
    auto& tensorProto = (*response->mutable_outputs())["sequence_id"];
    tensorProto.mutable_tensor_shape()->add_dim()->set_size(1);
    tensorProto.set_dtype(tensorflow::DataType::DT_UINT64);
    tensorProto.add_uint64_val(sequenceProcessingSpec.getSequenceId());
    return StatusCode::OK;
}
template <>
Status StatefulRequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>::release() {
    SPDLOG_DEBUG("Received SEQUENCE_END signal. Removing sequence");
    sequenceLock->unlock();
    Status status;
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_END) {
        sequenceManagerLock->lock();
        if (!this->sequenceId.has_value()) {
            SPDLOG_DEBUG("sequenceId is not set");
            return StatusCode::INTERNAL_ERROR;
        }
        status = sequenceManager.removeSequence(this->sequenceId.value());
    }
    return status;
}
}  // namespace ovms
