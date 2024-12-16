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
#include "statefulrequestprocessor.hpp"

#include <openvino/openvino.hpp>
#include <openvino/pass/low_latency.hpp>

#include "logging.hpp"
#include "model_metric_reporter.hpp"
//#include "modelconfig.hpp"
#include "predict_request_validation_utils.hpp"
#include "profiler.hpp"
#include "sequence_processing_spec.hpp"
#include "serialization.hpp"
#include "timer.hpp"

namespace ovms {
const Status StatefulModelInstance::postInferenceProcessing(tensorflow::serving::PredictResponse* response,
    ov::InferRequest& inferRequest, Sequence& sequence, SequenceProcessingSpec& sequenceProcessingSpec) {
    // Reset inferRequest states on SEQUENCE_END
    if (sequenceProcessingSpec.getSequenceControlInput() == SEQUENCE_END) {
        spdlog::debug("Received SEQUENCE_END signal. Resetting model state and removing sequence");
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    } else {
        auto modelState = inferRequest.query_state();
        sequence.updateMemoryState(modelState);
    }

    // Include sequence_id in server response
    auto& tensorProto = (*response->mutable_outputs())["sequence_id"];
    tensorProto.mutable_tensor_shape()->add_dim()->set_size(1);
    tensorProto.set_dtype(tensorflow::DataType::DT_UINT64);
    tensorProto.add_uint64_val(sequenceProcessingSpec.getSequenceId());

    return StatusCode::OK;
}
}  // namespace ovms
