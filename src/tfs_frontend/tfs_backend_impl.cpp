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
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <string>

#include "tfs_request_utils.hpp"  // TODO @atobisze order matters here
#include "tfs_utils.hpp"
#include "../status.hpp"
#include "../modelinstance.hpp"
#include "../deserialization_main.hpp"
#include "../inference_executor.hpp"
#include "../ovms.h"  // NOLINT
#include "../statefulrequestprocessor.hpp"
#include "../status.hpp"
#include "serialization.hpp"
#include "../serialization_common.hpp"
#include "deserialization.hpp"
#include "../deserialization_common.hpp"
#include "../requesttensorextractor.hpp"
#include "tfs_request_utils.hpp"

namespace ovms {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
OVMS_InferenceRequestCompletionCallback_t getCallback(const TFSPredictRequest* request) {
    return nullptr;
}
#pragma GCC diagnostic pop
template Status infer<TFSPredictRequest, TFSPredictResponse>(ModelInstance& instance, const TFSPredictRequest*, TFSPredictResponse*, std::unique_ptr<ModelInstanceUnloadGuard>&);
template class RequestTensorExtractor<TFSPredictRequest, TFSInputTensorType, ExtractChoice::EXTRACT_INPUT>;
}  // namespace ovms
template class ovms::RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ovms::ExtractChoice::EXTRACT_INPUT>;
template class ovms::RequestTensorExtractor<tensorflow::serving::PredictRequest, tensorflow::TensorProto, ovms::ExtractChoice::EXTRACT_INPUT>;
