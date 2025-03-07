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

#include "kfs_utils.hpp"
#include "kfs_request_utils.hpp"

#include "../status.hpp"
#include "../modelinstance.hpp"
#include "../deserialization_main.hpp"
#include "../inference_executor.hpp"
#include "../ovms.h"  // NOLINT

namespace ovms {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
OVMS_InferenceRequestCompletionCallback_t getCallback(const KFSRequest* request) {
    return request->getResponseCompleteCallback();
}  // TODO check if this exact impl is used @atobisze
#pragma GCC diagnostic pop
template Status modelInferAsync<KFSRequest, KFSResponse>(ModelInstance& instance, const KFSRequest*, std::unique_ptr<ModelInstanceUnloadGuard>&);
template Status infer<KFSRequest, KFSResponse>(ModelInstance& instance, const KFSRequest*, KFSResponse*, std::unique_ptr<ModelInstanceUnloadGuard>&);

// TODO @atobisze use from dags?
using TensorMap = std::unordered_map<std::string, ov::Tensor>;
template Status ovms::serializePredictResponse<const TensorMap&>(
    OutputGetter<const TensorMap&>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent);

template Status ovms::serializePredictResponse<ov::InferRequest&>(
    OutputGetter<ov::InferRequest&>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const KFSRequest* request,
    KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent);  // TODO move to serialization.cpp
template Status serializePredictResponse<KFSRequest, KFSResponse, ov::InferRequest&>(
    OutputGetter<ov::InferRequest&>& outputGetter,
    const std::string& servableName,
    model_version_t servableVersion,
    const tensor_map_t& outputMap,
    const KFSRequest* request,
    KFSResponse* response,
    outputNameChooser_t outputNameChooser,
    bool useSharedOutputContent);
}  // namespace ovms
