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

#include "capi_request_utils.hpp"

#include "../status.hpp"
#include "../modelinstance.hpp"
#include "../deserialization_main.hpp"
#include "../inference_executor.hpp"
#include "../ovms.h"  // NOLINT
#include "../status.hpp"
#include "capi_utils.hpp"
#include "serialization.hpp"
#include "deserialization.hpp"
#include "inferencerequest.hpp"
#include "inferenceresponse.hpp"
#include "capi_validation.hpp"

namespace ovms {
//const Status ModelInstance::validate(const InferenceRequest* request);

template Status modelInferAsync<InferenceRequest, InferenceResponse>(ModelInstance& instance, const InferenceRequest*, std::unique_ptr<ModelInstanceUnloadGuard>&);
template Status infer<InferenceRequest, InferenceResponse>(ModelInstance& instance, const InferenceRequest*, InferenceResponse*, std::unique_ptr<ModelInstanceUnloadGuard>&);
template class OutputGetter<ov::InferRequest>;
}  // namespace ovms
