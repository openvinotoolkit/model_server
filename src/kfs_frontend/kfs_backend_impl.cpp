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

#include "kfs_request_utils.hpp"
#include "kfs_utils.hpp"
#include "../status.hpp"
#include "../modelinstance.hpp"
#include "../deserialization_main.hpp"
#include "serialization.hpp"
#include "deserialization.hpp"
#include "../inference_executor.hpp"
#include "../ovms.h"  // NOLINT
#include "../statefulrequestprocessor.hpp"
#include "../status.hpp"
#include "validation.hpp"

namespace ovms {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
OVMS_InferenceRequestCompletionCallback_t getCallback(const KFSRequest* request) {
    return nullptr; // TODO @atobisze is there no spec impl?
} // TODO check if this exact impl is used @atobisze
#pragma GCC diagnostic pop
template Status infer<KFSRequest, KFSResponse>(ModelInstance& instance, const KFSRequest*, KFSResponse*, std::unique_ptr<ModelInstanceUnloadGuard>&);
}  // namespace ovms
