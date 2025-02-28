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
#include "capi_request_utils.hpp"

#include <string>
#include <vector>

#include "../ovms.h"  // NOLINT
#include "inferencerequest.hpp"
#include "../shape.hpp"
#include "../logging.hpp"
#include "../status.hpp" // TODO move impl @atobisze

namespace ovms {
class InferenceRequest;
class InferenceResponse;
class InferenceTensor;
class Status;
std::optional<Dimension> getRequestBatchSize(const InferenceRequest* request, const size_t batchSizeIndex) {
    size_t bs = 0;
    auto status = request->getBatchSize(bs, batchSizeIndex);
    if (!status.ok()) {
        return std::nullopt;
    }
    return bs;
}

std::map<std::string, shape_t> getRequestShapes(const InferenceRequest* request) {
    return request->getRequestShapes();
}
bool useSharedOutputContentFn(const InferenceRequest* request) {
    // does not apply for C-API frontend
    return false;
}
}  // namespace ovms
