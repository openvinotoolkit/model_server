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

#include "kfs_utils.hpp"
#include "../extractchoice.hpp"
#include "../requesttensorextractor.hpp"
#include "../shape.hpp"
#include "../status.hpp"

namespace ovms {
std::optional<Dimension> getRequestBatchSize(const ::KFSRequest* request, const size_t batchSizeIndex);
std::map<std::string, shape_t> getRequestShapes(const ::KFSRequest* request);

template <>
class RequestTensorExtractor<KFSRequest, KFSTensorInputProto, ExtractChoice::EXTRACT_OUTPUT> {
public:
    static Status extract(const KFSRequest& request, const std::string& name, const KFSTensorInputProto** tensor, size_t* bufferId) {
        return StatusCode::NOT_IMPLEMENTED;
    }
};

template <>
class RequestTensorExtractor<KFSRequest, KFSTensorInputProto, ExtractChoice::EXTRACT_INPUT> {
public:
    static Status extract(const KFSRequest& request, const std::string& name, const KFSTensorInputProto** tensor, size_t* bufferId);
};

/**
 * This is specific check required for passing KFS API related info
 * which informs how response should be formatted. Therefore return value should not have an impact for
 * any other frontend.
 */
bool useSharedOutputContentFn(const ::KFSRequest* request);
}  // namespace ovms
