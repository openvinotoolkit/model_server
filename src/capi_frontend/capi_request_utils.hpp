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
#include <string>
#include <vector>

#include "../ovms.h"  // NOLINT
#include "../precision.hpp"
#include "inferencerequest.hpp"
#include "../shape.hpp"
#include "../logging.hpp"
#include "../status.hpp" // TODO move impl @atobisze
#include "../extractchoice.hpp"
#include "../requesttensorextractor.hpp"
namespace ovms {
class InferenceRequest;
class InferenceResponse;
class InferenceTensor;
class Status;

std::optional<Dimension> getRequestBatchSize(const InferenceRequest* request, const size_t batchSizeIndex);
std::map<std::string, shape_t> getRequestShapes(const InferenceRequest* request);
bool useSharedOutputContentFn(const InferenceRequest* request);

template <>
class RequestTensorExtractor<InferenceRequest, InferenceTensor, ExtractChoice::EXTRACT_OUTPUT> {
public:
    static Status extract(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor, size_t* bufferId = nullptr) {
        SPDLOG_TRACE("Extracting output: {}", name);
        return request.getOutput(name.c_str(), tensor);
    }
};

template <>
class RequestTensorExtractor<InferenceRequest, InferenceTensor, ExtractChoice::EXTRACT_INPUT> {
public:
    static Status extract(const InferenceRequest& request, const std::string& name, const InferenceTensor** tensor, size_t* bufferId = nullptr) {
        SPDLOG_TRACE("Extracting input: {}", name);
        return request.getInput(name.c_str(), tensor);
    }
};

}  // namespace ovms
