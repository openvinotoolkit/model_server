//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <optional>
#include <set>
#include <string>

#include "kfs_frontend/kfs_grpc_inference_service.hpp"

namespace ovms {
class Status;
Status makeJsonFromPredictResponse(
    const ::KFSResponse& response_proto,
    std::string* response_json,
    std::optional<int>& inferenceHeaderContentLength,
    const std::set<std::string>& requestedBinaryOutputsNames = {});

Status decodeBase64(std::string& bytes, std::string& decodedBytes);

}  // namespace ovms
