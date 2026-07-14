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

#include <string>

#include "src/port/rapidjson_document.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "kfs_frontend/kfs_grpc_inference_service.hpp"
#pragma GCC diagnostic pop

namespace ovms {
class Status;

class RestParser {};

class KFSRestParser : RestParser {
    ::KFSRequest requestProto;
    Status parseId(rapidjson::Value& node);
    Status parseRequestParameters(rapidjson::Value& node);
    Status parseInputParameters(rapidjson::Value& node, ::KFSRequest::InferInputTensor& input);
    Status parseOutputParameters(rapidjson::Value& node, ::KFSRequest::InferRequestedOutputTensor& input);
    Status parseOutput(rapidjson::Value& node);
    Status parseOutputs(rapidjson::Value& node);
    Status parseData(rapidjson::Value& node, ::KFSRequest::InferInputTensor& input);
    Status parseInput(rapidjson::Value& node, bool onlyOneInput);
    Status parseInputs(rapidjson::Value& node);

public:
    Status parse(const char* json);
    ::KFSRequest& getProto() { return requestProto; }
};

}  // namespace ovms
