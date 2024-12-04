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
#include "requestprocessor.hpp"

#include "status.hpp"

namespace ovms {
template <typename RequestType, typename ResponseType>
RequestProcessor<RequestType, ResponseType>::RequestProcessor() = default;
template <typename RequestType, typename ResponseType>
RequestProcessor<RequestType, ResponseType>::~RequestProcessor() = default;
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::extractRequestParameters(const RequestType* request) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::prepare() { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::preInferenceProcessing(ov::InferRequest& inferRequest) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::postInferenceProcessing(ResponseType* response, ov::InferRequest& inferRequest) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::release() { return StatusCode::OK; }
}  // namespace ovms
