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
#include <iostream>

#include "ovms.h"
#include "extractchoice.hpp"
// TODO @atobisze this is request interface to be implementend by backend
// do we need more headers or just one? For now keeping it in here
namespace ovms {
class Status;
template <typename Request, typename InputTensorType, ExtractChoice choice>
class RequestTensorExtractor {
public:
    static Status extract(const Request& request, const std::string& name, const InputTensorType** tensor, size_t* bufferId = nullptr);
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function";
template <typename RequestType>
inline OVMS_InferenceRequestCompletionCallback_t getCallback(RequestType request) {
    return nullptr;
}
#pragma GCC diagnostic pop
}  // namespace ovms
