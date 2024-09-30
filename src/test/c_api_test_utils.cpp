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
#include "c_api_test_utils.hpp"

#include "../logging.hpp"
#include "../ovms.h"  // NOLINT

void callbackMarkingItWasUsedWith42(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    using ovms::StatusCode;
    SPDLOG_INFO("Using callback: callbackMarkingItWasUsedWith42!");
    uint32_t* usedFlag = reinterpret_cast<uint32_t*>(userStruct);
    *usedFlag = 42;
    OVMS_InferenceResponseDelete(response);
}
