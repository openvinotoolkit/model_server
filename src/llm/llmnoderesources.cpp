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
#include <string>
#include <spdlog/spdlog.h>
#include "../logging.hpp"
#include "../status.hpp"
#include "llmnoderesources.hpp"
#include "apis/openai_completions.hpp"

namespace ovms {
    /*
absl::Status LLMNodeResources::loadRequest(ov::AnyMap& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext["endpoint"] = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/completions" || payload.uri == "/v3/v1/completions") {
        executionContext["endpoint"] = Endpoint::COMPLETIONS;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
    }
    executionContext["payload"] = payload;
    return absl::OkStatus();
}
    */
}  // namespace ovms
