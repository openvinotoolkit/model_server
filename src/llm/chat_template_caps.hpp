//*****************************************************************************
// Copyright 2026 Intel Corporation
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

namespace ovms {

struct ChatTemplateCaps {
    // TODO: Do we keep it?
    bool supportsToolCalls = false;

    // Some templates require tool_call arguments to be a dict/object rather than a stringified JSON.
    bool requiresObjectArguments = false;

    // Messages with tool_calls may require content="" rather than content=null for some templates (e.g. llama3).
    bool requiresNonNullContent = false;

    bool needsWorkarounds() const {
        return requiresObjectArguments || requiresNonNullContent;
    }
};

}  // namespace ovms
