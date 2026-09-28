//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

namespace ovms {

struct RuntimeChatTemplateRuntimeApi {
    using CreatePreparedChatTemplateRuntimeFn = bool (*)(
        const char* modelsPath,
        const char* chatTemplate,
        const char* bosToken,
        const char* eosToken,
        void** preparedHandle,
        const char** output);

    using ApplyPreparedChatTemplateRuntimeFn = bool (*)(
        void* preparedHandle,
        const char* requestBody,
        const char** output);

    using DestroyPreparedChatTemplateRuntimeFn = void (*)(void* preparedHandle);

    CreatePreparedChatTemplateRuntimeFn createPreparedFn = nullptr;
    ApplyPreparedChatTemplateRuntimeFn applyPreparedFn = nullptr;
    DestroyPreparedChatTemplateRuntimeFn destroyPreparedFn = nullptr;
};

// Returns nullptr when runtime library or required symbols are unavailable.
const RuntimeChatTemplateRuntimeApi* getRuntimeChatTemplateRuntimeApi();

}  // namespace ovms
