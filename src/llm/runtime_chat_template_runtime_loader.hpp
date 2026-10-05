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

#include <string>

namespace ovms {

bool ensurePythonRuntimeInitialized(std::string& errorMessage);

class PreparedChatTemplateRuntime {
public:
    PreparedChatTemplateRuntime() = default;
    PreparedChatTemplateRuntime(const PreparedChatTemplateRuntime&) = delete;
    PreparedChatTemplateRuntime& operator=(const PreparedChatTemplateRuntime&) = delete;
    ~PreparedChatTemplateRuntime();

    bool prepare(const std::string& modelsPath, const std::string& chatTemplate,
        const std::string& bosToken, const std::string& eosToken, std::string& errorMessage);
    bool apply(const std::string& requestBody, std::string& output) const;
    bool isPrepared() const { return handle != nullptr; }

private:
    void* handle = nullptr;
    void (*destroy)(void*) = nullptr;
    bool (*render)(void*, const char*, const char**) = nullptr;
};

}  // namespace ovms
