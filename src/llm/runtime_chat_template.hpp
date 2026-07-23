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

#include <string>

namespace ovms {

enum class RuntimeChatTemplateError {
    NONE,
    PYTHON_RUNTIME_INITIALIZATION,
    EXECUTION_FAILURE,
};

enum class RuntimeChatTemplatePrepareStatus {
    PREPARED,
    UNAVAILABLE,
    ERROR,
};

struct PreparedRuntimeChatTemplate;

RuntimeChatTemplatePrepareStatus prepareRuntimeChatTemplate(
    const std::string& modelsPath,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    PreparedRuntimeChatTemplate& preparedTemplate,
    std::string& output,
    RuntimeChatTemplateError* error = nullptr);

enum class RuntimeChatTemplateStatus {
    APPLIED,
    UNAVAILABLE,
    ERROR,
};

RuntimeChatTemplateStatus tryApplyPreparedChatTemplateRuntime(
    const PreparedRuntimeChatTemplate& preparedTemplate,
    const std::string& requestBody,
    std::string& output,
    RuntimeChatTemplateError* error = nullptr);

struct PreparedRuntimeChatTemplate {
    PreparedRuntimeChatTemplate() = default;
    PreparedRuntimeChatTemplate(const PreparedRuntimeChatTemplate&) = delete;
    PreparedRuntimeChatTemplate& operator=(const PreparedRuntimeChatTemplate&) = delete;
    PreparedRuntimeChatTemplate(PreparedRuntimeChatTemplate&& other) noexcept;
    PreparedRuntimeChatTemplate& operator=(PreparedRuntimeChatTemplate&& other) noexcept;
    ~PreparedRuntimeChatTemplate();

    bool isPrepared() const;
    void reset();

private:
    void* handle = nullptr;
    void (*destroyFn)(void*) = nullptr;

    friend RuntimeChatTemplatePrepareStatus prepareRuntimeChatTemplate(
        const std::string& modelsPath,
        const std::string& chatTemplate,
        const std::string& bosToken,
        const std::string& eosToken,
        PreparedRuntimeChatTemplate& preparedTemplate,
        std::string& output,
        RuntimeChatTemplateError* error);
    friend RuntimeChatTemplateStatus tryApplyPreparedChatTemplateRuntime(
        const PreparedRuntimeChatTemplate& preparedTemplate,
        const std::string& requestBody,
        std::string& output,
        RuntimeChatTemplateError* error);
};

RuntimeChatTemplateStatus tryApplyChatTemplateRuntime(
    const std::string& modelsPath,
    const std::string& requestBody,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    std::string& output);

}  // namespace ovms
