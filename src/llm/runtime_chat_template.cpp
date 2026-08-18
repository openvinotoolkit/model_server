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

#include "runtime_chat_template.hpp"

#include "runtime_chat_template_runtime_loader.hpp"

#include <string>
#include <utility>

namespace ovms {
namespace {

constexpr const char* PY_RUNTIME_INIT_ERROR_PREFIX = "OVMS_PY_RUNTIME_INIT_ERROR: ";

RuntimeChatTemplateError classifyRuntimeError(const std::string& runtimeOutput) {
    if (runtimeOutput.rfind(PY_RUNTIME_INIT_ERROR_PREFIX, 0) == 0) {
        return RuntimeChatTemplateError::PYTHON_RUNTIME_INITIALIZATION;
    }
    return RuntimeChatTemplateError::EXECUTION_FAILURE;
}

}  // namespace

PreparedRuntimeChatTemplate::PreparedRuntimeChatTemplate(PreparedRuntimeChatTemplate&& other) noexcept :
    handle(std::exchange(other.handle, nullptr)),
    destroyFn(std::exchange(other.destroyFn, nullptr)) {}

PreparedRuntimeChatTemplate& PreparedRuntimeChatTemplate::operator=(PreparedRuntimeChatTemplate&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    handle = std::exchange(other.handle, nullptr);
    destroyFn = std::exchange(other.destroyFn, nullptr);
    return *this;
}

PreparedRuntimeChatTemplate::~PreparedRuntimeChatTemplate() {
    reset();
}

bool PreparedRuntimeChatTemplate::isPrepared() const {
    return handle != nullptr;
}

void PreparedRuntimeChatTemplate::reset() {
    if (handle != nullptr && destroyFn != nullptr) {
        destroyFn(handle);
    }
    handle = nullptr;
    destroyFn = nullptr;
}

RuntimeChatTemplatePrepareStatus prepareRuntimeChatTemplate(
    const std::string& modelsPath,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    PreparedRuntimeChatTemplate& preparedTemplate,
    std::string& output,
    RuntimeChatTemplateError* error) {
    if (error != nullptr) {
        *error = RuntimeChatTemplateError::NONE;
    }
    const auto* api = getRuntimeChatTemplateRuntimeApi();
    if (api == nullptr || api->createPreparedFn == nullptr || api->destroyPreparedFn == nullptr) {
        return RuntimeChatTemplatePrepareStatus::UNAVAILABLE;
    }

    void* runtimeHandle = nullptr;
    const char* runtimeOutput = nullptr;
    bool ok = api->createPreparedFn(
        modelsPath.c_str(),
        chatTemplate.c_str(),
        bosToken.c_str(),
        eosToken.c_str(),
        &runtimeHandle,
        &runtimeOutput);

    if (runtimeOutput != nullptr) {
        output = runtimeOutput;
    } else {
        output.clear();
    }

    if (!ok) {
        if (output.empty()) {
            output = "Failed to prepare chat template using runtime Python Jinja";
        }
        if (error != nullptr) {
            *error = classifyRuntimeError(output);
        }
        return RuntimeChatTemplatePrepareStatus::ERROR;
    }

    if (runtimeHandle == nullptr) {
        if (output.empty()) {
            output = "Failed to prepare chat template using runtime Python Jinja";
        }
        if (error != nullptr) {
            *error = RuntimeChatTemplateError::EXECUTION_FAILURE;
        }
        return RuntimeChatTemplatePrepareStatus::ERROR;
    }

    PreparedRuntimeChatTemplate newPreparedTemplate;
    newPreparedTemplate.handle = runtimeHandle;
    newPreparedTemplate.destroyFn = api->destroyPreparedFn;
    preparedTemplate = std::move(newPreparedTemplate);
    return RuntimeChatTemplatePrepareStatus::PREPARED;
}

RuntimeChatTemplateStatus tryApplyPreparedChatTemplateRuntime(
    const PreparedRuntimeChatTemplate& preparedTemplate,
    const std::string& requestBody,
    std::string& output,
    RuntimeChatTemplateError* error) {
    if (error != nullptr) {
        *error = RuntimeChatTemplateError::NONE;
    }
    const auto* api = getRuntimeChatTemplateRuntimeApi();
    if (api == nullptr || api->applyPreparedFn == nullptr || !preparedTemplate.isPrepared()) {
        return RuntimeChatTemplateStatus::UNAVAILABLE;
    }

    const char* runtimeOutput = nullptr;
    bool ok = api->applyPreparedFn(
        preparedTemplate.handle,
        requestBody.c_str(),
        &runtimeOutput);

    if (runtimeOutput != nullptr) {
        output = runtimeOutput;
    } else {
        output.clear();
    }

    if (!ok) {
        if (output.empty()) {
            output = "Failed to apply prepared chat template using runtime Python Jinja";
        }
        if (error != nullptr) {
            *error = classifyRuntimeError(output);
        }
        return RuntimeChatTemplateStatus::ERROR;
    }

    return RuntimeChatTemplateStatus::APPLIED;
}

RuntimeChatTemplateStatus tryApplyChatTemplateRuntime(
    const std::string& modelsPath,
    const std::string& requestBody,
    const std::string& chatTemplate,
    const std::string& bosToken,
    const std::string& eosToken,
    std::string& output) {
    PreparedRuntimeChatTemplate preparedTemplate;
    auto prepareStatus = prepareRuntimeChatTemplate(
        modelsPath,
        chatTemplate,
        bosToken,
        eosToken,
        preparedTemplate,
        output);
    if (prepareStatus == RuntimeChatTemplatePrepareStatus::UNAVAILABLE) {
        return RuntimeChatTemplateStatus::UNAVAILABLE;
    }
    if (prepareStatus == RuntimeChatTemplatePrepareStatus::ERROR) {
        return RuntimeChatTemplateStatus::ERROR;
    }
    return tryApplyPreparedChatTemplateRuntime(preparedTemplate, requestBody, output);
}

}  // namespace ovms
