//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "env_guard.hpp"

#include "../logging.hpp"

#include <stdlib.h>

const std::string GetEnvVar(const std::string& var) {
    std::string val = "";
    const char* envCred = std::getenv(var.c_str());
    if (envCred)
        val = std::string(envCred);
    return val;
}

void SetEnvironmentVar(const std::string& var, const std::string& val) {
    SPDLOG_INFO("Setting environment variable: {} to: {}", var, val);
#ifdef _WIN32
    _putenv_s(var.c_str(), val.c_str());
#elif __linux__
    ::setenv(var.c_str(), val.c_str(), 1);
#endif
}

void UnSetEnvironmentVar(const std::string& var) {
    SPDLOG_INFO("Unsetting environment variable: {}", var);
#ifdef _WIN32
    _putenv_s(var.c_str(), "");
#elif __linux__
    ::unsetenv(var.c_str());
#endif
}

EnvGuard::EnvGuard() {
    SPDLOG_TRACE("EnvGuardConstructor");
}

void EnvGuard::set(const std::string& name, const std::string& value) {
    std::optional<std::string> originalValue = std::nullopt;
    const char* currentVal = std::getenv(name.c_str());
    if (currentVal) {
        SPDLOG_TRACE("Var:{} is set to value:{}", name, currentVal);
        originalValue = std::string(currentVal);
    } else {
        SPDLOG_TRACE("Var:{} was not set", name);
    }
    if (originalValues.find(name) == originalValues.end()) {
        SPDLOG_TRACE("Var:{} value was not stored yet", name);
        originalValues[name] = originalValue;
    }
    SetEnvironmentVar(name, value);
}

void EnvGuard::unset(const std::string& name) {
    std::optional<std::string> originalValue = std::nullopt;
    const char* currentVal = std::getenv(name.c_str());
    if (currentVal) {
        SPDLOG_TRACE("Var:{} is set to value:{}", name, currentVal);
        originalValue = std::string(currentVal);
    } else {
        SPDLOG_TRACE("Var:{} was not set", name);
    }
    if (originalValues.find(name) == originalValues.end()) {
        SPDLOG_TRACE("Var:{} value was not stored yet", name);
        originalValues[name] = originalValue;
    }
    UnSetEnvironmentVar(name);
}

EnvGuard::~EnvGuard() {
    SPDLOG_TRACE("EnvGuardDestructor");
    for (auto& [k, v] : originalValues) {
        if (v.has_value()) {
            SPDLOG_TRACE("Var:{} was set to value:{}", k, v.value());
            SetEnvironmentVar(k, v.value());
        } else {
            SPDLOG_TRACE("Var:{} was empty", k);
            UnSetEnvironmentVar(k);
        }
    }
}
