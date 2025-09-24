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
#pragma once

#include <map>
#include <optional>
#include <unordered_map>
#include <string>

void SetEnvironmentVar(const std::string& var, const std::string& val);
void UnSetEnvironmentVar(const std::string& var);
const std::string GetEnvVar(const std::string& var);

struct EnvGuard {
    EnvGuard();
    void set(const std::string& name, const std::string& value);
    void unset(const std::string& name);
    ~EnvGuard();

private:
    std::unordered_map<std::string, std::optional<std::string>> originalValues;
};
