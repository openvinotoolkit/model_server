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
//*****************************************************************************/
#include "runtime_config.hpp"

#include <sstream>
#include <utility>

namespace {
ovms::RuntimeConfig gRuntimeConfig;
}

namespace ovms {

const RuntimeConfig& getRuntimeConfig() {
    return gRuntimeConfig;
}

void setRuntimeConfig(const char* allowedLocalMediaPath,
    const char* allowedMediaDomains,
    const char* cacheDir,
    uint32_t restWorkers,
    bool verboseResponse) {
    gRuntimeConfig.allowedLocalMediaPath = (allowedLocalMediaPath != nullptr && allowedLocalMediaPath[0] != '\0')
                                               ? std::make_optional<std::string>(allowedLocalMediaPath)
                                               : std::nullopt;
    gRuntimeConfig.allowedMediaDomains.clear();
    if (allowedMediaDomains != nullptr) {
        std::stringstream domains(allowedMediaDomains);
        std::string domain;
        while (std::getline(domains, domain, ',')) {
            if (!domain.empty()) {
                gRuntimeConfig.allowedMediaDomains.push_back(std::move(domain));
            }
        }
    }
    gRuntimeConfig.cacheDir = cacheDir != nullptr ? cacheDir : "";
    gRuntimeConfig.restWorkers = restWorkers;
    gRuntimeConfig.verboseResponse = verboseResponse;
}

}  // namespace ovms
