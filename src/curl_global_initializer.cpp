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
#include "curl_global_initializer.hpp"

#include <curl/curl.h>

#include "logging.hpp"

namespace ovms {

Status initializeCurlGlobal() {
    const CURLcode initResult = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (initResult != CURLE_OK) {
        SPDLOG_ERROR("curl error: {}. Error code: {}", curl_easy_strerror(initResult), (int)initResult);
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

void cleanupCurlGlobal() {
    curl_global_cleanup();
}

}  // namespace ovms
