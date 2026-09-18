#pragma once
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
#include <cstddef>
#include <string>

namespace ovms {
class Status;

Status downloadFileWithCurl(const std::string& url, const std::string& filePath);
Status downloadFileWithCurl(const std::string& url, const std::string& filePath, const std::string& authTokenHF);
Status fetchUrlToString(const std::string& url, const std::string& authToken, std::string& responseBody);

// Number of filled cells in a barWidth-wide progress bar for count out of max bytes,
// clamped to [0, barWidth]. max == 0 means the server sent no Content-Length, so there is
// no ratio to render and the result is 0. Declared here so the arithmetic can be unit tested.
int computeProgressBarCells(size_t count, size_t max, int barWidth);

}  // namespace ovms
