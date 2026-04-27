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

#include <memory>
#include <string>
#include <thread>

namespace ovms {
class Server;
}  // namespace ovms

void randomizeAndEnsureFree(std::string& port);
void randomizeAndEnsureFrees(std::string& port1, std::string& port2);

extern const int64_t SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS;

void EnsureServerStartedWithTimeout(ovms::Server& server, int timeoutSeconds);
void EnsureServerModelDownloadFinishedWithTimeout(ovms::Server& server, int timeoutSeconds);

void SetUpServerForDownload(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& source_model, std::string& download_path, std::string& task, int expected_code = EXIT_SUCCESS, int timeoutSeconds = SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);
void SetUpServerForDownloadWithDraft(std::unique_ptr<std::thread>& t, ovms::Server& server,
    std::string& draftModel, std::string& source_model, std::string& download_path, std::string& task, int expected_code, int timeoutSeconds = 2 * SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);
void SetUpServerForDownloadAndStartGGUF(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& ggufFilename, std::string& sourceModel, std::string& downloadPath, std::string& task, int timeoutSeconds = 4 * SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);
void SetUpServerForDownloadAndStart(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& source_model, std::string& download_path, std::string& task, int timeoutSeconds = SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);
void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* configPath, int timeoutSeconds = SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS, std::string apiKeyFile = "");
void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* modelPath, const char* modelName, int timeoutSeconds = SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS);
