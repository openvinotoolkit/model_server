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
#include "test_server_utils.hpp"

#include <chrono>
#include <random>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "platform_utils.hpp"

#include "src/network_utils.hpp"
#include "src/servablemanagermodule.hpp"
#include "src/server.hpp"

void randomizeAndEnsureFree(std::string& port) {
    std::mt19937_64 eng{std::random_device{}()};
    std::uniform_int_distribution<> dist{0, 9};
    int tryCount = 3;
    while (tryCount--) {
        for (auto j : {1, 2, 3}) {
            char* digitToRandomize = (char*)port.c_str() + j;
            *digitToRandomize = '0' + dist(eng);
        }
        if (ovms::isPortAvailable(std::stoi(port))) {
            return;
        } else {
            continue;
        }
    }
    EXPECT_TRUE(false) << "Could not find random available port";
}

void randomizeAndEnsureFrees(std::string& port1, std::string& port2) {
    randomizeAndEnsureFree(port1);
    randomizeAndEnsureFree(port2);
    while (port2 == port1) {
        randomizeAndEnsureFree(port2);
    }
}

const int64_t SERVER_START_FROM_CONFIG_TIMEOUT_SECONDS = 60;

void EnsureServerStartedWithTimeout(ovms::Server& server, int timeoutSeconds) {
    auto start = std::chrono::high_resolution_clock::now();
    int timestepMs = 20;
    while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < timeoutSeconds)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(timestepMs));
    }
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED) << "OVMS did not fully load until allowed time:" << timeoutSeconds << "s. Check machine load";
}

void EnsureServerModelDownloadFinishedWithTimeout(ovms::Server& server, int timeoutSeconds) {
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::HF_MODEL_PULL_MODULE_NAME) != ovms::ModuleState::SHUTDOWN) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < timeoutSeconds)) {
    }

    ASSERT_EQ(server.getModuleState(ovms::HF_MODEL_PULL_MODULE_NAME), ovms::ModuleState::SHUTDOWN) << "OVMS did not download model in allowed time:" << timeoutSeconds << "s. Check machine load and network load";
}

void SetUpServerForDownload(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& source_model, std::string& download_path, std::string& task, int expected_code, int timeoutSeconds) {
    server.setShutdownRequest(0);
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)source_model.c_str(),
        (char*)"--model_repository_path",
        (char*)download_path.c_str(),
        (char*)"--task",
        (char*)task.c_str()};

    int argc = 8;
    t.reset(new std::thread([&argc, &argv, &server, expected_code]() {
        EXPECT_EQ(expected_code, server.start(argc, argv));
    }));

    EnsureServerModelDownloadFinishedWithTimeout(server, timeoutSeconds);
}

void SetUpServerForDownloadWithDraft(std::unique_ptr<std::thread>& t, ovms::Server& server,
    std::string& draftModel, std::string& source_model, std::string& download_path, std::string& task, int expected_code, int timeoutSeconds) {
    server.setShutdownRequest(0);
    char* argv[] = {(char*)"ovms",
        (char*)"--pull",
        (char*)"--source_model",
        (char*)source_model.c_str(),
        (char*)"--model_repository_path",
        (char*)download_path.c_str(),
        (char*)"--task",
        (char*)task.c_str(),
        (char*)"--draft_source_model",
        (char*)draftModel.c_str()};

    int argc = 10;
    t.reset(new std::thread([&argc, &argv, &server, expected_code]() {
        EXPECT_EQ(expected_code, server.start(argc, argv));
    }));

    EnsureServerModelDownloadFinishedWithTimeout(server, timeoutSeconds);
}

void SetUpServerForDownloadAndStart(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& source_model, std::string& download_path, std::string& task, int timeoutSeconds) {
    server.setShutdownRequest(0);
    std::string port = "9133";
    randomizeAndEnsureFree(port);
    char* argv[] = {(char*)"ovms",
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--source_model",
        (char*)source_model.c_str(),
        (char*)"--model_repository_path",
        (char*)download_path.c_str(),
        (char*)"--task",
        (char*)task.c_str()};

    int argc = 9;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));

    EnsureServerStartedWithTimeout(server, timeoutSeconds);
}

void SetUpServerForDownloadAndStartGGUF(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& ggufFilename, std::string& sourceModel, std::string& downloadPath, std::string& task, int timeoutSeconds) {
    server.setShutdownRequest(0);
    std::string port = "9133";
    randomizeAndEnsureFree(port);
    char* argv[] = {
        (char*)"ovms",
        (char*)"--port",
        (char*)port.c_str(),
        (char*)"--source_model",
        (char*)sourceModel.c_str(),
        (char*)"--model_repository_path",
        (char*)downloadPath.c_str(),
        (char*)"--task",
        (char*)task.c_str(),
        (char*)"--gguf_filename",
        (char*)ggufFilename.c_str(),
    };

    int argc = 11;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));

    EnsureServerStartedWithTimeout(server, timeoutSeconds);
}

void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* configPath, int timeoutSeconds, std::string api_key) {
    server.setShutdownRequest(0);
    randomizeAndEnsureFree(port);
    if (!api_key.empty()) {
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str(),
            (char*)"--api_key_file",
            (char*)api_key.c_str()};
        int argc = 7;
        t.reset(new std::thread([&argc, &argv, &server]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        EnsureServerStartedWithTimeout(server, timeoutSeconds);
    } else {
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, &server]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        EnsureServerStartedWithTimeout(server, timeoutSeconds);
    }
}

void SetUpServer(std::unique_ptr<std::thread>& t, ovms::Server& server, std::string& port, const char* modelPath, const char* modelName, int timeoutSeconds) {
    server.setShutdownRequest(0);
    randomizeAndEnsureFree(port);
    char* argv[] = {(char*)"ovms",
        (char*)"--model_name",
        (char*)modelName,
        (char*)"--model_path",
        (char*)getGenericFullPathForSrcTest(modelPath).c_str(),
        (char*)"--port",
        (char*)port.c_str()};
    int argc = 7;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));
    EnsureServerStartedWithTimeout(server, timeoutSeconds);
}
