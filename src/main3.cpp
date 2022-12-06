//*****************************************************************************
// Copyright 2022 Intel Corporation
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
// #include <chrono>
// #include <iostream>
// #include <thread>

#include <signal.h>
#include <stdio.h>

#include "pocapi.hpp"

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

static void onInterrupt(int status) {
    shutdown_request = 1;
}

static void onTerminate(int status) {
    shutdown_request = 1;
}

static void onIllegal(int status) {
    shutdown_request = 2;
}

static void installSignalHandlers() {
    static struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = onInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    static struct sigaction sigTermHandler;
    sigTermHandler.sa_handler = onTerminate;
    sigemptyset(&sigTermHandler.sa_mask);
    sigTermHandler.sa_flags = 0;
    sigaction(SIGTERM, &sigTermHandler, NULL);

    static struct sigaction sigIllHandler;
    sigIllHandler.sa_handler = onIllegal;
    sigemptyset(&sigIllHandler.sa_mask);
    sigIllHandler.sa_flags = 0;
    sigaction(SIGILL, &sigIllHandler, NULL);
}

int main(int argc, char** argv) {
    installSignalHandlers();

    OVMS_ServerGeneralOptions* go = 0;
    OVMS_ServerMultiModelOptions* mmo = 0;
    OVMS_Server* srv;

    OVMS_ServerGeneralOptionsNew(&go);
    OVMS_ServerMultiModelOptionsNew(&mmo);
    OVMS_ServerNew(&srv);

    OVMS_ServerGeneralOptionsSetGrpcPort(go, 11337);
    OVMS_ServerGeneralOptionsSetRestPort(go, 11338);

    OVMS_ServerGeneralOptionsSetLogLevel(go, OVMS_LOG_DEBUG);
    OVMS_ServerMultiModelOptionsSetConfigPath(mmo, "/ovms/src/test/c_api/config.json");

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, go, mmo);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);

        fprintf(stderr, "error during start: code %d, details: %s\n", code, details);

        OVMS_StatusDelete(res);

        OVMS_ServerDelete(srv);
        OVMS_ServerMultiModelOptionsDelete(mmo);
        OVMS_ServerGeneralOptionsDelete(go);
        return 1;
    }

    fprintf(stdout, "Server ready for inference\n");

    // infer 1
    // infer 2
    // infer 3

    // Application loop if required (C++):
    // while (shutdown_request == 0) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // }

    fprintf(stdout, "No more job to be done, will shut down\n");

    OVMS_ServerDelete(srv);
    OVMS_ServerMultiModelOptionsDelete(mmo);
    OVMS_ServerGeneralOptionsDelete(go);

    fprintf(stdout, "main() exit\n");
    return 0;
}
