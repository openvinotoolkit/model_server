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
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <signal.h>
#include <stdio.h>

#include "ovms.h"  // NOLINT

const char* MODEL_NAME = "dummy";
const int64_t MODEL_VERSION = 1;
const char* INPUT_NAME = "b";
constexpr size_t DIM_COUNT = 2;
constexpr int64_t SHAPE[DIM_COUNT] = {1, 10};

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

    uint32_t major = 0, minor = 0;
    OVMS_ApiVersion(&major, &minor);
    std::cout << "C-API Version: " << major << "." << minor << std::endl;

    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    OVMS_Server* srv;

    OVMS_ServerSettingsNew(&serverSettings);
    OVMS_ModelsSettingsNew(&modelsSettings);
    OVMS_ServerNew(&srv);

    OVMS_ServerSettingsSetGrpcPort(serverSettings, 9178);
    OVMS_ServerSettingsSetRestPort(serverSettings, 11338);

    OVMS_ServerSettingsSetLogLevel(serverSettings, OVMS_LOG_DEBUG);
    OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_standard_dummy.json");

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cerr << "error during start: code:" << code << "; details:" << details << std::endl;

        OVMS_StatusDelete(res);

        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }

    std::cout << "Server ready for inference" << std::endl;

    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, srv, MODEL_NAME, MODEL_VERSION);
    OVMS_InferenceRequestAddInput(request, INPUT_NAME, OVMS_DATATYPE_FP32, SHAPE, DIM_COUNT);
    std::array<float, SHAPE[1]> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    OVMS_InferenceRequestInputSetData(request, INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, 0);

    // run sync request
    OVMS_InferenceResponse* response = nullptr;
    res = OVMS_Inference(srv, request, &response);
    if (res != nullptr) {
        uint32_t code = 0;
        const char* details = 0;
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cout << "Error occured during inference. Code:" << code
                  << ", details:" << details << std::endl;
        OVMS_StatusDelete(res);
        OVMS_InferenceRequestDelete(request);
        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }
    // read output
    uint32_t outputCount = 0;
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    const void* voutputData;
    size_t bytesize = 0;
    uint32_t outputId = outputCount - 1;
    OVMS_DataType datatype = (OVMS_DataType)42;
    const int64_t* shape{nullptr};
    size_t dimCount = 0;
    OVMS_BufferType bufferType = (OVMS_BufferType)42;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId);

    std::stringstream ss;
    ss << "Got response from OVMS via C-API. "
       << "Request for model: " << MODEL_NAME
       << "; version: " << MODEL_VERSION
       << "ms; output name: " << outputName
       << "; response with values:\n";
    for (size_t i = 0; i < shape[1]; ++i) {
        ss << *(reinterpret_cast<const float*>(voutputData) + i) << " ";
    }
    std::vector<float> expectedOutput;
    std::transform(data.begin(), data.end(), std::back_inserter(expectedOutput),
        [](const float& s) -> float {
            return s + 1;
        });

    if (std::memcmp(voutputData, expectedOutput.data(), expectedOutput.size() * sizeof(float)) != 0) {
        std::cout << "Incorrect result of inference" << std::endl;
    }
    // comment line below to have app running similarly to OVMS
    shutdown_request = 1;
    while (shutdown_request == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "No more job to be done, will shut down" << std::endl;

    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);

    fprintf(stdout, "main() exit\n");
    return 0;
}
