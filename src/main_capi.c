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

#include <stdio.h>
#include <string.h>

#include "ovms.h"

int main() {
    uint32_t major = 0, minor = 0;
    OVMS_ApiVersion(&major, &minor);
    printf("C-API Version: %d.%d\n", major, minor);

    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    OVMS_Server* srv;

    OVMS_ServerSettingsNew(&serverSettings);
    OVMS_ModelsSettingsNew(&modelsSettings);
    OVMS_ServerNew(&srv);

    OVMS_ServerSettingsSetGrpcPort(serverSettings, 11337);
    OVMS_ServerSettingsSetRestPort(serverSettings, 11338);

    OVMS_ServerSettingsSetLogLevel(serverSettings, OVMS_LOG_DEBUG);
    OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config.json");

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = 0;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);

        fprintf(stderr, "error during start: code %d, details: %s\n", code, details);

        OVMS_StatusDelete(res);

        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }

    printf("Server ready for inference\n");

    const int64_t SHAPE_N = 30;
    const int64_t SHAPE_C = 20;
    const int64_t SHAPE[2] = {SHAPE_N, SHAPE_C};
    const size_t NUM_ELEMENTS = SHAPE_N * SHAPE_C;
    const size_t DATA_SIZE = NUM_ELEMENTS * sizeof(float);
    const float INPUT_ELEMENT_VALUE = 3.2f;

    float inputData[DATA_SIZE];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        inputData[i] = INPUT_ELEMENT_VALUE;
    }

    OVMS_InferenceRequest* request = NULL;
    OVMS_InferenceRequestNew(&request, srv, "dummy", 1);
    OVMS_InferenceRequestAddInput(request, "b", OVMS_DATATYPE_FP32, SHAPE, 2);
    OVMS_InferenceRequestInputSetData(request, "b", inputData, DATA_SIZE, OVMS_BUFFERTYPE_CPU, 0);

    OVMS_InferenceResponse* response = NULL;
    res = OVMS_Inference(srv, request, &response);
    if (res) {
        uint32_t code = 0;
        const char* details = 0;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);

        fprintf(stderr, "error during inference: code %d, details: %s\n", code, details);

        OVMS_StatusDelete(res);

        OVMS_InferenceRequestDelete(request);

        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }

    const char* oName = NULL;  // not needed
    OVMS_DataType oType;  // not needed
    const int64_t* oShape;  // not needed
    size_t oDims;  // not needed
    const void* oData = NULL;
    size_t oNumBytes = 0;
    OVMS_BufferType oBuffType;  // not needed
    uint32_t oDeviceId;  // not needed
    OVMS_InferenceResponseGetOutput(response, 0, &oName, &oType, &oShape, &oDims, &oData, &oNumBytes, &oBuffType, &oDeviceId);

    float expectedOutput[DATA_SIZE];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        expectedOutput[i] = INPUT_ELEMENT_VALUE + 1.0f;
    }
    if (memcmp(oData, expectedOutput, DATA_SIZE) != 0) {
        fprintf(stderr, "output is not correct\n");

        OVMS_InferenceResponseDelete(response);
        OVMS_InferenceRequestDelete(request);

        OVMS_ServerDelete(srv);

        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    } else {
        printf("output is correct\n");
    }

    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);

    printf("No more job to be done, will shut down\n");

    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);

    printf("main() exit\n");
    return 0;
}
