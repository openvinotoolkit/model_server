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

#include <memory>
#include <string>

#include <gtest/gtest.h>

// TODO we should not include classes from OVMS here
// consider how to workaround test_utils
#include "../inferenceresponse.hpp"
#include "../pocapi.h"
#include "test_utils.hpp"

using namespace ovms;
using testing::ElementsAreArray;

static void testDefaultSingleModelOptions(MultiModelOptionsImpl* mmo) {
    EXPECT_EQ(mmo->modelName, "");
    EXPECT_EQ(mmo->modelPath, "");
    EXPECT_EQ(mmo->batchSize, "");
    EXPECT_EQ(mmo->shape, "");
    EXPECT_EQ(mmo->layout, "");
    EXPECT_EQ(mmo->modelVersionPolicy, "");
    EXPECT_EQ(mmo->nireq, 0);
    EXPECT_EQ(mmo->targetDevice, "");
    EXPECT_EQ(mmo->pluginConfig, "");
    EXPECT_EQ(mmo->stateful, std::nullopt);
    EXPECT_EQ(mmo->lowLatencyTransformation, std::nullopt);
    EXPECT_EQ(mmo->maxSequenceNumber, std::nullopt);
    EXPECT_EQ(mmo->idleSequenceCleanup, std::nullopt);
}

#define ASSERT_CAPI_STATUS_NULL(C_API_CALL)   \
    {                                         \
        auto* err = C_API_CALL;               \
        if (err != nullptr) {                 \
            const char* msg = nullptr;        \
            OVMS_StatusGetDetails(err, &msg); \
            std::string smsg(msg);            \
            OVMS_StatusDelete(err);           \
            ASSERT_TRUE(false) << smsg;       \
        }                                     \
    }

#define ASSERT_CAPI_STATUS_NOT_NULL(C_API_CALL) \
    {                                           \
        auto* err = C_API_CALL;                 \
        if (err != nullptr) {                   \
            OVMS_StatusDelete(err);             \
        } else {                                \
            ASSERT_NE(err, nullptr);            \
        }                                       \
    }

#define ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(C_API_CALL, EXPECTED_STATUS_CODE)                              \
    {                                                                                                          \
        auto* err = C_API_CALL;                                                                                \
        if (err != nullptr) {                                                                                  \
            uint32_t code = 0;                                                                                 \
            const char* details = nullptr;                                                                     \
            ASSERT_EQ(OVMS_StatusGetCode(err, &code), nullptr);                                                \
            ASSERT_EQ(OVMS_StatusGetDetails(err, &details), nullptr);                                          \
            ASSERT_NE(details, nullptr);                                                                       \
            ASSERT_EQ(code, static_cast<uint32_t>(EXPECTED_STATUS_CODE))                                       \
                << std::string{"wrong code: "} + std::to_string(code) + std::string{"; details: "} << details; \
            OVMS_StatusDelete(err);                                                                            \
        } else {                                                                                               \
            ASSERT_NE(err, nullptr);                                                                           \
        }                                                                                                      \
    }

TEST(CApiConfigTest, MultiModelConfiguration) {
    OVMS_ServerGeneralOptions* _go = 0;
    OVMS_ServerMultiModelOptions* _mmo = 0;

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsNew(nullptr), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsNew(&_go));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMultiModelOptionsNew(nullptr), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsNew(&_mmo));

    ASSERT_NE(_go, nullptr);
    ASSERT_NE(_mmo, nullptr);

    GeneralOptionsImpl* go = reinterpret_cast<GeneralOptionsImpl*>(_go);
    MultiModelOptionsImpl* mmo = reinterpret_cast<MultiModelOptionsImpl*>(_mmo);

    // Test default values
    EXPECT_EQ(go->grpcPort, 9178);
    EXPECT_EQ(go->restPort, 0);
    EXPECT_EQ(go->grpcWorkers, 1);
    EXPECT_EQ(go->grpcBindAddress, "0.0.0.0");
    EXPECT_EQ(go->restWorkers, std::nullopt);
    EXPECT_EQ(go->restBindAddress, "0.0.0.0");
    EXPECT_EQ(go->metricsEnabled, false);
    EXPECT_EQ(go->metricsList, "");
    EXPECT_EQ(go->cpuExtensionLibraryPath, "");
    EXPECT_EQ(go->logLevel, "INFO");
    EXPECT_EQ(go->logPath, "");
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(go->grpcChannelArguments, "");
    EXPECT_EQ(go->filesystemPollWaitSeconds, 1);
    EXPECT_EQ(go->sequenceCleanerPollWaitMinutes, 5);
    EXPECT_EQ(go->resourcesCleanerPollWaitSeconds, 1);
    EXPECT_EQ(go->cacheDir, "");

    testDefaultSingleModelOptions(mmo);
    EXPECT_EQ(mmo->configPath, "");

    // Set non default values
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcPort(_go, 5555));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetRestPort(_go, 6666));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcWorkers(_go, 30));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcBindAddress(_go, "2.2.2.2"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetRestWorkers(_go, 31));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetRestBindAddress(_go, "3.3.3.3"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcChannelArguments(_go, "grpcargs"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(_go, 2));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(_go, 3));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerIntervalSeconds(_go, 4));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetCpuExtensionPath(_go, "/ovms/src/test"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetCacheDir(_go, "/tmp/cache"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogLevel(_go, OVMS_LOG_INFO));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogLevel(_go, OVMS_LOG_ERROR));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogLevel(_go, OVMS_LOG_DEBUG));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogLevel(_go, OVMS_LOG_WARNING));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogLevel(_go, OVMS_LOG_TRACE));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetLogLevel(_go, static_cast<OVMS_LogLevel>(99)), StatusCode::NONEXISTENT_LOG_LEVEL);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetLogPath(_go, "/logs"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsSetConfigPath(_mmo, "/config"));
    // check nullptr
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcPort(nullptr, 5555), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetRestPort(nullptr, 6666), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcWorkers(nullptr, 30), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcBindAddress(nullptr, "2.2.2.2"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcBindAddress(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetRestWorkers(nullptr, 31), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetRestBindAddress(nullptr, "3.3.3.3"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetRestBindAddress(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcChannelArguments(nullptr, "grpcargs"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetGrpcChannelArguments(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetFileSystemPollWaitSeconds(nullptr, 2), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetSequenceCleanerPollWaitMinutes(nullptr, 3), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetCustomNodeResourcesCleanerIntervalSeconds(nullptr, 4), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetCpuExtensionPath(nullptr, "/ovms/src/test"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetCpuExtensionPath(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetCacheDir(nullptr, "/tmp/cache"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetCacheDir(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetLogLevel(nullptr, OVMS_LOG_TRACE), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetLogPath(nullptr, "/logs"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsSetLogPath(_go, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMultiModelOptionsSetConfigPath(nullptr, "/config"), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMultiModelOptionsSetConfigPath(_mmo, nullptr), StatusCode::NONEXISTENT_STRING);

    // Test non default values
    EXPECT_EQ(go->grpcPort, 5555);
    EXPECT_EQ(go->restPort, 6666);
    EXPECT_EQ(go->grpcWorkers, 30);
    EXPECT_EQ(go->grpcBindAddress, "2.2.2.2");
    EXPECT_EQ(go->restWorkers, 31);
    EXPECT_EQ(go->restBindAddress, "3.3.3.3");
    // EXPECT_EQ(go->metricsEnabled, false);  // TODO: enable testing once metrics will be configurable via api
    // EXPECT_EQ(go->metricsList, "");
    EXPECT_EQ(go->cpuExtensionLibraryPath, "/ovms/src/test");
    EXPECT_EQ(go->logLevel, "TRACE");
    EXPECT_EQ(go->logPath, "/logs");
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(go->grpcChannelArguments, "grpcargs");
    EXPECT_EQ(go->filesystemPollWaitSeconds, 2);
    EXPECT_EQ(go->sequenceCleanerPollWaitMinutes, 3);
    EXPECT_EQ(go->resourcesCleanerPollWaitSeconds, 4);
    EXPECT_EQ(go->cacheDir, "/tmp/cache");

    testDefaultSingleModelOptions(mmo);
    EXPECT_EQ(mmo->configPath, "/config");

    // Test config parser
    ConstructorEnabledConfig cfg;
    ASSERT_TRUE(cfg.parse(go, mmo));
    EXPECT_EQ(cfg.port(), 5555);
    EXPECT_EQ(cfg.restPort(), 6666);
    EXPECT_EQ(cfg.grpcWorkers(), 30);
    EXPECT_EQ(cfg.grpcBindAddress(), "2.2.2.2");
    EXPECT_EQ(cfg.restWorkers(), 31);
    EXPECT_EQ(cfg.restBindAddress(), "3.3.3.3");
    // EXPECT_EQ(go->metricsEnabled, false);  // TODO: enable testing once metrics will be configurable via api
    // EXPECT_EQ(go->metricsList, "");
    EXPECT_EQ(cfg.cpuExtensionLibraryPath(), "/ovms/src/test");
    EXPECT_EQ(cfg.logLevel(), "TRACE");
    EXPECT_EQ(cfg.logPath(), "/logs");
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(cfg.grpcChannelArguments(), "grpcargs");
    EXPECT_EQ(cfg.filesystemPollWaitSeconds(), 2);
    EXPECT_EQ(cfg.sequenceCleanerPollWaitMinutes(), 3);
    EXPECT_EQ(cfg.resourcesCleanerPollWaitSeconds(), 4);
    EXPECT_EQ(cfg.cacheDir(), "/tmp/cache");

    EXPECT_EQ(cfg.modelName(), "");
    EXPECT_EQ(cfg.modelPath(), "");
    EXPECT_EQ(cfg.batchSize(), "0");
    EXPECT_EQ(cfg.shape(), "");
    EXPECT_EQ(cfg.layout(), "");
    EXPECT_EQ(cfg.modelVersionPolicy(), "");
    EXPECT_EQ(cfg.nireq(), 0);
    EXPECT_EQ(cfg.targetDevice(), "CPU");
    EXPECT_EQ(cfg.pluginConfig(), "");
    EXPECT_FALSE(cfg.stateful());
    EXPECT_FALSE(cfg.lowLatencyTransformation());
    EXPECT_EQ(cfg.maxSequenceNumber(), DEFAULT_MAX_SEQUENCE_NUMBER);
    EXPECT_TRUE(cfg.idleSequenceCleanup());

    EXPECT_EQ(cfg.configPath(), "/config");

    OVMS_ServerMultiModelOptionsDelete(nullptr);
    OVMS_ServerMultiModelOptionsDelete(_mmo);
    OVMS_ServerGeneralOptionsDelete(nullptr);
    OVMS_ServerGeneralOptionsDelete(_go);
}

TEST(CApiConfigTest, SingleModelConfiguration) {
    GTEST_SKIP() << "Use C-API to initialize in next stages, currently not supported";
}

TEST(CApiStartTest, InitializingMultipleServers) {
    OVMS_Server* srv1 = nullptr;
    OVMS_Server* srv2 = nullptr;

    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv1));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv2));
    ASSERT_EQ(srv1, srv2);
    OVMS_ServerDelete(srv1);
}

TEST(CApiStartTest, StartFlow) {
    OVMS_Server* srv = 0;
    OVMS_ServerGeneralOptions* go = 0;
    OVMS_ServerMultiModelOptions* mmo = 0;

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerNew(nullptr), StatusCode::NONEXISTENT_SERVER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerGeneralOptionsNew(nullptr), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMultiModelOptionsNew(nullptr), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsNew(&go));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsNew(&mmo));

    ASSERT_NE(srv, nullptr);
    ASSERT_NE(go, nullptr);
    ASSERT_NE(mmo, nullptr);

    // Cannot start due to configuration error
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcPort(go, 5555));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetRestPort(go, 5555));  // The same port
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsSetConfigPath(mmo, "/ovms/src/test/c_api/config.json"));

    // Expect fail
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(srv, go, mmo),
        StatusCode::OPTIONS_USAGE_ERROR);

    // Fix and expect ok
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetRestPort(go, 6666));  // Different port
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(srv, go, mmo));

    // Try to start again, expect failure
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(srv, go, mmo),
        StatusCode::SERVER_ALREADY_STARTED);

    // TODO: Is infer ok?

    OVMS_ServerMultiModelOptionsDelete(mmo);
    OVMS_ServerGeneralOptionsDelete(go);
    OVMS_ServerDelete(srv);
}

TEST(CApiStatusTest, GetCodeAndDetails) {
    std::unique_ptr<Status> s = std::make_unique<Status>(
        StatusCode::INTERNAL_ERROR, "custom message");
    OVMS_Status* sts = reinterpret_cast<OVMS_Status*>(s.get());
    uint32_t code = 0;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetCode(nullptr, &code), StatusCode::NONEXISTENT_STATUS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetCode(sts, nullptr), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusGetCode(sts, &code));
    EXPECT_EQ(code, static_cast<uint32_t>(StatusCode::INTERNAL_ERROR));
    const char* details = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetDetails(nullptr, &details), StatusCode::NONEXISTENT_STATUS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetDetails(sts, nullptr), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusGetDetails(sts, &details));
    std::stringstream ss;
    ss << Status(StatusCode::INTERNAL_ERROR).string() << " - custom message";
    EXPECT_EQ(std::string(details), ss.str());
    OVMS_StatusDelete(reinterpret_cast<OVMS_Status*>(s.release()));
}

class CapiInference : public ::testing::Test {};

TEST_F(CapiInference, Basic) {
    // request creation
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, "dummy", 1));
    ASSERT_NE(nullptr, request);

    // adding input
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    // add parameters
    const uint64_t sequenceId{42};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)));
    // 2nd time should get error
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::DOUBLE_PARAMETER_INSERT);
    const uint32_t sequenceControl{1};  // SEQUENCE_START
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_control_input", OVMS_DATATYPE_U32, reinterpret_cast<const void*>(&sequenceControl), sizeof(sequenceControl)));

    //////////////////
    //  INFERENCE
    //////////////////
    // remove when C-API start implemented
    std::string port = "9000";
    randomizePort(port);
    // prepare options
    OVMS_ServerGeneralOptions* go = 0;
    OVMS_ServerMultiModelOptions* mmo = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsNew(&go));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsNew(&mmo));
    ASSERT_NE(go, nullptr);
    ASSERT_NE(mmo, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcPort(go, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsSetConfigPath(mmo, "/ovms/src/test/c_api/config_standard_dummy.json"));

    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, go, mmo));
    ASSERT_NE(cserver, nullptr);

    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // verify GetOutputCount
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutputCount(nullptr, &outputCount), StatusCode::NONEXISTENT_RESPONSE);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutputCount(response, nullptr), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameterCount(nullptr, &parameterCount), StatusCode::NONEXISTENT_RESPONSE);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameterCount(response, nullptr), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const uint64_t* shape{nullptr};
    uint32_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(nullptr, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_RESPONSE);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, nullptr, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, nullptr, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, nullptr, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_TABLE);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, nullptr, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, nullptr, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_DATA);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, nullptr, &bufferType, &deviceId), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, nullptr, &deviceId), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, nullptr), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);

    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }

    ///////////////
    // CLEANUP
    ///////////////
    // cleanup response
    OVMS_InferenceResponseDelete(response);
    // cleanup request
    // here we will add additional inputs to verify 2 ways of cleanup
    // - direct call to remove whole request
    // - separate calls to remove partial data
    //
    // here we will just add inputs to remove them later
    // one original will be removed together with buffer during whole request removal
    // one will be removed together with request but without buffer attached
    // one will be removed with buffer directly
    // one will be removed without buffer directly
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_WITH_REQUEST", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    // we will add buffer and remove it to check separate buffer removal
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));

    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputRemoveData(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY"));
    // second time we should get error
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputRemoveData(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY"), StatusCode::NONEXISTENT_BUFFER_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveInput(request, "INPUT_WITHOUT_BUFFER_REMOVED_DIRECTLY"));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveInput(request, "INPUT_WITH_BUFFER_REMOVED_DIRECTLY"));
    // we will remove 1 of two parameters
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveParameter(request, "sequence_id"));
    // 2nd time should report error
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(request, "sequence_id"), StatusCode::NONEXISTENT_PARAMETER_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(nullptr, "sequence_id"), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(request, nullptr), StatusCode::NONEXISTENT_STRING);

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(request, "NONEXISTENT_TENSOR"), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(nullptr, "INPUT_WITHOUT_BUFFER_REMOVED_WITH_REQUEST"), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(request, nullptr), StatusCode::NONEXISTENT_STRING);
    OVMS_InferenceRequestDelete(nullptr);
    OVMS_InferenceRequestDelete(request);

    OVMS_ServerDelete(cserver);
}

TEST_F(CapiInference, NegativeInference) {
    // first start OVMS
    std::string port = "9000";
    randomizePort(port);
    // prepare options
    OVMS_ServerGeneralOptions* go = 0;
    OVMS_ServerMultiModelOptions* mmo = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsNew(&go));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsNew(&mmo));
    ASSERT_NE(go, nullptr);
    ASSERT_NE(mmo, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerGeneralOptionsSetGrpcPort(go, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMultiModelOptionsSetConfigPath(mmo, "/ovms/src/test/c_api/config_standard_dummy.json"));

    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(nullptr, go, mmo), StatusCode::NONEXISTENT_SERVER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(cserver, nullptr, mmo), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(cserver, go, nullptr), StatusCode::NONEXISTENT_OPTIONS);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, go, mmo));
    // TODO add check for server ready strict in 2022.3 not available in C-API

    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestNew(nullptr, "dummy", 1), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestNew(&request, nullptr, 1), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, "dummy", 1));
    ASSERT_NE(nullptr, request);
    // negative no inputs
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_NO_OF_INPUTS);

    // negative no input buffer
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(nullptr, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, nullptr, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, nullptr, DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_TABLE);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    // fail with adding input second time
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::DOUBLE_TENSOR_INSERT);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_CONTENT_SIZE);

    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(nullptr, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, nullptr, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, nullptr, sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_DATA);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, "NONEXISTENT_TENSOR", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_TENSOR_FOR_SET_BUFFER);
    // add parameters
    const uint64_t sequenceId{42};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(nullptr, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, nullptr, OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::NONEXISTENT_STRING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, nullptr, sizeof(sequenceId)), StatusCode::NONEXISTENT_DATA);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)));
    // 2nd time should get error
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::DOUBLE_PARAMETER_INSERT);
    const uint32_t sequenceControl{1};  // SEQUENCE_START
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_control_input", OVMS_DATATYPE_U32, reinterpret_cast<const void*>(&sequenceControl), sizeof(sequenceControl)));

    // verify passing nullptrs
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(nullptr, request, &response), StatusCode::NONEXISTENT_SERVER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, nullptr, &response), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, nullptr), StatusCode::NONEXISTENT_RESPONSE);

    // negative inference with non existing model
    OVMS_InferenceRequest* requestNoModel{nullptr};
    OVMS_InferenceResponse* reponseNoModel{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, "NONEXISTENT_MODEL", 13));
    // negative no model
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::NOT_IMPLEMENTED);

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(nullptr, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_REQUEST);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, requestNoModel, &reponseNoModel), StatusCode::NONEXISTENT_REQUEST);
    OVMS_InferenceRequestDelete(requestNoModel);

    OVMS_ServerDelete(nullptr);
    OVMS_ServerDelete(cserver);
    OVMS_ServerDelete(nullptr);
}

// TODO negative test -> validate at the infer stage
// TODO reuse request after inference
namespace {
const std::string MODEL_NAME{"SomeModelName"};
const uint64_t MODEL_VERSION{42};
const std::string PARAMETER_NAME{"sequence_id"};  // TODO check if in ovms there is such constant
const OVMS_DataType PARAMETER_DATATYPE{OVMS_DATATYPE_I32};

const uint32_t PARAMETER_VALUE{13};
const uint32_t PRIORITY{7};
const uint64_t REQUEST_ID{3};

const std::string INPUT_NAME{"NOT_RANDOM_NAME"};
const shape_t INPUT_SHAPE{1, 3, 220, 230};
const std::array<float, DUMMY_MODEL_INPUT_SIZE> INPUT_DATA{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
constexpr size_t INPUT_DATA_BYTESIZE{INPUT_DATA.size() * sizeof(float)};
const OVMS_DataType DATATYPE{OVMS_DATATYPE_FP32};
}  // namespace

TEST_F(CapiInference, ResponseRetrieval) {
    auto cppResponse = std::make_unique<InferenceResponse>(MODEL_NAME, MODEL_VERSION);
    // add output
    std::array<size_t, 2> cppOutputShape{1, DUMMY_MODEL_INPUT_SIZE};
    auto cppStatus = cppResponse->addOutput(INPUT_NAME.c_str(), DATATYPE, cppOutputShape.data(), cppOutputShape.size());
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    InferenceTensor* cpptensor = nullptr;
    const std::string* cppOutputName;
    cppStatus = cppResponse->getOutput(0, &cppOutputName, &cpptensor);
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();

    // save data into output (it should have it's own copy in contrast to request)
    bool createCopy = true;
    cppStatus = cpptensor->setBuffer(INPUT_DATA.data(), INPUT_DATA_BYTESIZE, OVMS_BUFFERTYPE_CPU, std::nullopt, createCopy);
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    // add parameter to response
    uint64_t seqId = 666;
    cppStatus = cppResponse->addParameter("sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<void*>(&seqId));
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    ///////////////////////////
    // now response is prepared so we can test C-API
    ///////////////////////////
    OVMS_InferenceResponse* response = reinterpret_cast<OVMS_InferenceResponse*>(cppResponse.get());
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);

    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
    ASSERT_EQ(1, parameterCount);
    // verify get Parameter
    OVMS_DataType parameterDatatype = OVMS_DATATYPE_FP32;
    const void* parameterData{nullptr};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(nullptr, 0, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_RESPONSE);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 0, nullptr, &parameterData), StatusCode::NONEXISTENT_NUMBER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 0, &parameterDatatype, nullptr), StatusCode::NONEXISTENT_DATA);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameter(response, 0, &parameterDatatype, &parameterData));
    ASSERT_EQ(parameterDatatype, OVMS_DATATYPE_U64);
    EXPECT_EQ(0, std::memcmp(parameterData, (void*)&seqId, sizeof(seqId)));
    // verify get Output
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const uint64_t* shape{nullptr};
    uint32_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId + 42123, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(INPUT_NAME, outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);

    for (size_t i = 0; i < cppOutputShape.size(); ++i) {
        EXPECT_EQ(cppOutputShape[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < INPUT_DATA.size(); ++i) {
        EXPECT_EQ(INPUT_DATA[i], outputData[i]) << "Different at:" << i << " place.";
    }

    // test negative scenario with getting output without buffer
    cppStatus = cppResponse->addOutput("outputWithNoBuffer", DATATYPE, cppOutputShape.data(), cppOutputShape.size());
    ASSERT_EQ(cppStatus, StatusCode::OK) << cppStatus.string();
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId + 1, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::INTERNAL_ERROR);
    // negative scenario nonexistsing parameter
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 123, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_PARAMETER_FOR_REMOVAL);
    // final cleanup
    // we release unique_ptr ownership here so that we can free it safely via C-API
    cppResponse.release();
    OVMS_InferenceResponseDelete(nullptr);
    OVMS_InferenceResponseDelete(response);
}
// TODO make cleaner error codes reporting
// todo decide either use remove or delete for consistency
