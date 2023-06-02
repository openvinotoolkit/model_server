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
#include <thread>

#include <gtest/gtest.h>

#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../ovms.h"
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;
using testing::ElementsAreArray;

static void testDefaultSingleModelOptions(ModelsSettingsImpl* modelsSettings) {
    EXPECT_EQ(modelsSettings->modelName, "");
    EXPECT_EQ(modelsSettings->modelPath, "");
    EXPECT_EQ(modelsSettings->batchSize, "");
    EXPECT_EQ(modelsSettings->shape, "");
    EXPECT_EQ(modelsSettings->layout, "");
    EXPECT_EQ(modelsSettings->modelVersionPolicy, "");
    EXPECT_EQ(modelsSettings->nireq, 0);
    EXPECT_EQ(modelsSettings->targetDevice, "");
    EXPECT_EQ(modelsSettings->pluginConfig, "");
    EXPECT_EQ(modelsSettings->stateful, std::nullopt);
    EXPECT_EQ(modelsSettings->lowLatencyTransformation, std::nullopt);
    EXPECT_EQ(modelsSettings->maxSequenceNumber, std::nullopt);
    EXPECT_EQ(modelsSettings->idleSequenceCleanup, std::nullopt);
}

const uint AVAILABLE_CORES = std::thread::hardware_concurrency();

TEST(CAPIConfigTest, MultiModelConfiguration) {
    OVMS_ServerSettings* _serverSettings = nullptr;
    OVMS_ModelsSettings* _modelsSettings = nullptr;

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsNew(nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&_serverSettings));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ModelsSettingsNew(nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&_modelsSettings));

    ASSERT_NE(_serverSettings, nullptr);
    ASSERT_NE(_modelsSettings, nullptr);

    ServerSettingsImpl* serverSettings = reinterpret_cast<ServerSettingsImpl*>(_serverSettings);
    ModelsSettingsImpl* modelsSettings = reinterpret_cast<ModelsSettingsImpl*>(_modelsSettings);

    // Test default values
    EXPECT_EQ(serverSettings->grpcPort, 9178);
    EXPECT_EQ(serverSettings->restPort, 0);
    EXPECT_EQ(serverSettings->grpcWorkers, 1);
    EXPECT_EQ(serverSettings->grpcBindAddress, "0.0.0.0");
    EXPECT_EQ(serverSettings->restWorkers, std::nullopt);
    EXPECT_EQ(serverSettings->restBindAddress, "0.0.0.0");
    EXPECT_EQ(serverSettings->metricsEnabled, false);
    EXPECT_EQ(serverSettings->metricsList, "");
    EXPECT_EQ(serverSettings->cpuExtensionLibraryPath, "");
    EXPECT_EQ(serverSettings->logLevel, "INFO");
    EXPECT_EQ(serverSettings->logPath, "");
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(serverSettings->grpcChannelArguments, "");
    EXPECT_EQ(serverSettings->filesystemPollWaitSeconds, 1);
    EXPECT_EQ(serverSettings->sequenceCleanerPollWaitMinutes, 5);
    EXPECT_EQ(serverSettings->resourcesCleanerPollWaitSeconds, 1);
    EXPECT_EQ(serverSettings->cacheDir, "");

    testDefaultSingleModelOptions(modelsSettings);
    EXPECT_EQ(modelsSettings->configPath, "");

    // Set non default values
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(_serverSettings, 5555));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(_serverSettings, 6666));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcWorkers(_serverSettings, AVAILABLE_CORES));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcBindAddress(_serverSettings, "2.2.2.2"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestWorkers(_serverSettings, 31));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestBindAddress(_serverSettings, "3.3.3.3"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcChannelArguments(_serverSettings, "grpcargs"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(_serverSettings, 2));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(_serverSettings, 3));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(_serverSettings, 4));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCpuExtensionPath(_serverSettings, "/ovms/src/test"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCacheDir(_serverSettings, "/tmp/cache"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_INFO));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_ERROR));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_DEBUG));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_WARNING));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_TRACE));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogLevel(_serverSettings, static_cast<OVMS_LogLevel>(99)), StatusCode::NONEXISTENT_LOG_LEVEL);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogPath(_serverSettings, "/logs"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(_modelsSettings, "/config"));
    // check nullptr
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcPort(nullptr, 5555), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetRestPort(nullptr, 6666), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcWorkers(nullptr, 30), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcBindAddress(nullptr, "2.2.2.2"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcBindAddress(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetRestWorkers(nullptr, 31), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetRestBindAddress(nullptr, "3.3.3.3"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetRestBindAddress(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcChannelArguments(nullptr, "grpcargs"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcChannelArguments(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(nullptr, 2), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(nullptr, 3), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(nullptr, 4), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCpuExtensionPath(nullptr, "/ovms/src/test"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCpuExtensionPath(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCacheDir(nullptr, "/tmp/cache"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCacheDir(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogLevel(nullptr, OVMS_LOG_TRACE), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogPath(nullptr, "/logs"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogPath(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ModelsSettingsSetConfigPath(nullptr, "/config"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ModelsSettingsSetConfigPath(_modelsSettings, nullptr), StatusCode::NONEXISTENT_PTR);

    // Test non default values
    EXPECT_EQ(serverSettings->grpcPort, 5555);
    EXPECT_EQ(serverSettings->restPort, 6666);
    EXPECT_EQ(serverSettings->grpcWorkers, AVAILABLE_CORES);
    EXPECT_EQ(serverSettings->grpcBindAddress, "2.2.2.2");
    EXPECT_EQ(serverSettings->restWorkers, 31);
    EXPECT_EQ(serverSettings->restBindAddress, "3.3.3.3");
    // EXPECT_EQ(serverSettings->metricsEnabled, false);
    // EXPECT_EQ(serverSettings->metricsList, "");
    EXPECT_EQ(serverSettings->cpuExtensionLibraryPath, "/ovms/src/test");
    EXPECT_EQ(serverSettings->logLevel, "TRACE");
    EXPECT_EQ(serverSettings->logPath, "/logs");
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(serverSettings->grpcChannelArguments, "grpcargs");
    EXPECT_EQ(serverSettings->filesystemPollWaitSeconds, 2);
    EXPECT_EQ(serverSettings->sequenceCleanerPollWaitMinutes, 3);
    EXPECT_EQ(serverSettings->resourcesCleanerPollWaitSeconds, 4);
    EXPECT_EQ(serverSettings->cacheDir, "/tmp/cache");

    testDefaultSingleModelOptions(modelsSettings);
    EXPECT_EQ(modelsSettings->configPath, "/config");

    // Test config parser
    ConstructorEnabledConfig cfg;
    ASSERT_TRUE(cfg.parse(serverSettings, modelsSettings));
    EXPECT_EQ(cfg.port(), 5555);
    EXPECT_EQ(cfg.restPort(), 6666);
    EXPECT_EQ(cfg.grpcWorkers(), AVAILABLE_CORES);
    EXPECT_EQ(cfg.grpcBindAddress(), "2.2.2.2");
    EXPECT_EQ(cfg.restWorkers(), 31);
    EXPECT_EQ(cfg.restBindAddress(), "3.3.3.3");
    // EXPECT_EQ(serverSettings->metricsEnabled, false);
    // EXPECT_EQ(serverSettings->metricsList, "");
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

    OVMS_ModelsSettingsDelete(nullptr);
    OVMS_ModelsSettingsDelete(_modelsSettings);
    OVMS_ServerSettingsDelete(nullptr);
    OVMS_ServerSettingsDelete(_serverSettings);
}

TEST(CAPIConfigTest, SingleModelConfiguration) {
    GTEST_SKIP() << "Use C-API to initialize in next stages, currently not supported";
}

TEST(CAPIStartTest, InitializingMultipleServers) {
    OVMS_Server* srv1 = nullptr;
    OVMS_Server* srv2 = nullptr;

    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv1));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv2));
    ASSERT_EQ(srv1, srv2);
    OVMS_ServerDelete(srv1);
}

TEST(CAPIStartTest, StartFlow) {
    OVMS_Server* srv = nullptr;
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerNew(nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsNew(nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ModelsSettingsNew(nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&srv));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));

    ASSERT_NE(srv, nullptr);
    ASSERT_NE(serverSettings, nullptr);
    ASSERT_NE(modelsSettings, nullptr);

    // Cannot start due to configuration error
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, 5555));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, 5555));  // The same port
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config.json"));

    // Expect fail
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings),
        StatusCode::OPTIONS_USAGE_ERROR);

    // Fix and expect ok
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, 6666));  // Different port
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings));

    // Try to start again, expect failure
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings),
        StatusCode::SERVER_ALREADY_STARTED);

    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
    OVMS_ServerDelete(srv);
}

TEST(CAPIStatusTest, GetCodeAndDetails) {
    std::unique_ptr<Status> s = std::make_unique<Status>(
        StatusCode::INTERNAL_ERROR, "custom message");
    OVMS_Status* sts = reinterpret_cast<OVMS_Status*>(s.get());
    uint32_t code = 0;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetCode(nullptr, &code), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetCode(sts, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusGetCode(sts, &code));
    EXPECT_EQ(code, static_cast<uint32_t>(StatusCode::INTERNAL_ERROR));
    const char* details = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetDetails(nullptr, &details), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusGetDetails(sts, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusGetDetails(sts, &details));
    std::stringstream ss;
    ss << Status(StatusCode::INTERNAL_ERROR).string() << " - custom message";
    EXPECT_EQ(std::string(details), ss.str());
    OVMS_StatusDelete(reinterpret_cast<OVMS_Status*>(s.release()));
}

class CAPIInference : public ::testing::Test {};

TEST_F(CAPIInference, TensorSetMovedBuffer) {
    constexpr size_t elementsCount = 2;
    std::array<int64_t, elementsCount> shape{1, elementsCount};
    InferenceTensor tensor(OVMS_DATATYPE_FP32, shape.data(), shape.size());
    std::unique_ptr<Buffer> bufferNull;
    ASSERT_EQ(tensor.setBuffer(std::move(bufferNull)), ovms::StatusCode::OK);
    auto buffer = std::make_unique<Buffer>(sizeof(float) * elementsCount, OVMS_BUFFERTYPE_CPU, std::nullopt);
    ASSERT_EQ(tensor.setBuffer(std::move(buffer)), ovms::StatusCode::OK);
    auto buffer2 = std::make_unique<Buffer>(sizeof(float) * elementsCount, OVMS_BUFFERTYPE_CPU, std::nullopt);
    ASSERT_EQ(tensor.setBuffer(std::move(buffer)), ovms::StatusCode::DOUBLE_BUFFER_SET);
}

TEST_F(CAPIInference, Basic) {
    //////////////////////
    // start server
    //////////////////////
    // remove when C-API start implemented
    std::string port = "9000";
    randomizePort(port);
    // prepare options
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_NE(serverSettings, nullptr);
    ASSERT_NE(modelsSettings, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_standard_dummy.json"));

    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    ASSERT_NE(cserver, nullptr);
    ///////////////////////
    // request creation
    ///////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
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

    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // verify GetOutputCount
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutputCount(nullptr, &outputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutputCount(response, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameterCount(nullptr, &parameterCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameterCount(response, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(nullptr, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, nullptr, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, nullptr, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, nullptr, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, nullptr, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, nullptr, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, nullptr, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, nullptr, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, nullptr), StatusCode::NONEXISTENT_PTR);
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(request, "sequence_id"), StatusCode::NONEXISTENT_PARAMETER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(nullptr, "sequence_id"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveParameter(request, nullptr), StatusCode::NONEXISTENT_PTR);

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(request, "NONEXISTENT_TENSOR"), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(nullptr, "INPUT_WITHOUT_BUFFER_REMOVED_WITH_REQUEST"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveInput(request, nullptr), StatusCode::NONEXISTENT_PTR);
    OVMS_InferenceRequestDelete(nullptr);
    OVMS_InferenceRequestDelete(request);

    OVMS_ServerDelete(cserver);
}

TEST_F(CAPIInference, NegativeInference) {
    // first start OVMS
    std::string port = "9000";
    randomizePort(port);
    // prepare options
    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_NE(serverSettings, nullptr);
    ASSERT_NE(modelsSettings, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_standard_dummy.json"));

    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(nullptr, serverSettings, modelsSettings), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(cserver, nullptr, modelsSettings), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));

    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestNew(nullptr, cserver, "dummy", 1), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestNew(&request, cserver, nullptr, 1), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestNew(&request, nullptr, "dummy", 1), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, request);
    // negative no inputs
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_NO_OF_INPUTS);

    // negative no input buffer
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(nullptr, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, nullptr, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, nullptr, DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    // fail with adding input second time
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::DOUBLE_TENSOR_INSERT);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_CONTENT_SIZE);

    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(nullptr, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, nullptr, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, nullptr, sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, "NONEXISTENT_TENSOR", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_TENSOR_FOR_SET_BUFFER);
    // add parameters
    const uint64_t sequenceId{42};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(nullptr, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, nullptr, OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, nullptr, sizeof(sequenceId)), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)));
    // 2nd time should get error
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddParameter(request, "sequence_id", OVMS_DATATYPE_U64, reinterpret_cast<const void*>(&sequenceId), sizeof(sequenceId)), StatusCode::DOUBLE_PARAMETER_INSERT);
    const uint32_t sequenceControl{1};  // SEQUENCE_START
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddParameter(request, "sequence_control_input", OVMS_DATATYPE_U32, reinterpret_cast<const void*>(&sequenceControl), sizeof(sequenceControl)));

    // verify passing nullptrs
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(nullptr, request, &response), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, nullptr, &response), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, nullptr), StatusCode::NONEXISTENT_PTR);

    // negative inference with non existing model
    OVMS_InferenceRequest* requestNoModel{nullptr};
    OVMS_InferenceResponse* reponseNoModel{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "NONEXISTENT_MODEL", 13));
    // negative no model
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::PIPELINE_DEFINITION_NAME_MISSING);

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddInput(nullptr, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, requestNoModel, &reponseNoModel), StatusCode::NONEXISTENT_PTR);
    OVMS_InferenceRequestDelete(requestNoModel);

    OVMS_ServerDelete(nullptr);
    OVMS_ServerDelete(cserver);
    OVMS_ServerDelete(nullptr);
}

namespace {
const std::string MODEL_NAME{"SomeModelName"};
const int64_t MODEL_VERSION{42};
const std::string PARAMETER_NAME{"sequence_id"};
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

TEST_F(CAPIInference, ResponseRetrieval) {
    auto cppResponse = std::make_unique<InferenceResponse>(MODEL_NAME, MODEL_VERSION);
    // add output
    std::array<int64_t, 2> cppOutputShape{1, DUMMY_MODEL_INPUT_SIZE};
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(nullptr, 0, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 0, nullptr, &parameterData), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 0, &parameterDatatype, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameter(response, 0, &parameterDatatype, &parameterData));
    ASSERT_EQ(parameterDatatype, OVMS_DATATYPE_U64);
    EXPECT_EQ(0, std::memcmp(parameterData, (void*)&seqId, sizeof(seqId)));
    // verify get Output
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseGetParameter(response, 123, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_PARAMETER);
    // final cleanup
    // we release unique_ptr ownership here so that we can free it safely via C-API
    cppResponse.release();
    OVMS_InferenceResponseDelete(nullptr);
    OVMS_InferenceResponseDelete(response);
}

class CAPIMetadata : public ::testing::Test {
protected:
    static OVMS_Server* cserver;

public:
    static void SetUpTestSuite() {
        std::string port = "9000";
        randomizePort(port);
        // prepare options
        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
        ASSERT_NE(serverSettings, nullptr);
        ASSERT_NE(modelsSettings, nullptr);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_metadata_all.json"));
        cserver = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
    }
    static void TearDownTestSuite() {
        OVMS_ServerDelete(cserver);
        cserver = nullptr;
    }
    void checkMetadata(const std::string& servableName,
        int64_t servableVersion,
        const tensor_map_t& expectedInputsInfo,
        const tensor_map_t& expectedOutputsInfo) {
        OVMS_ServableMetadata* servableMetadata = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, servableName.c_str(), servableVersion, &servableMetadata));
        ASSERT_NE(nullptr, servableMetadata);
        uint32_t inputCount = 42;
        uint32_t outputCount = 42;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetInputCount(servableMetadata, &inputCount));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetOutputCount(servableMetadata, &outputCount));
        ASSERT_EQ(expectedInputsInfo.size(), inputCount);
        ASSERT_EQ(expectedOutputsInfo.size(), outputCount);

        uint32_t id = 0;
        OVMS_DataType datatype = (OVMS_DataType)199;
        int64_t* shapeMin{nullptr};
        int64_t* shapeMax{nullptr};
        size_t dimCount = 42;
        const char* tensorName{nullptr};
        std::set<std::string> inputNames;
        std::set<std::string> outputNames;
        for (id = 0; id < inputCount; ++id) {
            ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
            auto it = expectedInputsInfo.find(tensorName);
            ASSERT_NE(it, expectedInputsInfo.end());
            inputNames.insert(tensorName);
            EXPECT_EQ(datatype, ovms::getPrecisionAsOVMSDataType(it->second->getPrecision()));
            auto& expectedShape = it->second->getShape();
            ASSERT_EQ(expectedShape.size(), dimCount);
            for (size_t i = 0; i < expectedShape.size(); ++i) {
                EXPECT_EQ(expectedShape[i], ovms::Dimension(shapeMin[i], shapeMax[i]));
            }
        }
        EXPECT_EQ(inputNames.size(), inputCount);
        for (id = 0; id < outputCount; ++id) {
            ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
            auto it = expectedOutputsInfo.find(tensorName);
            ASSERT_NE(it, expectedOutputsInfo.end());
            outputNames.insert(tensorName);
            EXPECT_EQ(datatype, ovms::getPrecisionAsOVMSDataType(it->second->getPrecision()));
            auto& expectedShape = it->second->getShape();
            ASSERT_EQ(expectedShape.size(), dimCount);
            for (size_t i = 0; i < expectedShape.size(); ++i) {
                EXPECT_EQ(expectedShape[i], ovms::Dimension(shapeMin[i], shapeMax[i]));
            }
        }
        EXPECT_EQ(outputNames.size(), outputCount);
        const ov::AnyMap* servableMetadataRtInfo{nullptr};
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetInfo(servableMetadata, reinterpret_cast<const void**>(&servableMetadataRtInfo)));
        ASSERT_NE(nullptr, servableMetadataRtInfo);
        EXPECT_EQ(0, servableMetadataRtInfo->size());
        OVMS_ServableMetadataDelete(servableMetadata);
    }

    void checkServableAsDummy(const std::string& servableName) {
        model_version_t servableVersion = 1;
        ovms::tensor_map_t inputsInfo({{DUMMY_MODEL_INPUT_NAME,
            std::make_shared<ovms::TensorInfo>(DUMMY_MODEL_INPUT_NAME, ovms::Precision::FP32, ovms::Shape{1, 10})}});
        ovms::tensor_map_t outputsInfo({{DUMMY_MODEL_OUTPUT_NAME,
            std::make_shared<ovms::TensorInfo>(DUMMY_MODEL_OUTPUT_NAME, ovms::Precision::FP32, ovms::Shape{1, 10})}});
        checkMetadata(servableName, servableVersion, inputsInfo, outputsInfo);
    }
};
OVMS_Server* CAPIMetadata::cserver = nullptr;

TEST_F(CAPIMetadata, Negative) {
    OVMS_ServableMetadata* servableMetadata = nullptr;
    const std::string servableName = "dummy";
    model_version_t servableVersion = 1;
    // nullptr tests
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(nullptr, servableName.c_str(), servableVersion, &servableMetadata), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, nullptr, servableVersion, &servableMetadata), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, servableName.c_str(), servableVersion, nullptr), StatusCode::NONEXISTENT_PTR);
    // negative missing servable
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, "NONEXISTENT_NAME", servableVersion, &servableMetadata), StatusCode::MODEL_NAME_MISSING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, servableName.c_str(), -1, &servableMetadata), StatusCode::MODEL_VERSION_MISSING);
    // proper call
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, servableName.c_str(), servableVersion, &servableMetadata));
    ASSERT_NE(nullptr, servableMetadata);
    uint32_t inputCount = 42;
    uint32_t outputCount = 42;
    // OVMS_ServableMetadataGetInputCount
    // negative
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInputCount(nullptr, &inputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInputCount(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutputCount(nullptr, &outputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutputCount(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);

    // check inputs
    uint32_t id = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    int64_t* shapeMin{nullptr};
    int64_t* shapeMax{nullptr};
    size_t dimCount = 42;
    const char* tensorName{nullptr};
    // OVMS_ServableMetadataGetInput
    // negative
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(nullptr, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, 412, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, id, nullptr, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, id, &tensorName, nullptr, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, id, &tensorName, &datatype, nullptr, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, id, &tensorName, &datatype, &dimCount, nullptr, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, nullptr), StatusCode::NONEXISTENT_PTR);
    // check outputs
    id = 0;
    datatype = (OVMS_DataType)199;
    shapeMin = nullptr;
    shapeMax = nullptr;
    dimCount = 42;
    tensorName = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(nullptr, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, 412, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, id, nullptr, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, id, &tensorName, nullptr, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, id, &tensorName, &datatype, nullptr, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, nullptr, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, nullptr), StatusCode::NONEXISTENT_PTR);
    // check info
    const ov::AnyMap* servableMetadataRtInfo;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInfo(nullptr, reinterpret_cast<const void**>(&servableMetadataRtInfo)), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataGetInfo(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);

    OVMS_ServableMetadataDelete(nullptr);
}

TEST_F(CAPIMetadata, BasicDummy) {
    const std::string servableName{"dummy"};
    checkServableAsDummy(servableName);
}

TEST_F(CAPIMetadata, BasicDummyDag) {
    const std::string servableName{"pipeline1Dummy"};
    checkServableAsDummy(servableName);
}

TEST_F(CAPIMetadata, DummyDynamicShapes) {
    const std::string servableName = "dummyDynamic";
    model_version_t servableVersion = 1;
    ovms::tensor_map_t inputsInfo({{DUMMY_MODEL_INPUT_NAME,
        std::make_shared<ovms::TensorInfo>(DUMMY_MODEL_INPUT_NAME, ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), {1, 10}})}});
    ovms::tensor_map_t outputsInfo({{DUMMY_MODEL_OUTPUT_NAME,
        std::make_shared<ovms::TensorInfo>(DUMMY_MODEL_OUTPUT_NAME, ovms::Precision::FP32, ovms::Shape{ovms::Dimension::any(), {1, 10}})}});
    checkMetadata(servableName, servableVersion, inputsInfo, outputsInfo);
}

TEST_F(CAPIMetadata, TwoInputsAddModel) {
    const std::string servableName = "add";
    model_version_t servableVersion = 1;
    ovms::tensor_map_t inputsInfo({{SUM_MODEL_INPUT_NAME_1,
                                       std::make_shared<ovms::TensorInfo>(SUM_MODEL_INPUT_NAME_1, ovms::Precision::FP32, ovms::Shape{1, 3})},
        {SUM_MODEL_INPUT_NAME_2,
            std::make_shared<ovms::TensorInfo>(SUM_MODEL_INPUT_NAME_2, ovms::Precision::FP32, ovms::Shape{1, 3})}});
    ovms::tensor_map_t outputsInfo({{SUM_MODEL_OUTPUT_NAME,
        std::make_shared<ovms::TensorInfo>(SUM_MODEL_OUTPUT_NAME, ovms::Precision::FP32, ovms::Shape{1, 3})}});
    checkMetadata(servableName, servableVersion, inputsInfo, outputsInfo);
}

TEST_F(CAPIInference, CallInferenceServerNotStarted) {
    OVMS_Server* cserver = nullptr;
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, cserver);
    ASSERT_NE(nullptr, request);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::SERVER_NOT_READY);
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
    OVMS_ServerDelete(cserver);
}

class CAPIDagInference : public ::testing::Test {
protected:
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    OVMS_Server* cserver = nullptr;

    const uint32_t notUsedNum = 0;

    uint32_t outputCount = 42;
    uint32_t parameterCount = 42;

    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    void SetUp() override {
        std::string port = "9000";
        randomizePort(port);
        // prepare options
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
        ASSERT_NE(serverSettings, nullptr);
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
        ASSERT_NE(modelsSettings, nullptr);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
        ASSERT_NE(cserver, nullptr);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));

        outputCount = 42;
        parameterCount = 42;

        bytesize = 42;
        outputId = 0;
        datatype = (OVMS_DataType)199;
        shape = nullptr;
        dimCount = 42;
        bufferType = (OVMS_BufferType)199;
        deviceId = 42;
        outputName = nullptr;
    }
    void TearDown() override {
        OVMS_ServerDelete(cserver);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        serverSettings = nullptr;
        modelsSettings = nullptr;
        cserver = nullptr;
    }
};

TEST_F(CAPIDagInference, BasicDummyDag) {
    //////////////////////
    // start server
    //////////////////////
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_dummy_dag.json"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    ///////////////////////
    // request creation
    ///////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "pipeline1Dummy", 1));
    ASSERT_NE(nullptr, request);

    // adding input
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // verify GetOutputCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
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
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIDagInference, DynamicEntryDummyDag) {
    //////////////////////
    // start server
    //////////////////////
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_dummy_dynamic_entry_dag.json"));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    ///////////////////////
    // request creation
    ///////////////////////
    OVMS_InferenceRequest* request{nullptr};
    const std::string servableName{"pipeline1DummyDynamicDemultiplex"};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, servableName.c_str(), 1));
    ASSERT_NE(nullptr, request);

    // adding input
    const size_t demultiplyCount = 3;
    std::array<int64_t, demultiplyCount> inputShape{demultiplyCount, 1, 10};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, inputShape.data(), inputShape.size()));
    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE * demultiplyCount> data;
    std::iota(data.begin(), data.end(), 0);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // verify GetOutputCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 3);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);

    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        if (i == 0) {
            EXPECT_EQ(demultiplyCount, shape[i]) << "Different at:" << i << " place.";
        } else {
            EXPECT_EQ(DUMMY_MODEL_SHAPE[i - 1], shape[i]) << "Different at:" << i << " place.";
        }
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE * demultiplyCount);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
}

TEST(CAPI, ApiVersion) {
    uint32_t major = 9999, minor = 9999;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ApiVersion(nullptr, &minor), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ApiVersion(&major, nullptr), StatusCode::NONEXISTENT_PTR);

    ASSERT_CAPI_STATUS_NULL(OVMS_ApiVersion(&major, &minor));
    ASSERT_EQ(major, OVMS_API_VERSION_MAJOR);
    ASSERT_EQ(minor, OVMS_API_VERSION_MINOR);
}
