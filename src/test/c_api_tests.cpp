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
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "../capi_frontend/capi_request_utils.hpp"
#include "../deserialization_main.hpp"
#include "../inference_executor.hpp"
#include "../capi_frontend/capi_utils.hpp"
#include "../capi_frontend/serialization.hpp"
#include "../capi_frontend/deserialization.hpp"
#include "../capi_frontend/inferencerequest.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../capi_frontend/buffer.hpp"
#include "../capi_frontend/capi_dag_utils.hpp"
#include "../capi_frontend/servablemetadata.hpp"
#include "../dags/pipelinedefinitionstatus.hpp"
#include "../metric_module.hpp"
#include "../ovms.h"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../version.hpp"
#include "c_api_test_utils.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"
#include "light_test_utils.hpp"
#include "platform_utils.hpp"
#include "test_with_temp_dir.hpp"

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

const uint32_t AVAILABLE_CORES = std::thread::hardware_concurrency();

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
    EXPECT_EQ(serverSettings->grpcPort, 0);
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
    EXPECT_EQ(serverSettings->grpcMaxThreads, std::nullopt);
    EXPECT_EQ(serverSettings->grpcMemoryQuota, std::nullopt);
    EXPECT_EQ(serverSettings->filesystemPollWaitMilliseconds, 1000);
    EXPECT_EQ(serverSettings->sequenceCleanerPollWaitMinutes, 5);
    EXPECT_EQ(serverSettings->resourcesCleanerPollWaitSeconds, 300);
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
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcMaxThreads(_serverSettings, 100));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcMemoryQuota(_serverSettings, (size_t)1000000));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(_serverSettings, 2));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(_serverSettings, 3));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(_serverSettings, 4));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCpuExtensionPath(_serverSettings, getGenericFullPathForSrcTest("/ovms/src/test").c_str()));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetCacheDir(_serverSettings, getGenericFullPathForTmp("/tmp/cache").c_str()));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_INFO));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_ERROR));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_DEBUG));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_WARNING));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_TRACE));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogLevel(_serverSettings, static_cast<OVMS_LogLevel>(99)), StatusCode::NONEXISTENT_LOG_LEVEL);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetLogPath(_serverSettings, getGenericFullPathForTmp("/tmp/logs").c_str()));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(_modelsSettings, getGenericFullPathForTmp("/tmp/config").c_str()));
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcMaxThreads(nullptr, 100), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetGrpcMemoryQuota(nullptr, 1000000), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetFileSystemPollWaitSeconds(nullptr, 2), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetSequenceCleanerPollWaitMinutes(nullptr, 3), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCustomNodeResourcesCleanerIntervalSeconds(nullptr, 4), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCpuExtensionPath(nullptr, "/ovms/src/test"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCpuExtensionPath(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCacheDir(nullptr, getGenericFullPathForTmp("/tmp/cache").c_str()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetCacheDir(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogLevel(nullptr, OVMS_LOG_TRACE), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogPath(nullptr, getGenericFullPathForTmp("/tmp/logs").c_str()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerSettingsSetLogPath(_serverSettings, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ModelsSettingsSetConfigPath(nullptr, getGenericFullPathForTmp("/tmp/config").c_str()), StatusCode::NONEXISTENT_PTR);
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
    EXPECT_EQ(serverSettings->cpuExtensionLibraryPath, getGenericFullPathForSrcTest("/ovms/src/test"));
    EXPECT_EQ(serverSettings->logLevel, "TRACE");
    EXPECT_EQ(serverSettings->logPath, getGenericFullPathForTmp("/tmp/logs"));
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(serverSettings->grpcChannelArguments, "grpcargs");
    EXPECT_EQ(serverSettings->grpcMaxThreads, 100);
    EXPECT_EQ(serverSettings->grpcMemoryQuota, (size_t)1000000);
    EXPECT_EQ(serverSettings->filesystemPollWaitMilliseconds, 2000);
    EXPECT_EQ(serverSettings->sequenceCleanerPollWaitMinutes, 3);
    EXPECT_EQ(serverSettings->resourcesCleanerPollWaitSeconds, 4);
    EXPECT_EQ(serverSettings->cacheDir, getGenericFullPathForTmp("/tmp/cache"));

    testDefaultSingleModelOptions(modelsSettings);
    EXPECT_EQ(modelsSettings->configPath, getGenericFullPathForTmp("/tmp/config"));

    // Test config parser
    ConstructorEnabledConfig cfg;
#ifdef __linux__
    ASSERT_TRUE(cfg.parse(serverSettings, modelsSettings));
    EXPECT_EQ(cfg.grpcWorkers(), AVAILABLE_CORES);
#elif _WIN32
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcWorkers(_serverSettings, 1));
    ASSERT_TRUE(cfg.parse(serverSettings, modelsSettings));
    EXPECT_EQ(cfg.grpcWorkers(), 1);
#endif
    EXPECT_EQ(cfg.port(), 5555);
    EXPECT_EQ(cfg.restPort(), 6666);
    EXPECT_EQ(cfg.grpcBindAddress(), "2.2.2.2");
    EXPECT_EQ(cfg.restWorkers(), 31);
    EXPECT_EQ(cfg.restBindAddress(), "3.3.3.3");
    // EXPECT_EQ(serverSettings->metricsEnabled, false);
    // EXPECT_EQ(serverSettings->metricsList, "");
    EXPECT_EQ(cfg.cpuExtensionLibraryPath(), getGenericFullPathForSrcTest("/ovms/src/test"));
    EXPECT_EQ(cfg.logLevel(), "TRACE");
    EXPECT_EQ(cfg.logPath(), getGenericFullPathForTmp("/tmp/logs"));
    // trace path  // not tested since it is not supported in C-API
    EXPECT_EQ(cfg.grpcChannelArguments(), "grpcargs");
    EXPECT_EQ(cfg.filesystemPollWaitMilliseconds(), 2000);
    EXPECT_EQ(cfg.sequenceCleanerPollWaitMinutes(), 3);
    EXPECT_EQ(cfg.resourcesCleanerPollWaitSeconds(), 4);
    EXPECT_EQ(cfg.cacheDir(), getGenericFullPathForTmp("/tmp/cache"));

    EXPECT_EQ(cfg.modelName(), "");
    EXPECT_EQ(cfg.modelPath(), "");
    EXPECT_EQ(cfg.batchSize(), "");
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

    EXPECT_EQ(cfg.configPath(), getGenericFullPathForTmp("/tmp/config"));

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
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config.json").c_str()));

    // Expect fail
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings),
        StatusCode::OPTIONS_USAGE_ERROR);

    // Fix and expect ok
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, 0));
    // ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetRestPort(serverSettings, 6666));  // Different port TODO @atobisze @dkalinow not really needed for C-API
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusCode(nullptr, &code), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusCode(sts, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusCode(sts, &code));
    EXPECT_EQ(code, static_cast<uint32_t>(StatusCode::INTERNAL_ERROR));
    const char* details = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusDetails(nullptr, &details), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_StatusDetails(sts, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_StatusDetails(sts, &details));
    std::stringstream ss;
    ss << Status(StatusCode::INTERNAL_ERROR).string() << " - custom message";
    EXPECT_EQ(std::string(details), ss.str());
    OVMS_StatusDelete(reinterpret_cast<OVMS_Status*>(s.release()));
}

class CAPIInference : public ::testing::Test {};

TEST(CAPIServerMetadata, Basic) {
    OVMS_Metadata* metadata = nullptr;
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMetadata(nullptr, &metadata), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMetadata(cserver, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerMetadata(cserver, &metadata));
    const char* json;
    size_t size;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_SerializeMetadataToString(nullptr, &json, &size), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_SerializeMetadataToString(metadata, nullptr, &size), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_SerializeMetadataToString(metadata, &json, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_SerializeMetadataToString(metadata, &json, &size));
    ASSERT_EQ(std::string(json), std::string("{\"name\":\"" + std::string(PROJECT_NAME) + "\",\"version\":\"" + std::string(PROJECT_VERSION) + "\",\"ov_version\":\"" + std::string(OPENVINO_NAME) + "\"}"));
    ASSERT_EQ(size, std::strlen(json));
    OVMS_StringFree(json);
    const char* pointer = "/name";
    const char* value;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_MetadataFieldByPointer(nullptr, pointer, &value, &size), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_MetadataFieldByPointer(metadata, nullptr, &value, &size), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_MetadataFieldByPointer(metadata, pointer, nullptr, &size), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_MetadataFieldByPointer(metadata, pointer, &value, nullptr), StatusCode::NONEXISTENT_PTR);

    ASSERT_CAPI_STATUS_NULL(OVMS_MetadataFieldByPointer(metadata, pointer, &value, &size));
    ASSERT_EQ(std::string(value), std::string(PROJECT_NAME));
    ASSERT_EQ(size, std::strlen(value));
    OVMS_StringFree(value);

    pointer = "/version";
    ASSERT_CAPI_STATUS_NULL(OVMS_MetadataFieldByPointer(metadata, pointer, &value, &size));
    ASSERT_EQ(std::string(value), std::string(PROJECT_VERSION));
    ASSERT_EQ(size, std::strlen(value));
    OVMS_StringFree(value);

    pointer = "/ov_version";
    ASSERT_CAPI_STATUS_NULL(OVMS_MetadataFieldByPointer(metadata, pointer, &value, &size));
    ASSERT_EQ(std::string(value), std::string(OPENVINO_NAME));
    ASSERT_EQ(size, std::strlen(value));
    OVMS_StringFree(value);

    pointer = "/dummy";
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_MetadataFieldByPointer(metadata, pointer, &value, &size), StatusCode::JSON_SERIALIZATION_ERROR);

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerMetadataDelete(nullptr), StatusCode::NONEXISTENT_PTR);
    OVMS_ServerMetadataDelete(metadata);
    OVMS_ServerDelete(cserver);
}

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

TEST(CAPIServableMetadata, NoInputsAndOutputs) {
    tensor_map_t m;
    ovms::ServableMetadata sm("dummy", 1, m, m);
    OVMS_ServableMetadata* osm = reinterpret_cast<OVMS_ServableMetadata*>(&sm);
    uint32_t count;
    ASSERT_EQ(sm.getVersion(), 1);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInputCount(osm, &count));
    ASSERT_EQ(count, 0);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutputCount(osm, &count));
    ASSERT_EQ(count, 0);
}

TEST(CAPIInferenceRequest, Basic) {
    InferenceRequest* r = new InferenceRequest("dummy", 1);
    size_t batchSize;
    ASSERT_EQ(r->getBatchSize(batchSize, 1), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(r->removeInputBuffer("dummy"), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVE_BUFFER);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputRemoveData(nullptr, "dummy"), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputRemoveData(reinterpret_cast<OVMS_InferenceRequest*>(r), nullptr), StatusCode::NONEXISTENT_PTR);
    delete r;
}
TEST(CAPIInferenceResponse, Basic) {
    InferenceResponse* r = new InferenceResponse("dummy", 1);
    int64_t a[1] = {1};
    ASSERT_EQ(r->addOutput("n", OVMS_DataType::OVMS_DATATYPE_BIN, a, 1), StatusCode::OK);
    InferenceTensor* tensor;
    const std::string* name;
    OVMS_InferenceResponse* response = reinterpret_cast<OVMS_InferenceResponse*>(r);
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName = "n";
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::INTERNAL_ERROR);  // Test GetOutput without defined buffer
    ASSERT_EQ(r->getOutput(0, &name, &tensor), StatusCode::OK);
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(0, OVMS_BufferType::OVMS_BUFFERTYPE_CPU, 0);
    tensor->setBuffer(std::move(buffer));
    outputName = "n";
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));

    delete r;
}

TEST_F(CAPIInference, Validation) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, request);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_BIN, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    InferenceRequest* ir = reinterpret_cast<InferenceRequest*>(request);
    size_t size = 0;
    ASSERT_EQ(ir->getBatchSize(size, 10), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(ir->getBatchSize(size, 0), StatusCode::OK);
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_PRECISION);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, ValidationMaliciousTensorStaticShapeModel) {
    // Prepare request with tensor shape not alligining with data size with overflow attempt
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json"));
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, request);
    int64_t shape[2] = {std::numeric_limits<size_t>::max() / 5 + 2, 5};
    // multiplication will overflow size_t
    size_t numElements = shape[0] * shape[1];
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, shape, 2));
    std::vector<float> data(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<float>(i);
    }
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    InferenceRequest* ir = reinterpret_cast<InferenceRequest*>(request);
    size_t size = 0;
    ASSERT_EQ(ir->getBatchSize(size, 10), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(ir->getBatchSize(size, 0), StatusCode::OK);
    OVMS_InferenceResponse* response = nullptr;
    // Validation captures shape mismatch with model defined static shape
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_BATCH_SIZE);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, ValidationMaliciousTensorDynamicShapeModel) {
    // Prepare request with tensor shape not alligining with data size with overflow attempt
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_dummy_full_dynamic_shape.json"));
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, request);
    int64_t shape[2] = {std::numeric_limits<size_t>::max() / 5 + 2, 5};
    // multiplication will overflow size_t
    size_t numElements = shape[0] * shape[1];
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, shape, 2));
    std::vector<float> data(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<float>(i);
    }
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    InferenceRequest* ir = reinterpret_cast<InferenceRequest*>(request);
    size_t size = 0;
    ASSERT_EQ(ir->getBatchSize(size, 10), StatusCode::INTERNAL_ERROR);
    ASSERT_EQ(ir->getBatchSize(size, 0), StatusCode::OK);
    OVMS_InferenceResponse* response = nullptr;
    // For models with dynamic shape validation captures shape overflow
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::INVALID_SHAPE);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, AcceptInputRejectOutputStringPrecision) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_string.json"));
    OVMS_Server* cserver = serverGuard.server;
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "passthrough", 1));
    ASSERT_NE(nullptr, request);
    std::array<int64_t, 1> shape{1};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, PASSTHROUGH_STRING_MODEL_INPUT_NAME, OVMS_DATATYPE_STRING, shape.data(), shape.size()));
    std::array<std::string, 1> data{"RandomString"};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, PASSTHROUGH_STRING_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(std::string) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddOutput(request, PASSTHROUGH_STRING_MODEL_OUTPUT_NAME, OVMS_DATATYPE_STRING, shape.data(), shape.size()), StatusCode::NOT_IMPLEMENTED);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, TwoInputs) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_double_dummy.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "pipeline1Dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "b", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "b", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, "c", OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, "c", reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    uint32_t outputId = 0;
    const void* voutputData;
    size_t bytesize = 42;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string("a"), outputName);
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
    outputId = 1;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string("d"), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
}
TEST_F(CAPIInference, Basic) {
    //////////////////////
    // start server
    //////////////////////
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutputCount(nullptr, &outputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutputCount(response, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameterCount(nullptr, &parameterCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameterCount(response, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(nullptr, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, nullptr, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, nullptr, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, nullptr, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, nullptr, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, nullptr, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, nullptr, &bufferType, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, nullptr, &deviceId), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
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
}
TEST_F(CAPIInference, ReuseInputRemoveAndAddData) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
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

    //////////////////
    //  INFERENCE #1
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    const void* voutputData = nullptr;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
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
    //////////////////
    //  INFERENCE #2 - reuse request & input but reset the data
    //////////////////
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputRemoveData(request, DUMMY_MODEL_INPUT_NAME));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data2{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};  // here we have different data
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data2.data()), sizeof(float) * data2.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    voutputData = nullptr;
    bytesize = 42;
    outputId = 0;
    datatype = (OVMS_DataType)199;
    shape = nullptr;
    dimCount = 42;
    bufferType = (OVMS_BufferType)199;
    deviceId = 42;
    outputName = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data2.size(); ++i) {
        EXPECT_EQ(data2[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, ReuseRequestRemoveAndAddInput) {
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_dummy_dynamic_shape.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    ///////////////////////
    // request creation
    ///////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_NE(nullptr, request);
    // adding input
    const ovms::signed_shape_t firstRequestShape{1, 5};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, firstRequestShape.data(), firstRequestShape.size()));
    // setting buffer
    std::array<float, 5> data{0, 1, 2, 3, 4};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));

    //////////////////
    //  INFERENCE #1
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    const void* voutputData = nullptr;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    for (size_t i = 0; i < firstRequestShape.size(); ++i) {
        EXPECT_EQ(firstRequestShape[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * firstRequestShape[1]);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
    //////////////////
    //  INFERENCE #2 - reuse request but not input
    //////////////////
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveInput(request, DUMMY_MODEL_INPUT_NAME));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data2{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};  // here we have different data
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data2.data()), sizeof(float) * data2.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    voutputData = nullptr;
    bytesize = 42;
    outputId = 0;
    datatype = (OVMS_DataType)199;
    shape = nullptr;
    dimCount = 42;
    bufferType = (OVMS_BufferType)199;
    deviceId = 42;
    outputName = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data2.size(); ++i) {
        EXPECT_EQ(data2[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
}

TEST_F(CAPIInference, NegativeInference) {
    // first start OVMS
    std::string port = "9000";
    randomizeAndEnsureFree(port);
    // prepare options
    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_NE(serverSettings, nullptr);
    ASSERT_NE(modelsSettings, nullptr);
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json").c_str()));

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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, request, &response), StatusCode::NONEXISTENT_BUFFER);

    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(nullptr, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestInputSetData(request, nullptr, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    //
    // Tests for output setting
    //
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddOutput(nullptr, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddOutput(request, nullptr, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, nullptr, DUMMY_MODEL_SHAPE.size()), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    // fail with adding output second time
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()), StatusCode::DOUBLE_TENSOR_INSERT);
    // setting buffer
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputSetData(nullptr, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputSetData(request, nullptr, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum), StatusCode::DOUBLE_BUFFER_SET);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputRemoveData(nullptr, DUMMY_MODEL_OUTPUT_NAME), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputRemoveData(request, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputRemoveData(request, "NONEXISTENT_NAME"), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVE_BUFFER);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputRemoveData(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestOutputRemoveData(request, DUMMY_MODEL_OUTPUT_NAME), StatusCode::NONEXISTENT_BUFFER_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveOutput(nullptr, DUMMY_MODEL_OUTPUT_NAME), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveOutput(request, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveOutput(request, "NONEXISTENT_NAME"), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestRemoveOutput(request, DUMMY_MODEL_OUTPUT_NAME));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceRequestRemoveOutput(request, DUMMY_MODEL_OUTPUT_NAME), StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL);

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

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_Inference(cserver, requestNoModel, &reponseNoModel), StatusCode::NONEXISTENT_PTR);
    OVMS_InferenceRequestDelete(requestNoModel);

    OVMS_ServerDelete(nullptr);
    OVMS_ServerDelete(cserver);
    OVMS_ServerDelete(nullptr);
}
TEST_F(CAPIInference, String) {
    using std::string;
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_string.json"));
    OVMS_InferenceRequest* request{nullptr};
    const string MODEL_NAME = "passthrough";
    const string INPUT_NAME = "my_name";
    const string OUTPUT_NAME = "my_name";
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, serverGuard.server, MODEL_NAME.c_str(), 1));
    ASSERT_NE(nullptr, request);
    std::vector<string> data{{"Pan"}, {"Kleks"}};
    std::vector<int64_t> inShape{(int64_t)data.size()};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, INPUT_NAME.c_str(), OVMS_DATATYPE_STRING, inShape.data(), inShape.size()));
    // setting buffer
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, INPUT_NAME.c_str(), reinterpret_cast<void*>(&data[0]), int64_t(data.size() * sizeof(std::string)), OVMS_BUFFERTYPE_CPU, notUsedNum));
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(serverGuard.server, request, &response));
    // verify GetOutputCount
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(string(OUTPUT_NAME.c_str()), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_STRING);
    EXPECT_EQ(dimCount, inShape.size());
    EXPECT_TRUE(std::equal(inShape.begin(), inShape.end(), reinterpret_cast<const int64_t*>(shape)));
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    EXPECT_EQ(bytesize, sizeof(string) * data.size());
    EXPECT_EQ(data.size(), bytesize / sizeof(string));
    EXPECT_TRUE(std::equal(data.begin(), data.end(), reinterpret_cast<const string*>(voutputData)));
    // now perform second inference and then check if first output is still ok
    OVMS_InferenceResponse* response2 = nullptr;
    const std::vector<std::string> originalData = data;
    // request still points to original data
    data[0] = "Habana";
    data[1] = "Gaudi";
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(serverGuard.server, request, &response2));
    const void* voutputData2;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response2, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData2, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(string(OUTPUT_NAME.c_str()), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_STRING);
    EXPECT_EQ(dimCount, 1);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    EXPECT_EQ(bytesize, sizeof(string) * data.size());
    EXPECT_EQ(data.size(), bytesize / sizeof(string));
    EXPECT_TRUE(std::equal(data.begin(), data.end(), reinterpret_cast<const string*>(voutputData2)));
    // now we check previous response ensuring copy on output
    EXPECT_TRUE(std::equal(originalData.begin(), originalData.end(), reinterpret_cast<const string*>(voutputData)));
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceResponseDelete(response2);
}
TEST_F(CAPIInference, Scalar) {
    //////////////////////
    // start server
    //////////////////////
    ServerGuard serverGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_scalar.json").c_str());
    OVMS_Server* cserver = serverGuard.server;
    ASSERT_NE(cserver, nullptr);
    ///////////////////////
    // request creation
    ///////////////////////
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "scalar", 1));
    ASSERT_NE(nullptr, request);

    // adding input with shape dim count=0
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, SCALAR_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, nullptr, 0));
    // setting buffer
    std::array<float, 1> data{3.1f};
    uint32_t notUsedNum = 0;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, SCALAR_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum));

    //////////////////
    //  INFERENCE
    //////////////////

    OVMS_InferenceResponse* response = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    // verify GetOutputCount
    uint32_t outputCount = 42;
    ASSERT_NE(nullptr, response);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
    ASSERT_EQ(std::string(SCALAR_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 0);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);
    EXPECT_EQ(deviceId, 0);
    EXPECT_EQ(bytesize, sizeof(float));
    EXPECT_EQ(*((float*)voutputData), data[0]);

    ///////////////
    // CLEANUP
    ///////////////
    OVMS_InferenceResponseDelete(response);
    OVMS_InferenceRequestDelete(request);
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);

    uint32_t parameterCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(1, parameterCount);
    // verify get Parameter
    OVMS_DataType parameterDatatype = OVMS_DATATYPE_FP32;
    const void* parameterData{nullptr};
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameter(nullptr, 0, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameter(response, 0, nullptr, &parameterData), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameter(response, 0, &parameterDatatype, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameter(response, 0, &parameterDatatype, &parameterData));
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId + 42123, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseOutput(response, outputId + 1, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId), StatusCode::INTERNAL_ERROR);
    // negative scenario nonexistsing parameter
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_InferenceResponseParameter(response, 123, &parameterDatatype, &parameterData), StatusCode::NONEXISTENT_PARAMETER);
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
        randomizeAndEnsureFree(port);
        // prepare options
        OVMS_ServerSettings* serverSettings = nullptr;
        OVMS_ModelsSettings* modelsSettings = nullptr;
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
        ASSERT_NE(serverSettings, nullptr);
        ASSERT_NE(modelsSettings, nullptr);
        ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
        ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config_metadata_all.json").c_str()));
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
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInputCount(servableMetadata, &inputCount));
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutputCount(servableMetadata, &outputCount));
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
            ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
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
            ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax));
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
        ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataInfo(servableMetadata, reinterpret_cast<const void**>(&servableMetadataRtInfo)));
        ASSERT_NE(nullptr, servableMetadataRtInfo);
        std::cout << "SERVABLE:::" << servableName.c_str() << std::endl;
        try {
            if (servableName == "dummy") {
                EXPECT_EQ((*servableMetadataRtInfo).at("MO_version").as<std::string>(), "2020.1.0-61-gd349c3ba4a");
                EXPECT_EQ((*servableMetadataRtInfo).at("model_info").as<ov::AnyMap>().at("resolution").as<ov::AnyMap>().at("height").as<std::string>(), "200");
                EXPECT_EQ((*servableMetadataRtInfo).at("conversion_parameters").as<ov::AnyMap>().at("data_type").as<std::string>(), "float");
                EXPECT_EQ((*servableMetadataRtInfo).at("optimization").as<std::string>(), "");
                EXPECT_EQ(6, servableMetadataRtInfo->size());
            } else if (servableName == "scalar") {
                EXPECT_EQ((*servableMetadataRtInfo).at("MO_version").as<std::string>(), "2023.0.0-10926-b4452d56304-releases/2023/0");
                EXPECT_EQ((*servableMetadataRtInfo).at("conversion_parameters").as<ov::AnyMap>().at("layout").as<std::string>(), "...");
                EXPECT_EQ(6, servableMetadataRtInfo->size());
            } else if (servableName == "pipeline1Dummy") {
                EXPECT_EQ(0, servableMetadataRtInfo->size());
            }
        } catch (const std::out_of_range& e) {
            FAIL() << "Metadata key not found: " << e.what();
        } catch (const std::bad_any_cast& e) {
            FAIL() << "Bad any cast: " << e.what();
        } catch (const std::exception& e) {
            FAIL() << "Unexpected exception: " << e.what();
        }
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
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, "NONEXISTENT_NAME", servableVersion, &servableMetadata), StatusCode::PIPELINE_DEFINITION_NAME_MISSING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableMetadata(cserver, servableName.c_str(), -1, &servableMetadata), StatusCode::MODEL_VERSION_MISSING);
    // proper call
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, servableName.c_str(), servableVersion, &servableMetadata));
    ASSERT_NE(nullptr, servableMetadata);
    uint32_t inputCount = 42;
    uint32_t outputCount = 42;
    // OVMS_ServableMetadataGetInputCount
    // negative
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInputCount(nullptr, &inputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInputCount(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutputCount(nullptr, &outputCount), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutputCount(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);

    // check inputs
    uint32_t id = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    int64_t* shapeMin{nullptr};
    int64_t* shapeMax{nullptr};
    size_t dimCount = 42;
    const char* tensorName{nullptr};
    // OVMS_ServableMetadataGetInput
    // negative
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(nullptr, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, 412, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, id, nullptr, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, nullptr, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, nullptr, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, &dimCount, nullptr, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, nullptr), StatusCode::NONEXISTENT_PTR);
    // check outputs
    id = 0;
    datatype = (OVMS_DataType)199;
    shapeMin = nullptr;
    shapeMax = nullptr;
    dimCount = 42;
    tensorName = nullptr;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(nullptr, id, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, 412, &tensorName, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_TENSOR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, id, nullptr, &datatype, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, nullptr, &dimCount, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, &datatype, nullptr, &shapeMin, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, nullptr, &shapeMax), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataOutput(servableMetadata, id, &tensorName, &datatype, &dimCount, &shapeMin, nullptr), StatusCode::NONEXISTENT_PTR);
    // check info
    const ov::AnyMap* servableMetadataRtInfo;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInfo(nullptr, reinterpret_cast<const void**>(&servableMetadataRtInfo)), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServableMetadataInfo(servableMetadata, nullptr), StatusCode::NONEXISTENT_PTR);

    OVMS_ServableMetadataDelete(nullptr);
}
class CAPIState : public ::testing::Test {
public:
    static std::shared_ptr<MockModelInstanceChangingStates> modelInstance;
    static void TearDownTestSuite() {
        modelInstance.reset();
    }
    class MockModel : public Model {
    public:
        MockModel(const std::string& name, std::shared_ptr<ModelInstance> instance) :
            Model(name, false /*stateful*/, nullptr) {
            modelVersions.insert({instance->getVersion(), instance});
        }
    };
    class MockModelManager : public ModelManager {
    public:
        const std::string servableName = "dummy";
        MockModelManager() {
            ov::Core ieCore;
            CAPIState::modelInstance = std::make_shared<MockModelInstanceChangingStates>(servableName, 1, ieCore);
            std::shared_ptr<MockModel> model = std::make_shared<MockModel>(servableName, modelInstance);
            models[servableName] = model;
        }
    };
    class MockGrpcServerModule : public Module {
    public:
        MockGrpcServerModule() {
            state = ModuleState::INITIALIZED;
        }
        Status start(const ovms::Config& config) {
            return StatusCode::OK;
        }
        void shutdown() {}
    };
    class MockServableManagerModule : public ServableManagerModule {
    public:
        MockServableManagerModule(Server& server) :
            ServableManagerModule(server) {
            servableManager = std::make_unique<MockModelManager>();
            state = ModuleState::INITIALIZED;
        }
    };
    class MockServer : public Server {
    public:
        MockServer() {
            MetricModule* mm = new MetricModule();
            modules.insert({METRICS_MODULE_NAME, std::unique_ptr<Module>(mm)});
        }
        void setReady() {
            Module* msmm = new MockServableManagerModule(*this);
            modules.insert({SERVABLE_MANAGER_MODULE_NAME, std::unique_ptr<Module>(msmm)});
        }
        void setLive() {
            Module* grpc = new MockGrpcServerModule();
            modules.insert({CAPI_MODULE_NAME, std::unique_ptr<Module>(grpc)});
        }
    };
};

class CAPIStateIntegration : public TestWithTempDir {
protected:
    std::string configFilePath;
    void SetUp() {
        TestWithTempDir::SetUp();
        configFilePath = directoryPath + "/ovms_config.json";
    }
};

TEST_F(CAPIStateIntegration, LiveReadyFromMalformedConfig) {
    OVMS_Server* server = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&server));
    OVMS_ServerSettings* serverSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    bool isReady = true;
    bool isLive = true;
    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(!isLive);
    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(!isReady);
    createConfigFileWithContent("{", configFilePath);
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, configFilePath.c_str()));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, 5555));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerStartFromConfigurationFile(server, serverSettings, modelsSettings), StatusCode::JSON_INVALID);
    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(isLive);
    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(!isReady);
    OVMS_ServerDelete(server);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
}

TEST_F(CAPIStateIntegration, LiveReadyFromConfig) {
    OVMS_Server* server = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&server));
    OVMS_ServerSettings* serverSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    bool isReady;
    bool isLive;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerReady(nullptr, &isReady), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_ServerLive(nullptr, &isLive), StatusCode::NONEXISTENT_PTR);
    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(!isLive);
    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(!isReady);
    std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/configs/emptyConfig.json"), configFilePath, std::filesystem::copy_options::recursive);
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, configFilePath.c_str()));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, 5555));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(server, serverSettings, modelsSettings));
    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(isLive);
    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(isReady);
    OVMS_ServerDelete(server);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
}

TEST_F(CAPIStateIntegration, Config) {
    OVMS_Server* cserver = nullptr;
    OVMS_ServableState state;
    const std::string servableName = "dummy";
    const int64_t servableVersion = 1;
    ASSERT_CAPI_STATUS_NULL(
        OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(
        OVMS_GetServableState(cserver, nullptr, servableVersion, &state), StatusCode::NONEXISTENT_PTR);
    OVMS_ServerSettings* serverSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(
        OVMS_ServerSettingsNew(&serverSettings));
    ovms::ServerSettingsImpl* directPtrServerSettings = reinterpret_cast<ovms::ServerSettingsImpl*>(serverSettings);
    directPtrServerSettings->filesystemPollWaitMilliseconds = 20;  // mock 20ms config.json polling to make the test shorter
    directPtrServerSettings->grpcPort = 5555;                      // need to set to sth
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(
        OVMS_ModelsSettingsNew(&modelsSettings));
    std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/configs/emptyConfig.json"), configFilePath, std::filesystem::copy_options::recursive);
    ASSERT_CAPI_STATUS_NULL(
        OVMS_ModelsSettingsSetConfigPath(modelsSettings, configFilePath.c_str()));
    ASSERT_CAPI_STATUS_NULL(
        OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));

    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(
        OVMS_GetServableState(cserver, servableName.c_str(), servableVersion, &state), StatusCode::MODEL_NAME_MISSING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(
        OVMS_GetServableState(cserver, "pipeline1Dummy", servableVersion, &state), StatusCode::MODEL_NAME_MISSING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(
        OVMS_GetServableState(cserver, "mediaDummy", servableVersion, &state), StatusCode::MODEL_NAME_MISSING);
    std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_metadata_all.json"), configFilePath, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    Server* server = reinterpret_cast<Server*>(cserver);
    const ovms::Module* servableModule = server->getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    ModelManager* modelManager = &dynamic_cast<const ServableManagerModule*>(servableModule)->getServableManager();

    // Instead of waiting for config to get reloaded
    waitForOVMSConfigReload(*modelManager);
    ASSERT_CAPI_STATUS_NULL(
        OVMS_GetServableState(cserver, servableName.c_str(), servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_AVAILABLE);
    ASSERT_CAPI_STATUS_NULL(
        OVMS_GetServableState(cserver, "pipeline1Dummy", servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_AVAILABLE);
#if (MEDIAPIPE_DISABLE == 0)
    std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full.json"), configFilePath, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    waitForOVMSConfigReload(*modelManager);
    ASSERT_CAPI_STATUS_NULL(
        OVMS_GetServableState(cserver, "mediaDummy", servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_AVAILABLE);
#endif
    OVMS_ServerDelete(cserver);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
}

TEST_F(CAPIState, PipelineStates) {
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_BEGIN, convertToServableState(ovms::PipelineDefinitionStateCode::BEGIN));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_LOADING, convertToServableState(ovms::PipelineDefinitionStateCode::RELOADING));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_LOADING_FAILED, convertToServableState(ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_LOADING_FAILED, convertToServableState(ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_AVAILABLE, convertToServableState(ovms::PipelineDefinitionStateCode::AVAILABLE));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_AVAILABLE, convertToServableState(ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION));
    ASSERT_EQ(OVMS_ServableState::OVMS_STATE_RETIRED, convertToServableState(ovms::PipelineDefinitionStateCode::RETIRED));
}

TEST_F(CAPIState, ServerLive) {
    MockServer* cserver = new MockServer;
    OVMS_Server* server = reinterpret_cast<OVMS_Server*>(cserver);
    bool isLive;

    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(!isLive);
    cserver->setLive();
    OVMS_ServerLive(server, &isLive);
    ASSERT_TRUE(isLive);
}

TEST_F(CAPIState, ServerReady) {
    MockServer* cserver = new MockServer;
    OVMS_Server* server = reinterpret_cast<OVMS_Server*>(cserver);
    bool isReady;

    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(!isReady);
    cserver->setReady();
    OVMS_ServerReady(server, &isReady);
    ASSERT_TRUE(isReady);
}

std::shared_ptr<MockModelInstanceChangingStates> CAPIState::modelInstance = nullptr;
TEST_F(CAPIState, ServerNull) {
    MockServer* cserver = new MockServer;
    cserver->setReady();
    cserver->setLive();
    OVMS_Server* server = reinterpret_cast<OVMS_Server*>(cserver);
    OVMS_ServableState state;
    const std::string servableName = "dummy";
    const int64_t servableVersion = 1;
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableState(nullptr, servableName.c_str(), servableVersion, &state), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableState(server, nullptr, servableVersion, &state), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableState(server, servableName.c_str(), servableVersion, nullptr), StatusCode::NONEXISTENT_PTR);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableState(server, servableName.c_str(), -1, &state), StatusCode::MODEL_NAME_MISSING);
    ASSERT_CAPI_STATUS_NOT_NULL_EXPECT_CODE(OVMS_GetServableState(server, "", servableVersion, &state), StatusCode::MODEL_NAME_MISSING);
    delete cserver;
}
TEST_F(CAPIState, AllStates) {
    const std::string servableName = "dummy";
    const int64_t servableVersion = 1;
    auto uptrCserver = std::make_unique<MockServer>();
    MockServer* cserver = uptrCserver.get();
    cserver->setReady();
    cserver->setLive();
    OVMS_Server* server = reinterpret_cast<OVMS_Server*>(cserver);
    OVMS_ServableState state;

    CAPIState::modelInstance->setState(ovms::ModelVersionState::START);
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableState(server, servableName.c_str(), servableVersion, &state));
    ASSERT_EQ(state, OVMS_ServableState::OVMS_STATE_BEGIN);

    CAPIState::modelInstance->setState(ovms::ModelVersionState::AVAILABLE);
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableState(server, servableName.c_str(), servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_AVAILABLE);

    CAPIState::modelInstance->setState(ovms::ModelVersionState::UNLOADING);
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableState(server, servableName.c_str(), servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_UNLOADING);

    CAPIState::modelInstance->setState(ovms::ModelVersionState::END);
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableState(server, servableName.c_str(), servableVersion, &state));
    EXPECT_EQ(state, OVMS_ServableState::OVMS_STATE_RETIRED);

    CAPIState::modelInstance->setState(ovms::ModelVersionState::LOADING);
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableState(server, servableName.c_str(), servableVersion, &state));
    ASSERT_EQ(state, OVMS_ServableState::OVMS_STATE_LOADING);
}

TEST_F(CAPIMetadata, BasicDummy) {
    const std::string servableName{"dummy"};
    checkServableAsDummy(servableName);
}

TEST_F(CAPIMetadata, BasicDummyDag) {
    const std::string servableName{"pipeline1Dummy"};
    checkServableAsDummy(servableName);
}

TEST_F(CAPIMetadata, BasicScalar) {
    const std::string servableName{"scalar"};
    model_version_t servableVersion = 1;
    ovms::tensor_map_t inputsInfo({{SCALAR_MODEL_INPUT_NAME,
        std::make_shared<ovms::TensorInfo>(SCALAR_MODEL_INPUT_NAME, ovms::Precision::FP32, ovms::Shape{})}});
    ovms::tensor_map_t outputsInfo({{SCALAR_MODEL_OUTPUT_NAME,
        std::make_shared<ovms::TensorInfo>(SCALAR_MODEL_OUTPUT_NAME, ovms::Precision::FP32, ovms::Shape{})}});
    checkMetadata(servableName, servableVersion, inputsInfo, outputsInfo);
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
        randomizeAndEnsureFree(port);
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
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config_dummy_dag.json").c_str()));
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
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
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config_dummy_dynamic_entry_dag.json").c_str()));
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
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    // verify GetParameterCount
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseParameterCount(response, &parameterCount));
    ASSERT_EQ(0, parameterCount);
    // verify GetOutput
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId));
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

TEST(CAPI, MultipleThreadsStarting) {
    std::vector<std::unique_ptr<std::thread>> threads;
    size_t threadsCount = std::thread::hardware_concurrency();
    std::vector<std::promise<void>> promises(threadsCount);
    std::vector<std::future<void>> futures;
    futures.reserve(threadsCount);
    std::vector<std::promise<void>> promisesThreadReady(threadsCount);
    std::vector<std::future<void>> futuresThreadReady;
    futuresThreadReady.reserve(threadsCount);
    std::vector<uint32_t> retCodes(threadsCount);
    for (size_t i = 0; i < threadsCount; ++i) {
        futures.emplace_back(promises[i].get_future());
        futuresThreadReady.emplace_back(promisesThreadReady[i].get_future());
        threads.emplace_back(std::make_unique<std::thread>([i, &promisesThreadReady, &futures, &retCodes]() {
            OVMS_Server* cserver = nullptr;
            OVMS_ServerSettings* serverSettings = nullptr;
            OVMS_ModelsSettings* modelsSettings = nullptr;
            ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
            ASSERT_NE(cserver, nullptr);
            ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
            ASSERT_NE(serverSettings, nullptr);
            ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
            ASSERT_NE(modelsSettings, nullptr);

            ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, 8000 + i));
            ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, getGenericFullPathForSrcTest("/ovms/src/test/configs/config.json").c_str()));
            promisesThreadReady[i].set_value();
            futures[i].get();
            OVMS_Status* cstatus = OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings);
            uint32_t code = 0;
            OVMS_StatusCode(cstatus, &code);
            OVMS_StatusDelete(cstatus);
            retCodes[i] = code;
        }));
    }

    for (size_t i = 0; i < threadsCount; ++i) {
        futuresThreadReady[i].get();
    }
    for (size_t i = 0; i < threadsCount; ++i) {
        promises[i].set_value();
    }
    for (size_t i = 0; i < threadsCount; ++i) {
        threads[i]->join();
    }
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < threadsCount; ++i) {
        ss << retCodes[i] << ", ";
    }
    ss << "]";
    SPDLOG_ERROR("Error codes: {}", ss.str());
    SPDLOG_DEBUG("Will close server now");
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    OVMS_ServerDelete(cserver);
    SPDLOG_DEBUG("Closed server now");
    auto started = std::count_if(retCodes.begin(), retCodes.end(),
        [](const uint32_t v) {
            return v == (uint32_t)(ovms::StatusCode::OK);
        });
    auto alreadyStarting = std::count_if(retCodes.begin(), retCodes.end(),
        [](const uint32_t v) {
            return v == (uint32_t)(ovms::StatusCode::SERVER_ALREADY_STARTING);
        });
    auto alreadyStarted = std::count_if(retCodes.begin(), retCodes.end(),
        [](const uint32_t v) {
            return v == (uint32_t)(ovms::StatusCode::SERVER_ALREADY_STARTED);
        });
    SPDLOG_INFO("Started: {}, alreadyStarting: {}, alreadyStarted: {}", started, alreadyStarting, alreadyStarted);
    EXPECT_EQ(started, 1);
    EXPECT_EQ(alreadyStarted + alreadyStarting, threadsCount - 1);
}
struct CallbackUnblockingAndCheckingStruct : CallbackUnblockingStruct {
    std::promise<void> signal;
    float expectedValue{-1231571};
};

static void callbackCheckingIfErrorReported(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    SPDLOG_DEBUG("Using callback: callbackCheckingIfErrorReported!");
    EXPECT_NE(flag, 0);
    CallbackUnblockingAndCheckingStruct* callbackStruct = reinterpret_cast<CallbackUnblockingAndCheckingStruct*>(userStruct);
    callbackStruct->signal.set_value();
}
static void callbackUnblockingAndCheckingResponse(OVMS_InferenceResponse* response, uint32_t flag, void* userStruct) {
    EXPECT_EQ(flag, 0);
    SPDLOG_DEBUG("Using callback: callbackUnblockingAndFreeingRequest!");
    CallbackUnblockingAndCheckingStruct* callbackStruct = reinterpret_cast<CallbackUnblockingAndCheckingStruct*>(userStruct);
    callbackStruct->signal.set_value();
    checkDummyResponse(response, callbackStruct->expectedValue, 0);
    OVMS_InferenceResponseDelete(response);
}
const std::string DUMMY_MODEL_CPU_CONFIG_PATH{getGenericFullPathForSrcTest("/ovms/src/test/configs/config_cpu_dummy.json").c_str()};
class MockModelInstanceWithSetOutputInfo : public ovms::ModelInstance {
public:
    MockModelInstanceWithSetOutputInfo(ov::Core& ieCore) :
        ModelInstance(std::string("UNUSED_NAME"), 0, ieCore, nullptr, nullptr) {
        status = ovms::ModelVersionStatus("UNUSED_NAME", UNUSED_MODEL_VERSION, ovms::ModelVersionState::START);
    }
    virtual ~MockModelInstanceWithSetOutputInfo() {}
    ovms::Status loadModel(const ovms::ModelConfig& config) override {
        ModelInstance::loadModel(config);
        return ovms::StatusCode::OK;
    }
    void setOutputsInfo(const tensor_map_t& outputsInfo) {
        this->outputsInfo = outputsInfo;
    }
};

const float INITIAL_VALUE{0.13666};
const float GARBAGE_VALUE = 42.66613;
const float FLOAT_TOLERANCE{0.001};
TEST_F(CAPIInference, AsyncWithCallbackDummy) {
    std::vector<float> in(10, INITIAL_VALUE);
    std::vector<float> out(10, GARBAGE_VALUE);
    size_t inputByteSize = sizeof(float) * in.size();
    // start CAPI server
    bool startGrpc = true;  // WA to be able to test async inference but do not encounter closure issues CVS-164617
    ServerGuard serverGuard(DUMMY_MODEL_CPU_CONFIG_PATH, startGrpc);
    OVMS_Server* cserver = serverGuard.server;
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddOutput(request, DUMMY_MODEL_OUTPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::vector<float> data(DUMMY_MODEL_INPUT_SIZE, INITIAL_VALUE);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, in.data(), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestOutputSetData(request, DUMMY_MODEL_OUTPUT_NAME, out.data(), inputByteSize, OVMS_BUFFERTYPE_CPU, 0));
    // set callback
    CallbackUnblockingAndCheckingStruct callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    callbackStruct.bufferAddr = out.data();
    callbackStruct.expectedValue = INITIAL_VALUE + 1;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestSetCompletionCallback(request, callbackUnblockingAndCheckingResponse, reinterpret_cast<void*>(&callbackStruct)));
    // infer
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceAsync(cserver, request));
    // check
    unblockSignal.get();

    const float* outputData = reinterpret_cast<const float*>(out.data());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(in[i] + 1, outputData[i], FLOAT_TOLERANCE) << "Different at:" << i << " place.";
    }
    SPDLOG_INFO("Using callbacks!");
}
TEST_F(CAPIInference, AsyncErrorHandling) {
    ov::Core core;
    MockModelInstanceWithSetOutputInfo instance(core);
    instance.loadModel(DUMMY_MODEL_CONFIG);
    std::unique_ptr<ModelInstanceUnloadGuard> unloadGuard;  // we do not need it to be set
    ovms::InferenceRequest request("dummy", 0);
    std::vector<float> in(10, INITIAL_VALUE);
    request.addInput(DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    request.setInputBuffer(DUMMY_MODEL_INPUT_NAME, in.data(), DUMMY_MODEL_SHAPE[1] * sizeof(float), OVMS_BUFFERTYPE_CPU, 0);
    ovms::InferenceResponse response;
    auto outputInfo = instance.getOutputsInfo();
    outputInfo["NOT_EXISTING"] = std::make_shared<ovms::TensorInfo>("BADUMTSSS", ovms::Precision::UNDEFINED, shape_t{});
    instance.waitForLoaded(0, unloadGuard);
    CallbackUnblockingAndCheckingStruct callbackStruct;
    auto unblockSignal = callbackStruct.signal.get_future();
    request.setCompletionCallback(callbackCheckingIfErrorReported, &callbackStruct);
    instance.setOutputsInfo(outputInfo);
    auto status = ovms::modelInferAsync<ovms::InferenceRequest, ovms::InferenceResponse>(instance, &request, unloadGuard);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
    unblockSignal.get();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    unloadGuard.reset();
    instance.retireModel();
}
