//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_calculators/modelapiovmsadapter.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

class MediapipeFlowTest : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";
    void SetUpServer(const char* configPath) {
        server.setShutdownRequest(0);
        randomizePort(this->port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!server.isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
    }

    void SetUp() override {
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class MediapipeFlowAddTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    }
};
class MediapipeFlowDummyTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full.json");
    }
};
class MediapipeFlowDummyPathsRelativeToBasePathTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_relative_to_base_path.json");
    }
};

class MediapipeFlowDummyNoGraphPathTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_no_graph_path.json");
    }
};

class MediapipeFlowDummyOnlyGraphNameSpecified : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_only_name_specified.json");
    }
};

class MediapipeFlowDummySubconfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_subconfig.json");
    }
};

class MediapipeFlowDummyDefaultSubconfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_default_subconfig.json");
    }
};

static void performMediapipeInferTest(const ovms::Server& server, ::KFSRequest& request, ::KFSResponse& response, const Precision& precision, const std::string& modelName) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
}

TEST_F(MediapipeFlowDummyOnlyGraphNameSpecified, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "graphdummy";
    performMediapipeInferTest(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyDefaultSubconfigTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInferTest(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummyPathsRelativeToBasePathTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInferTest(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowDummySubconfigTest, Infer) {
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaDummy";
    performMediapipeInferTest(server, request, response, precision, modelName);

    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

class MediapipeFlowDummyDummyInSubconfigAndConfigTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full_dummy_in_both_config_and_subconfig.json");
    }
};

TEST_F(MediapipeFlowDummyDummyInSubconfigAndConfigTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "mediaDummy";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {{1, 12}, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 12);
}

TEST_F(MediapipeFlowDummyNoGraphPathTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = "graphdummy";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // TODO validate output
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_P(MediapipeFlowDummyTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // TODO validate output
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_P(MediapipeFlowAddTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in1", {DUMMY_MODEL_SHAPE, precision}},
        {"in2", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData1{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
    std::vector<float> requestData2{0., 0., 0., 0., 0., 0., 0, 0., 0., 0};
    preparePredictRequest(request, inputsMeta, requestData1);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
    checkAddResponse("out", requestData1, requestData2, request, response, 1, 1, modelName);
}

TEST_F(MediapipeFlowTest, InferWithParams) {
    SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_graph_with_side_packets.json");
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediaWithParams";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in_not_used", {{1, 1}, ovms::Precision::I32}}};
    std::vector<float> requestData{0.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    // here add params
    const std::string stringParamValue = "abecadlo";
    const bool boolParamValue = true;
    const int64_t int64ParamValue = 42;
    request.mutable_parameters()->operator[]("string_param").set_string_param(stringParamValue);
    request.mutable_parameters()->operator[]("bool_param").set_bool_param(boolParamValue);
    request.mutable_parameters()->operator[]("int64_param").set_int64_param(int64ParamValue);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    // here check outputs
    ASSERT_EQ(outputs.size(), 3);
    // 1st string
    auto it = response.outputs().begin();
    size_t outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_string") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "UINT8");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), stringParamValue.size());
        const std::string& content = response.raw_output_contents(outputId);
        SPDLOG_ERROR("Received output size:{} content:{}", content.size(), content);
        EXPECT_EQ(content, stringParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
    it = response.outputs().begin();
    outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_bool") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "BOOL");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), 1);
        const std::string& content = response.raw_output_contents(outputId);
        ASSERT_EQ(content.size(), sizeof(bool));
        const bool castContent = *((bool*)content.data());
        SPDLOG_ERROR("Received output size:{} content:{}; castContent:{}", content.size(), content, castContent);
        EXPECT_EQ(castContent, boolParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
    it = response.outputs().begin();
    outputId = 0;
    while (it != response.outputs().end()) {
        if (it->name() != "out_int64") {
            ++it;
            ++outputId;
            continue;
        }
        ASSERT_EQ(it->datatype(), "INT64");
        ASSERT_EQ(it->shape_size(), 1);
        ASSERT_EQ(it->shape(0), 1);
        const std::string& content = response.raw_output_contents(outputId);
        ASSERT_EQ(content.size(), sizeof(int64_t));
        const int64_t castContent = *((int64_t*)content.data());
        SPDLOG_ERROR("Received output size:{} content:{}; castContent:{}", content.size(), content, castContent);
        EXPECT_EQ(castContent, int64ParamValue);
        break;
    }
    ASSERT_NE(it, response.outputs().end());
}

using testing::ElementsAre;

TEST_P(MediapipeFlowAddTest, AdapterMetadata) {
    const std::string modelName = "add";
    mediapipe::ovms::OVMSInferenceAdapter adapter(modelName);
    const std::shared_ptr<const ov::Model> model;
    ov::Core unusedCore;
    ov::AnyMap notUsedAnyMap;
    adapter.loadModel(model, unusedCore, "NOT_USED", notUsedAnyMap);
    EXPECT_THAT(adapter.getInputNames(), ElementsAre(SUM_MODEL_INPUT_NAME_1, SUM_MODEL_INPUT_NAME_2));
    EXPECT_THAT(adapter.getOutputNames(), ElementsAre(SUM_MODEL_OUTPUT_NAME));
    EXPECT_EQ(adapter.getInputShape(SUM_MODEL_INPUT_NAME_1), ov::Shape({1, 10}));
    EXPECT_EQ(adapter.getInputShape(SUM_MODEL_INPUT_NAME_2), ov::Shape({1, 10}));
}

namespace {
class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, ieCore) {
    }
    ov::AnyMap getRTInfo() const override {
        std::vector<std::string> mockLabels;
        for (size_t i = 0; i < 5; i++) {
            mockLabels.emplace_back(std::to_string(i));
        }
        ov::AnyMap configuration = {
            {"layout", "data:HWCN"},
            {"resize_type", "unnatural"},
            {"labels", mockLabels}};
        return configuration;
    }
};

class MockModel : public ovms::Model {
public:
    MockModel(const std::string& name) :
        Model(name, false /*stateful*/, nullptr) {}
    std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const ovms::model_version_t, ov::Core& ieCore, ovms::MetricRegistry* registry = nullptr, const ovms::MetricConfig* metricConfig = nullptr) override {
        return std::make_shared<MockModelInstance>(ieCore);
    }
};

class MockModelManager : public ovms::ModelManager {
    ovms::MetricRegistry registry;

public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name, const bool isStateful) override {
        return std::make_shared<MockModel>(name);
    }

public:
    MockModelManager(const std::string& modelCacheDirectory = "") :
        ovms::ModelManager(modelCacheDirectory, &registry) {
    }
    ~MockModelManager() {
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        join();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        models.clear();
        spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    }
};

class MockedServableManagerModule : public ovms::ServableManagerModule {
    mutable MockModelManager mockModelManager;

public:
    MockedServableManagerModule(ovms::Server& ovmsServer) :
        ovms::ServableManagerModule(ovmsServer) {
    }
    ModelManager& getServableManager() const override {
        return mockModelManager;
    }
};

class MockedServer : public Server {
public:
    MockedServer() = default;
    std::unique_ptr<Module> createModule(const std::string& name) override {
        if (name != ovms::SERVABLE_MANAGER_MODULE_NAME)
            return Server::createModule(name);
        return std::make_unique<MockedServableManagerModule>(*this);
    };
    Module* getModule(const std::string& name) {
        return const_cast<Module*>(Server::getModule(name));
    }
};

}  // namespace

TEST(Mediapipe, AdapterRTInfo) {
    MockedServer server;
    OVMS_Server* cserver = reinterpret_cast<OVMS_Server*>(&server);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    std::string port{"5555"};
    randomizePort(port);
    uint32_t portNum = ovms::stou32(port).value();
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, portNum));
    // we will use dummy model that will have mocked rt_info
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config.json"));

    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    const std::string mockedModelName = "dummy";
    uint32_t servableVersion = 1;
    mediapipe::ovms::OVMSInferenceAdapter adapter(mockedModelName, servableVersion, cserver);
    const std::shared_ptr<const ov::Model> model;
    /*ov::AnyMap configuration = {
        {"layout", "data:HWCN"},
        {"resize_type", "unnatural"},
        {"labels", mockLabels}*/
    ov::Core unusedCore;
    ov::AnyMap notUsedAnyMap;
    adapter.loadModel(model, unusedCore, "NOT_USED", notUsedAnyMap);
    ov::AnyMap modelConfig = adapter.getModelConfig();
    auto checkModelInfo = [](const ov::AnyMap& modelConfig) {
        ASSERT_EQ(modelConfig.size(), 3);
        auto it = modelConfig.find("resize_type");
        ASSERT_NE(modelConfig.end(), it);
        EXPECT_EQ(std::string("unnatural"), it->second.as<std::string>());
        it = modelConfig.find("layout");
        ASSERT_NE(modelConfig.end(), it);
        ASSERT_EQ(std::string("data:HWCN"), it->second.as<std::string>());
        it = modelConfig.find("labels");
        ASSERT_NE(modelConfig.end(), it);
        const std::vector<std::string>& resultLabels = it->second.as<std::vector<std::string>>();
        EXPECT_THAT(resultLabels, ElementsAre("0", "1", "2", "3", "4"));
    };
    checkModelInfo(modelConfig);

    OVMS_ServableMetadata* servableMetadata = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_GetServableMetadata(cserver, mockedModelName.c_str(), servableVersion, &servableMetadata));

    const ov::AnyMap* servableMetadataRtInfo;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServableMetadataGetInfo(servableMetadata, reinterpret_cast<const void**>(&servableMetadataRtInfo)));
    ASSERT_NE(nullptr, servableMetadataRtInfo);
    checkModelInfo(*servableMetadataRtInfo);
    OVMS_ServableMetadataDelete(servableMetadata);
}

TEST(Mediapipe, MetadataDummy) {
    ConstructorEnabledModelManager manager;
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", "/ovms/src/test/mediapipe/graphdummy.pbtxt"};
    ovms::MediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc);
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    tensor_map_t inputs = mediapipeDummy.getInputsInfo();
    tensor_map_t outputs = mediapipeDummy.getOutputsInfo();
    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("in"), inputs.end());
    ASSERT_NE(outputs.find("out"), outputs.end());
    const auto& input = inputs.at("in");
    EXPECT_EQ(input->getShape(), Shape({}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::UNDEFINED);
    const auto& output = outputs.at("out");
    EXPECT_EQ(output->getShape(), Shape({}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::UNDEFINED);
}

const std::vector<std::string> mediaGraphsDummy{"mediaDummy",
    "mediaDummyADAPTFULL"};
const std::vector<std::string> mediaGraphsAdd{"mediapipeAdd",
    "mediapipeAddADAPTFULL"};

class MediapipeConfig : public MediapipeFlowTest {
public:
    void TearDown() override {}
};

const std::string NAME = "Name";
TEST_F(MediapipeConfig, MediapipeGraphDefinitionNonExistentFile) {
    ConstructorEnabledModelManager manager;
    MediapipeGraphConfig mgc{"noname", "/ovms/NONEXISTENT_FILE"};
    MediapipeGraphDefinition mgd(NAME, mgc);
    EXPECT_EQ(mgd.validate(manager), StatusCode::FILE_INVALID);
}

TEST_F(MediapipeConfig, MediapipeAdd) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsAdd) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        EXPECT_NE(graphDefinition, nullptr);
        EXPECT_EQ(graphDefinition->getStatus().isAvailable(), true);
    }

    manager.join();
}

class MediapipeNoTagMapping : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();
    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";
    void SetUpServer(const char* configPath) {
        server.setShutdownRequest(0);
        randomizePort(this->port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!server.isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
        TestWithTempDir::TearDown();
    }
};

TEST_F(MediapipeNoTagMapping, DummyUppercase) {
    // Here we use dummy with uppercase input/output
    // and we shouldn't need tag mapping
    ConstructorEnabledModelManager manager;
    // create config file
    std::string configJson = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummyUpper",
                "base_path": "/ovms/src/test/dummyUppercase"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeDummyUppercase",
        "graph_path":"PATH_TO_REPLACE"
    }
    ]
})";
    const std::string pathToReplace = "PATH_TO_REPLACE";
    auto it = configJson.find(pathToReplace);
    ASSERT_NE(it, std::string::npos);
    const std::string graphPbtxt = R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "ModelAPISessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIOVMSSessionCalculatorOptions]: {
      servable_name: "dummyUpper"
      servable_version: "1"
    }
  }
}
node {
  calculator: "ModelAPISideFeedCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "B:in"
  output_stream: "A:out"
})";
    const std::string pbtxtPath = this->directoryPath + "/graphDummyUppercase.pbtxt";
    createConfigFileWithContent(graphPbtxt, pbtxtPath);
    configJson.replace(it, pathToReplace.size(), pbtxtPath);

    const std::string configJsonPath = this->directoryPath + "/subconfig.json";
    createConfigFileWithContent(configJson, configJsonPath);
    this->SetUpServer(configJsonPath.c_str());
    // INFER
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = "mediapipeDummyUppercase";
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    std::vector<float> requestData{0., 0., 0, 0., 0., 0., 0., 0, 0., 0.};
    checkDummyResponse("out", requestData, request, response, 1, 1, modelName);
}

TEST_F(MediapipeConfig, MediapipeFullRelativePaths) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfig) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_add_subconfig.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfigBasePath) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy_subconfig_base_path.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("graphaddadapterfull");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);
    auto model = manager.findModelByName("dummy1");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("graphadd");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);
    model = manager.findModelByName("dummy2");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsNegative) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy_negative.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), false);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), false);

    manager.join();
}

class MediapipeConfigChanges : public TestWithTempDir {
    void SetUp() override {
        TestWithTempDir::SetUp();
    }

public:
    static const std::string mgdName;
    static const std::string configFileWithGraphPathToReplace;
    static const std::string configFileWithGraphPathToReplaceWithoutModel;
    static const std::string configFileWithGraphPathToReplaceAndSubconfig;
    static const std::string configFileWithoutGraph;
    static const std::string pbtxtContent;
    template <typename Request, typename Response>
    static void checkStatus(ModelManager& manager, ovms::StatusCode code) {
        std::shared_ptr<MediapipeGraphExecutor> executor;
        Request request;
        Response response;
        auto status = manager.createPipeline(executor, mgdName, &request, &response);
        EXPECT_EQ(status, code) << status.string();
    }
};
const std::string MediapipeConfigChanges::mgdName{"mediapipeGraph"};
const std::string MediapipeConfigChanges::configFileWithGraphPathToReplace = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithGraphPathToReplaceAndSubconfig = R"(
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ",
        "subconfig":"SUBCONFIG_PATH"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithGraphPathToReplaceWithoutModel = R"(
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"mediapipeGraph",
        "graph_path":"XYZ"
    }
    ]
}
)";

const std::string MediapipeConfigChanges::configFileWithoutGraph = R"(
{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy"
        }
        }
    ]
}
)";

const std::string MediapipeConfigChanges::pbtxtContent = R"(
input_stream: "in"
output_stream: "out"
node {
  calculator: "ModelAPISessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIOVMSSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "ModelAPISideFeedCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "B:in"
  output_stream: "A:out"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIInferenceCalculatorOptions]: {
      tag_to_input_tensor_names {
        key: "B"
        value: "b"
      }
      tag_to_output_tensor_names {
        key: "A"
        value: "a"
      }
    }
  }
}
)";

TEST_F(MediapipeConfigChanges, AddProperGraphThenRetireThenAddAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/subconfig.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we retire
    configFileContent = configFileWithoutGraph;
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::RETIRED);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE);
    // now we add again
    configFileContent = configFileWithGraphPathToReplace;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

TEST_F(MediapipeConfigChanges, AddImroperGraphThenFixWithReloadThenBreakAgain) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/subconfig.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    ovms::Status status;
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
    // TODO check for tfs as well - now not supported
    // checkStatus<TFSPredictRequest, TFSPredictResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
    // now we fix the config
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we break
    configFileContent = configFileWithGraphPathToReplace;
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(MediapipeConfigChanges, AddModelToConfigThenUnloadThenAddToSubconfig) {
    std::string configFileContent = configFileWithGraphPathToReplace;
    std::string configFilePath = directoryPath + "/config.json";
    std::string graphFilePath = directoryPath + "/graph.pbtxt";
    const std::string modelPathToReplace{"XYZ"};
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    createConfigFileWithContent(pbtxtContent, graphFilePath);
    ConstructorEnabledModelManager modelManager;
    modelManager.loadConfig(configFilePath);
    const MediapipeFactory& factory = modelManager.getMediapipeFactory();
    auto model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    auto definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we retire the model
    configFileContent = configFileWithGraphPathToReplaceWithoutModel;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    model = modelManager.findModelByName("dummy");
    ASSERT_EQ(nullptr, model->getDefaultModelInstance());
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
    // now we add model to subconfig
    std::string subconfigFilePath = directoryPath + "/subconfig.json";
    SPDLOG_ERROR("{}", subconfigFilePath);
    configFileContent = configFileWithoutGraph;
    createConfigFileWithContent(configFileContent, subconfigFilePath);
    configFileContent = configFileWithGraphPathToReplaceAndSubconfig;
    configFileContent.replace(configFileContent.find(modelPathToReplace), modelPathToReplace.size(), graphFilePath);
    const std::string subconfigPathToReplace{"SUBCONFIG_PATH"};
    configFileContent.replace(configFileContent.find(subconfigPathToReplace), subconfigPathToReplace.size(), subconfigFilePath);
    createConfigFileWithContent(configFileContent, configFilePath);
    modelManager.loadConfig(configFilePath);
    model = modelManager.findModelByName("dummy");
    ASSERT_NE(nullptr, model->getDefaultModelInstance());
    ASSERT_EQ(model->getDefaultModelInstance()->getStatus().getState(), ModelVersionState::AVAILABLE);
    definition = factory.findDefinitionByName(mgdName);
    ASSERT_NE(nullptr, definition);
    ASSERT_EQ(definition->getStatus().getStateCode(), PipelineDefinitionStateCode::AVAILABLE);
    checkStatus<KFSRequest, KFSResponse>(modelManager, StatusCode::OK);
}

// TEST_F(MediapipeConfig, MediapipeFullRelativePathsSubconfigNegative) {
//     ConstructorEnabledModelManager manager;
//     auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_add_subconfig_negative.json");
//     EXPECT_EQ(status, ovms::StatusCode::JSON_INVALID);
//     manager.join();
// }

INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowAddTest,
    ::testing::ValuesIn(mediaGraphsAdd),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });
INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowDummyTest,
    ::testing::ValuesIn(mediaGraphsDummy),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });
