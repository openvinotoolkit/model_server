//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include <iostream>
#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#pragma GCC diagnostic pop

#include "../dags/pipelinedefinition.hpp"
#include "../execution_context.hpp"
#include "../model_service.hpp"
#include "../model_version_policy.hpp"
#include "../modelmanager.hpp"
#include "../modelversionstatus.hpp"
#include "../server.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

using namespace ovms;

using TFSGetModelStatusRequest = tensorflow::serving::GetModelStatusRequest;
using TFSGetModelStatusResponse = tensorflow::serving::GetModelStatusResponse;
using TFSGetModelStatusInterface = std::pair<TFSGetModelStatusRequest, TFSGetModelStatusResponse>;
using KFSGetModelStatusInterface = std::pair<KFSGetModelStatusRequest, KFSGetModelStatusResponse>;

template <typename Pair,
    typename RequestType = typename Pair::first_type,
    typename ResponseType = typename Pair::second_type>
class ModelServiceTest : public ::testing::Test {
public:
    ConstructorEnabledModelManager manager;
    RequestType modelStatusRequest;
    ResponseType modelStatusResponse;
    void SetUp() {
        auto config = DUMMY_MODEL_CONFIG;
        ASSERT_EQ(this->manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);
        this->modelStatusRequest.Clear();
        this->modelStatusResponse.Clear();
    }
};

using MyTypes = ::testing::Types<
    TFSGetModelStatusInterface,
    KFSGetModelStatusInterface>;

TYPED_TEST_SUITE(ModelServiceTest, MyTypes);

static void executeModelStatus(const TFSGetModelStatusRequest& modelStatusRequest, TFSGetModelStatusResponse& modelStatusResponse, ModelManager& manager, ExecutionContext context, ovms::StatusCode statusCode = StatusCode::OK) {
    modelStatusResponse.Clear();
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&modelStatusRequest, &modelStatusResponse, manager, context), statusCode);
}

static void setModelStatusRequest(TFSGetModelStatusRequest& modelStatusRequest, const std::string& name, int version) {
    modelStatusRequest.Clear();
    auto model_spec = modelStatusRequest.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name(name);
    if (version) {
        model_spec->mutable_version()->set_value(version);
    }
}

static void verifyModelStatusResponse(const TFSGetModelStatusResponse& modelStatusResponse, const std::vector<int>& versions = {1}) {
    ASSERT_EQ(modelStatusResponse.model_version_status_size(), versions.size());
    for (size_t i = 0; i < versions.size(); i++) {
        auto& model_version_status = modelStatusResponse.model_version_status()[i];
        ASSERT_EQ(model_version_status.state(), tensorflow::serving::ModelVersionStatus_State_AVAILABLE);
        ASSERT_EQ(model_version_status.version(), versions[i]);
        ASSERT_EQ(model_version_status.has_status(), true);
        ASSERT_EQ(model_version_status.status().error_code(), tensorflow::error::OK);
        ASSERT_EQ(model_version_status.status().error_message(), "OK");
    }
}

static void verifyModelStatusResponse(const KFSGetModelStatusResponse& modelStatusResponse, const std::vector<int>& versions = {1}) {
    ASSERT_TRUE(modelStatusResponse.ready());
}

static void executeModelStatus(const KFSGetModelStatusRequest& modelStatusRequest, KFSGetModelStatusResponse& modelStatusResponse, ModelManager& manager, ExecutionContext context, ovms::StatusCode statusCode = StatusCode::OK) {
    modelStatusResponse.Clear();
    ASSERT_EQ(KFSInferenceServiceImpl::getModelReady(&modelStatusRequest, &modelStatusResponse, manager, context), statusCode);
}

static void setModelStatusRequest(KFSGetModelStatusRequest& modelStatusRequest, const std::string& name, int version) {
    modelStatusRequest.Clear();
    modelStatusRequest.set_name(name);
    if (version)
        modelStatusRequest.set_version(std::to_string(version));
}

TYPED_TEST(ModelServiceTest, empty_request) {
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT, StatusCode::MODEL_NAME_MISSING);
}

TYPED_TEST(ModelServiceTest, single_version_model) {
    const std::string name = "dummy";
    auto version = 1;  // existing version
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);
}

static const char* pipelineOneDummyConfig = R"(
{
    "model_config_list": [
        {
            "config": {
                "name": "dummy",
                "base_path": "/ovms/src/test/dummy",
                "target_device": "CPU",
                "model_version_policy": {"all": {}},
                "nireq": 1
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "dummyPipeline",
            "inputs": ["custom_dummy_input"],
            "nodes": [
                {
                    "name": "dummyNode",
                    "model_name": "dummy",
                    "type": "DL model",
                    "inputs": [
                        {"b": {"node_name": "request",
                               "data_item": "custom_dummy_input"}}
                    ], 
                    "outputs": [
                        {"data_item": "a",
                         "alias": "new_dummy_output"}
                    ] 
                }
            ],
            "outputs": [
                {"custom_dummy_output": {"node_name": "dummyNode",
                                         "data_item": "new_dummy_output"}
                }
            ]
        }
    ]
})";

TYPED_TEST(ModelServiceTest, pipeline) {
    std::string fileToReload = "/tmp/ovms_single_version_pipeline.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ASSERT_EQ(this->manager.startFromFile(fileToReload), StatusCode::OK);

    const std::string name = "dummyPipeline";

    // existing version
    int version = 1;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);

    // No version specified - with 0 version value is not set in helper function
    version = 0;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);

    // Any version
    version = 5;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);
}

#if (MEDIAPIPE_DISABLE == 0)
TYPED_TEST(ModelServiceTest, MediapipeGraph) {
    std::string fileToReload = "/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full.json";
    ASSERT_EQ(this->manager.startFromFile(fileToReload), StatusCode::OK);

    const std::string name = "mediaDummyADAPTFULL";

    // existing version
    int version = 1;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);

    // No version specified - with 0 version value is not set in helper function
    version = 0;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);

    // Any version
    version = 5;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(this->modelStatusResponse);
}
#endif

TYPED_TEST(ModelServiceTest, non_existing_model) {
    const std::string name = "non_existing_model";
    int version = 0;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT, StatusCode::MODEL_NAME_MISSING);
}

TYPED_TEST(ModelServiceTest, non_existing_version) {
    const std::string name = "dummy";
    int version = 989464;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT, StatusCode::MODEL_VERSION_MISSING);
}

TYPED_TEST(ModelServiceTest, negative_version) {
    const std::string name = "dummy";
    int version = -1;
    setModelStatusRequest(this->modelStatusRequest, name, version);
    executeModelStatus(this->modelStatusRequest, this->modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT, StatusCode::MODEL_VERSION_MISSING);
}

TEST(RestModelStatus, CreateGrpcRequestVersionSet) {
    std::string model_name = "dummy";
    const std::optional<int64_t> model_version = 1;
    tensorflow::serving::GetModelStatusRequest request_grpc;
    tensorflow::serving::GetModelStatusRequest* request_p = &request_grpc;
    Status status = GetModelStatusImpl::createGrpcRequest(model_name, model_version, request_p);
    bool has_requested_version = request_p->model_spec().has_version();
    auto requested_version = request_p->model_spec().version().value();
    std::string requested_model_name = request_p->model_spec().name();
    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(has_requested_version, true);
    EXPECT_EQ(requested_version, 1);
    EXPECT_EQ(requested_model_name, "dummy");
}

TEST(RestModelStatus, CreateGrpcRequestNoVersion) {
    std::string model_name = "dummy1";
    const std::optional<int64_t> model_version;
    tensorflow::serving::GetModelStatusRequest request_grpc;
    tensorflow::serving::GetModelStatusRequest* request_p = &request_grpc;
    Status status = GetModelStatusImpl::createGrpcRequest(model_name, model_version, request_p);
    bool has_requested_version = request_p->model_spec().has_version();
    std::string requested_model_name = request_p->model_spec().name();
    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(has_requested_version, false);
    EXPECT_EQ(requested_model_name, "dummy1");
}

TEST(RestModelStatus, serialize2Json) {
    const char* expected_json = R"({
 "model_version_status": [
  {
   "version": "2",
   "state": "START",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
)";
    tensorflow::serving::GetModelStatusResponse response;
    model_version_t requested_version = 2;
    const std::string& model_name = "dummy";
    ModelVersionStatus status = ModelVersionStatus(model_name, requested_version, ModelVersionState::START);
    addStatusToResponse(&response, requested_version, status);
    const tensorflow::serving::GetModelStatusResponse response_const = response;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json);
}

const ovms::ModelConfig DUMMY_MODEL_WITH_ONLY_NAME_CONFIG{
    "dummy",
};

// Some tests are specific for TFS because you can ask for more versions than one in one request
class ModelServiceDummyWith2Versions : public ::testing::Test {
protected:
    ConstructorEnabledModelManager manager;

    void SetUp() override {
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

        const std::string directoryName = std::string(test_info->test_suite_name());
        directoryPath = "/tmp/" + directoryName;
        modelPath = directoryPath + "/dummy";

        // Copy dummy model to temporary destination
        std::filesystem::remove_all(directoryPath);
        std::filesystem::create_directories(modelPath + "/1/");
        std::filesystem::create_directories(modelPath + "/2/");
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/1", std::filesystem::copy_options::recursive);
        std::filesystem::copy("/ovms/src/test/dummy/1", modelPath + "/2", std::filesystem::copy_options::recursive);
    }

    void TearDown() override {
        // Clean up temporary destination
        std::filesystem::remove_all(directoryPath);
    }

    std::string directoryPath;
    std::string modelPath;
};

TEST_F(ModelServiceDummyWith2Versions, all_versions) {
    tensorflow::serving::GetModelStatusRequest modelStatusRequest;
    tensorflow::serving::GetModelStatusResponse modelStatusResponse;
    auto config = DUMMY_MODEL_CONFIG;
    config.setBasePath(modelPath);
    config.setModelVersionPolicy(std::make_shared<AllModelVersionPolicy>());
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // no version specified
    const std::string name = "dummy";
    int version = 0;
    setModelStatusRequest(modelStatusRequest, name, version);
    executeModelStatus(modelStatusRequest, modelStatusResponse, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(modelStatusResponse, {1, 2});
}

TEST_F(ModelServiceDummyWith2Versions, getAllModelsStatuses_one_model_two_versions) {
    auto config = DUMMY_MODEL_WITH_ONLY_NAME_CONFIG;
    this->manager.reloadModelWithVersions(config);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, this->manager, DEFAULT_TEST_CONTEXT);
    EXPECT_EQ(modelsStatuses.size(), 1);
    EXPECT_EQ(modelsStatuses.begin()->second.model_version_status_size(), 0);

    config = DUMMY_MODEL_CONFIG;
    config.setBasePath(modelPath);
    config.setModelVersionPolicy(std::make_shared<AllModelVersionPolicy>());
    this->manager.reloadModelWithVersions(config);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatusesAfterReload;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatusesAfterReload, this->manager, DEFAULT_TEST_CONTEXT);

    ASSERT_EQ(modelsStatusesAfterReload.size(), 1);
    verifyModelStatusResponse(modelsStatusesAfterReload.begin()->second, {1, 2});
}

// Some tests are specific for TFS because you can ask for more versions than one in one request
using TFSModelServiceTest = ModelServiceTest<TFSGetModelStatusInterface>;

TEST_F(TFSModelServiceTest, getAllModelsStatuses_two_models_with_one_versions) {
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, this->manager, DEFAULT_TEST_CONTEXT);
    verifyModelStatusResponse(modelsStatuses.begin()->second);

    auto config = SUM_MODEL_CONFIG;
    this->manager.reloadModelWithVersions(config);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatusesAfterReload;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatusesAfterReload, this->manager, DEFAULT_TEST_CONTEXT);
    ASSERT_EQ(modelsStatusesAfterReload.size(), 2);
    auto dummyModelStatus = modelsStatusesAfterReload.find("dummy");
    auto sumModelStatus = modelsStatusesAfterReload.find("sum");
    ASSERT_NE(dummyModelStatus, modelsStatusesAfterReload.end());
    ASSERT_NE(sumModelStatus, modelsStatusesAfterReload.end());
    verifyModelStatusResponse(dummyModelStatus->second);
    verifyModelStatusResponse(sumModelStatus->second);
}

TEST_F(TFSModelServiceTest, config_reload) {
    std::string port = "9000";
    randomizePort(port);
    char* argv[] = {
        (char*)"OpenVINO Model Server",
        (char*)"--model_name",
        (char*)"dummy",
        (char*)"--model_path",
        (char*)"/ovms/src/test/dummy",
        (char*)"--log_level",
        (char*)"DEBUG",
        (char*)"--port",
        (char*)port.c_str(),
        nullptr};
    ovms::Server& server = ovms::Server::instance();
    std::thread t([&argv, &server]() {
        ASSERT_EQ(EXIT_SUCCESS, server.start(9, argv));
    });
    auto start = std::chrono::high_resolution_clock::now();
    while ((server.getModuleState(ovms::GRPC_SERVER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
           (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
    }
    ModelServiceImpl s(server);
    tensorflow::serving::ReloadConfigRequest modelStatusRequest;
    tensorflow::serving::ReloadConfigResponse modelStatusResponse;

    spdlog::info("req={} this->res={}", this->modelStatusRequest.DebugString(), this->modelStatusResponse.DebugString());
    ::grpc::Status ret = s.HandleReloadConfigRequest(nullptr, &modelStatusRequest, &modelStatusResponse);
    spdlog::info("returned grpc status: ok={} code={} msg='{}'", ret.ok(), ret.error_code(), ret.error_details());
    EXPECT_EQ(ret.ok(), true);
    server.setShutdownRequest(1);
    t.join();
    server.setShutdownRequest(0);
}

TEST_F(TFSModelServiceTest, getAllModelsStatuses_one_model_one_version) {
    ConstructorEnabledModelManager manager;  // intentionally uses separate manager as we don't want any unloaded models
    auto config = DUMMY_MODEL_WITH_ONLY_NAME_CONFIG;
    manager.reloadModelWithVersions(config);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatuses, manager, DEFAULT_TEST_CONTEXT);
    EXPECT_EQ(modelsStatuses.size(), 1);
    EXPECT_EQ(modelsStatuses.begin()->second.model_version_status_size(), 0);

    config = DUMMY_MODEL_CONFIG;
    manager.reloadModelWithVersions(config);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatusesAfterReload;
    GetModelStatusImpl::getAllModelsStatuses(modelsStatusesAfterReload, manager, DEFAULT_TEST_CONTEXT);

    ASSERT_EQ(modelsStatusesAfterReload.size(), 1);
    verifyModelStatusResponse(modelsStatusesAfterReload.begin()->second);
}

TEST_F(TFSModelServiceTest, serializeModelsStatuses2Json_with_one_response) {
    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "2",
   "state": "START",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    tensorflow::serving::GetModelStatusResponse modelStatusResponse;
    model_version_t requestedVersion = 2;
    const std::string& model_name = "dummy";
    ModelVersionStatus modelStatus = ModelVersionStatus(model_name, requestedVersion, ModelVersionState::START);
    addStatusToResponse(&modelStatusResponse, requestedVersion, modelStatus);
    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    modelsStatuses.insert(std::pair<std::string, tensorflow::serving::GetModelStatusResponse>("dummy", modelStatusResponse));
    std::string jsonOutput;
    Status status = GetModelStatusImpl::serializeModelsStatuses2Json(modelsStatuses, jsonOutput);
    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(jsonOutput, expectedJson);
}

TEST_F(TFSModelServiceTest, serializeModelsStatuses2Json_with_two_responses) {
    const char* expectedJson = R"({
"dummy1" : 
{
 "model_version_status": [
  {
   "version": "2",
   "state": "START",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
},
"dummy2" : 
{
 "model_version_status": [
  {
   "version": "3",
   "state": "LOADING",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    tensorflow::serving::GetModelStatusResponse firstResponse;
    model_version_t requestedVersion = 2;
    const std::string& modelName1 = "dummy1";
    ModelVersionStatus modelStatus = ModelVersionStatus(modelName1, requestedVersion, ModelVersionState::START);
    addStatusToResponse(&firstResponse, requestedVersion, modelStatus);

    tensorflow::serving::GetModelStatusResponse secondResponse;
    requestedVersion = 3;
    const std::string& modelName2 = "dummy2";
    modelStatus = ModelVersionStatus(modelName2, requestedVersion, ModelVersionState::LOADING);
    addStatusToResponse(&secondResponse, requestedVersion, modelStatus);

    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    modelsStatuses.insert(std::pair<std::string, tensorflow::serving::GetModelStatusResponse>("dummy1", firstResponse));
    modelsStatuses.insert(std::pair<std::string, tensorflow::serving::GetModelStatusResponse>("dummy2", secondResponse));

    std::string jsonOutput;
    Status status = GetModelStatusImpl::serializeModelsStatuses2Json(modelsStatuses, jsonOutput);
    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(jsonOutput, expectedJson);
}

TEST_F(TFSModelServiceTest, serializeModelsStatuses2Json_one_response_with_two_versions) {
    const char* expectedJson = R"({
"dummy" : 
{
 "model_version_status": [
  {
   "version": "2",
   "state": "START",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  },
  {
   "version": "3",
   "state": "LOADING",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
})";
    tensorflow::serving::GetModelStatusResponse response;
    model_version_t requestedVersion = 2;
    const std::string& modelName = "dummy";
    ModelVersionStatus modelStatus = ModelVersionStatus(modelName, requestedVersion, ModelVersionState::START);
    addStatusToResponse(&response, requestedVersion, modelStatus);

    requestedVersion = 3;
    modelStatus = ModelVersionStatus(modelName, requestedVersion, ModelVersionState::LOADING);
    addStatusToResponse(&response, requestedVersion, modelStatus);

    std::map<std::string, tensorflow::serving::GetModelStatusResponse> modelsStatuses;
    modelsStatuses.insert(std::pair<std::string, tensorflow::serving::GetModelStatusResponse>("dummy", response));

    std::string jsonOutput;
    Status status = GetModelStatusImpl::serializeModelsStatuses2Json(modelsStatuses, jsonOutput);
    ASSERT_EQ(status, StatusCode::OK);
    EXPECT_EQ(jsonOutput, expectedJson);
}
