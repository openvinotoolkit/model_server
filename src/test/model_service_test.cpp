//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#pragma GCC diagnostic pop

#include "../model_service.hpp"
#include "../modelmanager.hpp"
#include "../modelversionstatus.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

using namespace ovms;

TEST(ModelService, config_reload) {
    ModelServiceImpl s;

    tensorflow::serving::ReloadConfigRequest req;
    tensorflow::serving::ReloadConfigResponse res;

    spdlog::info("req={} res={}", req.DebugString(), res.DebugString());
    ::grpc::Status ret = s.HandleReloadConfigRequest(nullptr, &req, &res);
    spdlog::info("returned grpc status: ok={} code={} msg='{}'", ret.ok(), ret.error_code(), ret.error_details());
    EXPECT_EQ(ret.ok(), true);
}

TEST(ModelService, empty_request) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::MODEL_NAME_MISSING);
}

TEST(ModelService, single_version_model) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);  // existing version

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::OK);

    ASSERT_EQ(res.model_version_status_size(), 1);
    ASSERT_EQ(res.model_version_status().begin()->state(), tensorflow::serving::ModelVersionStatus_State_AVAILABLE);
    ASSERT_EQ(res.model_version_status().begin()->has_status(), true);
    ASSERT_EQ(res.model_version_status().begin()->status().error_code(), tensorflow::error::OK);
    ASSERT_EQ(res.model_version_status().begin()->status().error_message(), "OK");
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

TEST(ModelService, pipeline) {
    std::string fileToReload = "/tmp/ovms_single_version_pipeline.json";
    createConfigFileWithContent(pipelineOneDummyConfig, fileToReload);
    ConstructorEnabledModelManager manager;
    ASSERT_EQ(manager.startFromFile(fileToReload), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto checkModelStatusResponse = [](tensorflow::serving::GetModelStatusResponse& response) {
        ASSERT_EQ(response.model_version_status_size(), 1);
        ASSERT_EQ(response.model_version_status().begin()->state(), tensorflow::serving::ModelVersionStatus_State_AVAILABLE);
        ASSERT_EQ(response.model_version_status().begin()->version(), 1);
        ASSERT_EQ(response.model_version_status().begin()->has_status(), true);
        ASSERT_EQ(response.model_version_status().begin()->status().error_code(), tensorflow::error::OK);
        ASSERT_EQ(response.model_version_status().begin()->status().error_message(), "OK");
    };

    // No version specified
    req.Clear();
    res.Clear();
    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummyPipeline");
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::OK);
    checkModelStatusResponse(res);

    // Version 1
    req.Clear();
    res.Clear();
    model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummyPipeline");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::OK);
    checkModelStatusResponse(res);

    // Any version
    req.Clear();
    res.Clear();
    model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummyPipeline");
    model_spec->mutable_version()->set_value(5);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::OK);
    checkModelStatusResponse(res);
}

class ModelServiceDummyWith2Versions : public ::testing::Test {
protected:
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
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    config.setBasePath(modelPath);
    config.setModelVersionPolicy(std::make_shared<AllModelVersionPolicy>());
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    // no version specified

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::OK);

    ASSERT_EQ(res.model_version_status_size(), 2);
    for (int i = 0; i < 2; i++) {
        auto& model_version_status = res.model_version_status()[i];
        ASSERT_EQ(model_version_status.state(), tensorflow::serving::ModelVersionStatus_State_AVAILABLE);
        ASSERT_EQ(model_version_status.version(), i + 1);
        ASSERT_EQ(model_version_status.has_status(), true);
        ASSERT_EQ(model_version_status.status().error_code(), tensorflow::error::OK);
        ASSERT_EQ(model_version_status.status().error_message(), "OK");
    }
}

TEST(ModelService, non_existing_model) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("non_existing_model");

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::MODEL_NAME_MISSING);
}

TEST(ModelService, non_existing_version) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(9894689454358);

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::MODEL_VERSION_MISSING);
}

TEST(ModelService, negative_version) {
    ConstructorEnabledModelManager manager;
    auto config = DUMMY_MODEL_CONFIG;
    ASSERT_EQ(manager.reloadModelWithVersions(config), StatusCode::OK);
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(-1);

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager), StatusCode::MODEL_VERSION_MISSING);
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
