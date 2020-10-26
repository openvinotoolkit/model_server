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

#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"

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

static void test_LoadModels() {
    spdlog::info("Loading model data for tests...");
}

::grpc::Status test_PerformModelStatusRequest(ModelServiceImpl& s, tensorflow::serving::GetModelStatusRequest& req, tensorflow::serving::GetModelStatusResponse& res) {
    auto config = DUMMY_MODEL_CONFIG;
    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    manager.reloadModelWithVersions(config);
    spdlog::info("req={} res={}", req.DebugString(), res.DebugString());
    ::grpc::Status ret = s.GetModelStatus(nullptr, &req, &res);
    spdlog::info("returned grpc status: ok={} code={} msg='{}'", ret.ok(), ret.error_code(), ret.error_details());
    return ret;
}

TEST(ModelService, empty_request) {
    test_LoadModels();
    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;
    ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
    EXPECT_EQ(ret.ok(), false);
}

/* Gated by ModelManager CRTP-based mock.

TEST(ModelService, single_version)
{
  test_LoadModels();
  ModelServiceImpl s;
  tensorflow::serving::GetModelStatusRequest req;
  tensorflow::serving::GetModelStatusResponse res;

  auto model_spec = req.mutable_model_spec();
  model_spec->Clear();
  model_spec->set_name("existing_model");
  model_spec->mutable_version()->set_value(1); // existing version

  ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
  EXPECT_EQ(ret.ok(), true);
}

TEST(ModelService, all_versions)
{
  test_LoadModels();
  ModelServiceImpl s;
  tensorflow::serving::GetModelStatusRequest req;
  tensorflow::serving::GetModelStatusResponse res;

  auto model_spec = req.mutable_model_spec();
  model_spec->Clear();
  model_spec->set_name("existing_model");
  // version field is not set, so we'll fetch all versions.

  ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
  EXPECT_EQ(ret.ok(), true);
}
*/

TEST(ModelService, non_existing_model) {
    test_LoadModels();
    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("non_existing_model");

    ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
    EXPECT_EQ(ret.ok(), false);
}

TEST(ModelService, non_existing_version) {
    test_LoadModels();
    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(9894689454358);  // non-existing version

    ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
    EXPECT_EQ(ret.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST(ModelService, negative_version) {
    test_LoadModels();
    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(-1);  // negative version

    ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
    EXPECT_EQ(ret.error_code(), grpc::StatusCode::NOT_FOUND);
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
