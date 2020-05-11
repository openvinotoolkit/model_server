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

#include "gtest/gtest.h"

#include "../modelversionstatus.hpp"
#include "../model_service.hpp"

#include <iostream>
#include <fstream>

#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#include "tensorflow_serving/apis/get_model_status.pb.h"

using namespace ovms;

TEST(ModelService, config_reload)
{
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
  spdlog::info("req={} res={}", req.DebugString(), res.DebugString());
  ::grpc::Status ret = s.GetModelStatus(nullptr, &req, &res);
  spdlog::info("returned grpc status: ok={} code={} msg='{}'", ret.ok(), ret.error_code(), ret.error_details()); 
  return ret;
}

TEST(ModelService, empty_request)
{
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

TEST(ModelService, non_existing_model)
{
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

TEST(ModelService, non_existing_version)
{
  test_LoadModels();
  ModelServiceImpl s;
  tensorflow::serving::GetModelStatusRequest req;
  tensorflow::serving::GetModelStatusResponse res;

  auto model_spec = req.mutable_model_spec();
  model_spec->Clear();
  model_spec->set_name("existing_model");
  model_spec->mutable_version()->set_value(9894689454358); // non-existing version

  ::grpc::Status ret = test_PerformModelStatusRequest(s, req, res);
  EXPECT_EQ(ret.ok(), false);
}


