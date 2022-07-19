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


#include <gtest/gtest.h>
#include "../http_rest_api_handler.hpp"

using ovms::KFS_GetModelMetadata;
using ovms::KFS_GetModelReady;

TEST(HttpRestApiHandler, RegexParseReady){
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRestApiHandler handler(5);
    ovms::HttpRequestComponents comp;

    handler.parseRequestComponents(comp, "GET", request);

    ASSERT_EQ(comp.type, KFS_GetModelReady);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST(HttpRestApiHandler, RegexParseMetadata){
    std::string request = "/v2/models/dummy/versions/1/";
    ovms::HttpRestApiHandler handler(5);
    ovms::HttpRequestComponents comp;

    handler.parseRequestComponents(comp, "GET", request);

    ASSERT_EQ(comp.type, KFS_GetModelMetadata);
    ASSERT_EQ(comp.model_version, 1);
    ASSERT_EQ(comp.model_name, "dummy");
}

TEST(HttpRestApiHandler, dispatchMetadata){
    std::string request = "/v2/models/dummy/versions/1/";
    ovms::HttpRestApiHandler handler(5);
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler.registerHandler(KFS_GetModelMetadata, [&](const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelMetadata;
    std::string discard;
    handler.dispatchToProcessor(std::string(), &discard, comp);

    ASSERT_EQ(c, 1);
}

TEST(HttpRestApiHandler, dispatchReady){
    std::string request = "/v2/models/dummy/versions/1/ready";
    ovms::HttpRestApiHandler handler(5);
    ovms::HttpRequestComponents comp;
    int c = 0;

    handler.registerHandler(KFS_GetModelReady, [&](const ovms::HttpRequestComponents& request_components, std::string& response, const std::string& request_body) {
        c++;
        return ovms::StatusCode::OK;
    });
    comp.type = ovms::KFS_GetModelReady;
    std::string discard;
    handler.dispatchToProcessor(std::string(), &discard, comp);

    ASSERT_EQ(c, 1);
}