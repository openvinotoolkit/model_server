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

#include <gtest/gtest.h>

#include "../get_model_metadata_impl.hpp"
#include "../status.hpp"

class GetModelMetadataValidation : public ::testing::Test {
protected:
    tensorflow::serving::GetModelMetadataRequest request;

    void SetUp() override {
        request.Clear();
        request.mutable_model_spec()->set_name("ResNet50");
        request.mutable_metadata_field()->Add("signature_def");
    }
};

TEST_F(GetModelMetadataValidation, ValidRequestWithNoVersionSpecified) {
    auto status = ovms::GetModelMetadataImpl::validate(&request);
    EXPECT_TRUE(status.ok());
}

TEST_F(GetModelMetadataValidation, ValidRequestWithVersionSpecified) {
    request.mutable_model_spec()->mutable_version()->set_value(170);
    auto status = ovms::GetModelMetadataImpl::validate(&request);
    EXPECT_TRUE(status.ok());
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

TEST_F(GetModelMetadataValidation, RequestMissingModelSpec) {
    request.release_model_spec();
    auto status = ovms::GetModelMetadataImpl::validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::MODEL_SPEC_MISSING);
}

#pragma GCC diagnostic pop

TEST_F(GetModelMetadataValidation, RequestMissingMetadataField) {
    request.mutable_metadata_field()->RemoveLast();
    auto status = ovms::GetModelMetadataImpl::validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SIGNATURE_DEF);
}

TEST_F(GetModelMetadataValidation, RequestMetadataInvalidSignatureName) {
    request.mutable_metadata_field()->at(0) = "wrong_signature_name";
    auto status = ovms::GetModelMetadataImpl::validate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SIGNATURE_DEF);
}
