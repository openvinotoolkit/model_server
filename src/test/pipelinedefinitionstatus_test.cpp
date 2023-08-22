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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../dags/pipelinedefinitionstatus.hpp"
#include "../modelinstance.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "../timer.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

using testing::_;
using testing::Return;

const std::string unusedPipelineName{"UNUSED_PIPELINE_NAME"};
const std::string unusedPipelineType{"UNUSED_PIPELINE_TYPE"};
const std::string modelNotifyingDetails{"Model:NonExisting version:i^2"};

TEST(PipelineDefinitionStatus, ValidationPassThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenUsedModelChangeThenPassThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenUsedModelChangeThriceThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenUsedModelChangeThriceThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenUsedModelChangeShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    ASSERT_THROW(pds.handle(UsedModelChangedEvent(modelNotifyingDetails)), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenRetireShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    ASSERT_THROW(pds.handle(RetireEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenRetireThenReloadThenValidationPass) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
}

TEST(PipelineDefinitionStatus, ValidationPassThenRetireThenReloadThenValidationFail) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, RetireShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    ASSERT_THROW(pds.handle(RetireEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenValidationPassShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationPassedEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenValidationFailShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationFailedEvent(modelNotifyingDetails)), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationFailThenValidationFailShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationFailedEvent(modelNotifyingDetails)), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationFailThenValidationPassShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationPassedEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationFailThenReloadThenValidationFail) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    pds.handle(ValidationFailedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenReloadThenValidationPass) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
}

TEST(PipelineDefinitionStatus, ValidationPassThenRetireThenValidationPassShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationPassedEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenRetireThenValidationFailShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    // should be reload in between
    ASSERT_THROW(pds.handle(ValidationFailedEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, ReloadFromBeginShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    ASSERT_THROW(pds.handle(ReloadEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, ReloadFromReloadShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    ASSERT_THROW(pds.handle(ReloadEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, UsedModelChangedFromBeginShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    ASSERT_THROW(pds.handle(UsedModelChangedEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, UsedModelChangedFromReloadShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    ASSERT_THROW(pds.handle(UsedModelChangedEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, RetireFromReloadShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(ReloadEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RELOADING);
    ASSERT_THROW(pds.handle(RetireEvent()), std::logic_error);
}
TEST(PipelineDefinitionStatus, ValidationFailFromFailedRequiredRevalidation) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}
TEST(PipelineDefinitionStatus, ConvertToModelStatus) {
    PipelineDefinitionStatus pds(unusedPipelineType, unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::LOADING, ModelVersionStatusErrorCode::OK)), pds.convertToModelStatus());
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::AVAILABLE, ModelVersionStatusErrorCode::OK)), pds.convertToModelStatus());
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::AVAILABLE, ModelVersionStatusErrorCode::OK)), pds.convertToModelStatus());
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::LOADING, ModelVersionStatusErrorCode::FAILED_PRECONDITION)), pds.convertToModelStatus());
    pds.handle(UsedModelChangedEvent(modelNotifyingDetails));
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::LOADING, ModelVersionStatusErrorCode::OK)), pds.convertToModelStatus());
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    ASSERT_EQ((std::tuple<ModelVersionState, ModelVersionStatusErrorCode>(ModelVersionState::END, ModelVersionStatusErrorCode::OK)), pds.convertToModelStatus());
}
