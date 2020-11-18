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

#include "../modelinstance.hpp"
#include "../pipelinedefinitionstatus.hpp"
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

TEST(PipelineDefinitionStatus, ValidationPassThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenUsedModelChangeThenPassThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenUsedModelChangeThriceThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenUsedModelChangeThriceThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(UsedModelChangedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetire) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenUsedModelChangeShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    ASSERT_THROW(pds.handle(UsedModelChangedEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenRetireShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    ASSERT_THROW(pds.handle(RetireEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenValidationPass) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
}

TEST(PipelineDefinitionStatus, ValidationPassThenThenRetireThenValidationFail) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, ValidationPassThenRetireThenValidationFail) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(RetireEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::RETIRED);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, RetireShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    ASSERT_THROW(pds.handle(RetireEvent()), std::logic_error);
}

TEST(PipelineDefinitionStatus, ValidationPassThenValidationPassShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
}

TEST(PipelineDefinitionStatus, ValidationPassThenValidationFailShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenValidationFailShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
}

TEST(PipelineDefinitionStatus, ValidationFailThenValidationPassShouldThrow) {
    PipelineDefinitionStatus pds(unusedPipelineName);
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::BEGIN);
    pds.handle(ValidationFailedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED);
    pds.handle(ValidationPassedEvent());
    ASSERT_EQ(pds.getStateCode(), ovms::PipelineDefinitionStateCode::AVAILABLE);
}
