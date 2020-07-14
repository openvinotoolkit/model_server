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

#include "../model_version_policy.hpp"

using namespace ovms;

using ::testing::UnorderedElementsAreArray;

struct ModelVersionPolicyParameter {
    std::shared_ptr<ModelVersionPolicy> policy;
    std::vector<model_version_t> filteredVersions;
    std::string name;
};

class ModelVersionPolicyFixture : public ::testing::TestWithParam<ModelVersionPolicyParameter> {
protected:
    std::vector<model_version_t> allAvailableVersions{3, 4, 5, 18, 25, 100};
};

TEST_P(ModelVersionPolicyFixture, Filter) {
    const auto& param = GetParam();
    EXPECT_THAT(param.policy->filter(allAvailableVersions), UnorderedElementsAreArray(param.filteredVersions));
}

const auto paramToString = [](const testing::TestParamInfo<ModelVersionPolicyParameter>& info) {
    return info.param.name;
};

INSTANTIATE_TEST_SUITE_P(
    DefaultModelVersionPolicy,
    ModelVersionPolicyFixture,
    ::testing::Values(
        ModelVersionPolicyParameter{ModelVersionPolicy::getDefaultVersionPolicy(), {100}, "ReturnsHighestVersion"}),
    paramToString);

INSTANTIATE_TEST_SUITE_P(
    LatestModelVersionPolicy,
    ModelVersionPolicyFixture,
    ::testing::Values(
        ModelVersionPolicyParameter{std::make_shared<LatestModelVersionPolicy>(), {100}, "DefaultReturnsHighest"},
        ModelVersionPolicyParameter{std::make_shared<LatestModelVersionPolicy>(1), {100}, "1_HighestVersion"},
        ModelVersionPolicyParameter{std::make_shared<LatestModelVersionPolicy>(2), {100, 25}, "2_HighestVersions"},
        ModelVersionPolicyParameter{std::make_shared<LatestModelVersionPolicy>(6), {100, 25, 18, 5, 4, 3}, "6_HighestVersions"},
        ModelVersionPolicyParameter{std::make_shared<LatestModelVersionPolicy>(10), {100, 25, 18, 5, 4, 3}, "10_HighestVersions"}),
    paramToString);

INSTANTIATE_TEST_SUITE_P(
    AllModelVersionPolicy,
    ModelVersionPolicyFixture,
    ::testing::Values(
        ModelVersionPolicyParameter{std::make_shared<AllModelVersionPolicy>(), {3, 4, 5, 18, 25, 100}, "All"}),
    paramToString);

static const std::vector<model_version_t> specificRequestedVersions[] = {
    {4, 25},
    {1, 8, 28},
    {4, 5, 6, 7},
    {4, 18, 100, 125}};

INSTANTIATE_TEST_SUITE_P(
    SpecificModelVersionPolicy,
    ModelVersionPolicyFixture,
    ::testing::Values(
        ModelVersionPolicyParameter{std::make_shared<SpecificModelVersionPolicy>(specificRequestedVersions[0]), {4, 25}, "Existing"},
        ModelVersionPolicyParameter{std::make_shared<SpecificModelVersionPolicy>(specificRequestedVersions[1]), {}, "NonExisting"},
        ModelVersionPolicyParameter{std::make_shared<SpecificModelVersionPolicy>(specificRequestedVersions[2]), {4, 5}, "ExistingAndNonExisting"}),
    paramToString);
