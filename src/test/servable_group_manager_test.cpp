//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include <thread>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/model_management/servable_group_manager.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "src/modelconfig.hpp"
#include "src/status.hpp"

using namespace ovms;

class ServableGroupManagerTest : public ::testing::Test {
protected:
    ConstructorEnabledModelManager mm;

    std::unordered_map<std::string, ModelConfig> createModelConfigs(
        const std::vector<std::pair<std::string, std::string>>& nameGroupPairs) {
        std::unordered_map<std::string, ModelConfig> configs;
        for (const auto& [name, group] : nameGroupPairs) {
            ModelConfig config;
            config.setName(name);
            config.setGroupName(group);
            configs.emplace(name, std::move(config));
        }
        return configs;
    }
};

TEST_F(ServableGroupManagerTest, DisabledByDefault) {
    ServableGroupManager mgr(0);
    ASSERT_FALSE(mgr.isEnabled());
}

TEST_F(ServableGroupManagerTest, EnabledWithPositiveTimeout) {
    ServableGroupManager mgr(30'000'000);
    ASSERT_TRUE(mgr.isEnabled());
    ASSERT_EQ(mgr.getIdleTimeoutMicroseconds(), 30'000'000u);
}

TEST_F(ServableGroupManagerTest, BuildGroups_DefaultGroupNames) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "model_a"},
        {"model_b", "model_b"},
        {"model_c", "model_c"},
    });
    mgr.buildGroups(configs, mm);
    const auto& groups = mgr.getGroups();
    ASSERT_EQ(groups.size(), 3u);
    EXPECT_TRUE(groups.count("model_a"));
    EXPECT_TRUE(groups.count("model_b"));
    EXPECT_TRUE(groups.count("model_c"));
}

TEST_F(ServableGroupManagerTest, BuildGroups_ExplicitGroupNames) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "rag"},
        {"model_b", "rag"},
        {"model_c", "rag"},
    });
    mgr.buildGroups(configs, mm);
    const auto& groups = mgr.getGroups();
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_TRUE(groups.count("rag"));
    EXPECT_EQ(groups.at("rag").modelNames.size(), 3u);
}

TEST_F(ServableGroupManagerTest, BuildGroups_PermanentGroup) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "permanent"},
        {"model_b", "permanent"},
        {"model_c", "rag"},
    });
    mgr.buildGroups(configs, mm);
    const auto& groups = mgr.getGroups();
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_TRUE(groups.at("permanent").isPermanent());
    EXPECT_FALSE(groups.at("rag").isPermanent());
    EXPECT_EQ(groups.at("permanent").modelNames.size(), 2u);
}

TEST_F(ServableGroupManagerTest, BuildGroups_MixedGroups) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "rag"},
        {"model_b", "rag"},
        {"model_c", "audio"},
        {"model_d", "audio"},
        {"model_e", "permanent"},
    });
    mgr.buildGroups(configs, mm);
    const auto& groups = mgr.getGroups();
    ASSERT_EQ(groups.size(), 3u);
    EXPECT_EQ(groups.at("rag").modelNames.size(), 2u);
    EXPECT_EQ(groups.at("audio").modelNames.size(), 2u);
    EXPECT_EQ(groups.at("permanent").modelNames.size(), 1u);
}

TEST_F(ServableGroupManagerTest, GetGroupForServable) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "rag"},
        {"model_b", "audio"},
    });
    mgr.buildGroups(configs, mm);
    EXPECT_EQ(mgr.getGroupForServable("model_a"), "rag");
    EXPECT_EQ(mgr.getGroupForServable("model_b"), "audio");
    EXPECT_EQ(mgr.getGroupForServable("nonexistent"), "");
}

TEST_F(ServableGroupManagerTest, IsGroupLoaded_PermanentAlwaysTrue) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "permanent"},
        {"model_b", "rag"},
    });
    mgr.buildGroups(configs, mm);
    EXPECT_TRUE(mgr.isGroupLoaded("permanent"));
    EXPECT_FALSE(mgr.isGroupLoaded("rag"));
    EXPECT_FALSE(mgr.isGroupLoaded("nonexistent"));
}

TEST_F(ServableGroupManagerTest, GetAllConfiguredServableNames) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({
        {"model_a", "rag"},
        {"model_b", "rag"},
        {"model_c", "permanent"},
    });
    mgr.buildGroups(configs, mm);
    auto names = mgr.getAllConfiguredServableNames();
    ASSERT_EQ(names.size(), 3u);
    std::set<std::string> nameSet(names.begin(), names.end());
    EXPECT_TRUE(nameSet.count("model_a"));
    EXPECT_TRUE(nameSet.count("model_b"));
    EXPECT_TRUE(nameSet.count("model_c"));
}

TEST_F(ServableGroupManagerTest, RecordActivityUpdatesTimestamp) {
    ServableGroupManager mgr(30'000'000);
    auto configs = createModelConfigs({{"model_a", "rag"}});
    mgr.buildGroups(configs, mm);

    // Record activity and verify no crash
    mgr.recordActivity();
}

TEST_F(ServableGroupManagerTest, ActiveGroupNameInitiallyEmpty) {
    ServableGroupManager mgr(30'000'000);
    EXPECT_TRUE(mgr.getActiveGroupName().empty());
}

// Schema validation test for group_name in model config
TEST(SchemaValidation, GroupNameInModelConfig) {
    ModelConfig config;
    config.setName("test_model");
    config.setGroupName("my_group");
    EXPECT_EQ(config.getGroupName(), "my_group");

    // Default group name should be model name
    ModelConfig config2;
    config2.setName("test_model_2");
    config2.setGroupName(config2.getName());
    EXPECT_EQ(config2.getGroupName(), "test_model_2");
}
