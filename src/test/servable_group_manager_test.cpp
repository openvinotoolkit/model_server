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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/model_management/servable_group_manager.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "src/model.hpp"
#include "src/model_management/servable_loading_queue.hpp"
#include "src/model_management/servable_loading_task.hpp"
#include "src/modelconfig.hpp"
#include "src/modelinstance.hpp"
#include "src/modelversionstatus.hpp"
#include "src/status.hpp"
#include "test_models.hpp"
#include "test_with_temp_dir.hpp"

using namespace ovms;

static std::unordered_map<std::string, ModelConfig> createModelConfigs(
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

class ServableGroupManagerTest : public ::testing::Test {
protected:
    ConstructorEnabledModelManager mm;
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

struct RecordedEvent {
    TaskEvent event;
    ServableLoadingTaskType type;
    std::string name;
    bool urgent;
};

class ServableGroupSwapTest : public TestWithTempDir {
protected:
    std::unique_ptr<ConstructorEnabledModelManager> mm;
    std::mutex recordMtx;
    std::vector<RecordedEvent> events;

    static std::string swapConfig() {
        return R"({"model_config_list": [
            {"config": {"name": "a1", "base_path": ")" +
               dummy_model_location + R"(", "target_device": "CPU", "nireq": 1, "group_name": "groupA"}},
            {"config": {"name": "a2", "base_path": ")" +
               dummy_model_location + R"(", "target_device": "CPU", "nireq": 1, "group_name": "groupA"}},
            {"config": {"name": "b1", "base_path": ")" +
               dummy_model_location + R"(", "target_device": "CPU", "nireq": 1, "group_name": "groupB"}},
            {"config": {"name": "b2", "base_path": ")" +
               dummy_model_location + R"(", "target_device": "CPU", "nireq": 1, "group_name": "groupB"}}
        ]})";
    }

    void SetUp() override {
        TestWithTempDir::SetUp();
        std::string configFilePath = directoryPath + "/config.json";
        std::ofstream(configFilePath) << swapConfig();

        mm = std::make_unique<ConstructorEnabledModelManager>(uint64_t{30'000'000});
        auto status = mm->loadConfig(configFilePath);
        ASSERT_TRUE(status.ok()) << status.string();
        groupManager = mm->getGroupManager();
        ASSERT_NE(groupManager, nullptr);
        ASSERT_TRUE(groupManager->getActiveGroupName().empty());

        // Installed after loadConfig so that only swap traffic is recorded.
        mm->getLoadingQueue().setTaskObserver(
            [this](TaskEvent event, const ServableLoadingTask& task) {
                std::lock_guard<std::mutex> lock(recordMtx);
                events.push_back({event, task.type, task.name, task.urgent});
            });
    }

    ServableGroupManager* groupManager = nullptr;

    void clearRecorded() {
        std::lock_guard<std::mutex> lock(recordMtx);
        events.clear();
    }

    std::vector<RecordedEvent> recorded(TaskEvent event) {
        std::lock_guard<std::mutex> lock(recordMtx);
        std::vector<RecordedEvent> filtered;
        std::copy_if(events.begin(), events.end(), std::back_inserter(filtered),
            [event](const RecordedEvent& e) { return e.event == event; });
        return filtered;
    }

    static bool isUnload(ServableLoadingTaskType type) {
        return type == ServableLoadingTaskType::PutToSleepModel ||
               type == ServableLoadingTaskType::PutToSleepMediapipe;
    }

    static size_t countLoadsOf(const std::vector<RecordedEvent>& tasks, const std::string& name) {
        return std::count_if(tasks.begin(), tasks.end(), [&name](const RecordedEvent& t) {
            return t.name == name && t.type == ServableLoadingTaskType::WakeUpModel;
        });
    }

    void ensureLoaded(const std::string& servableName) {
        auto status = groupManager->ensureServableLoaded(servableName, *mm);
        ASSERT_TRUE(status.ok()) << servableName << ": " << status.string();
        auto instance = mm->findModelByName(servableName)->getDefaultModelInstance();
        ASSERT_NE(instance, nullptr);
        EXPECT_EQ(instance->getStatus().getState(), ModelVersionState::AVAILABLE);
    }

    void expectRequestedLoadsFirst(const std::string& requested) {
        ASSERT_NO_FATAL_FAILURE(ensureLoaded(requested));

        auto executed = recorded(TaskEvent::Executed);
        ASSERT_FALSE(executed.empty());
        EXPECT_EQ(executed[0].name, requested)
            << "the servable that triggered the wake-up should load first to shorten "
               "time-to-first-response";
        EXPECT_TRUE(executed[0].urgent);
    }
};

TEST_F(ServableGroupSwapTest, SwapUnloadsPreviousGroupBeforeLoadingNew) {
    ASSERT_NO_FATAL_FAILURE(ensureLoaded("a1"));
    ASSERT_EQ(groupManager->getActiveGroupName(), "groupA");
    clearRecorded();

    ASSERT_NO_FATAL_FAILURE(ensureLoaded("b1"));
    auto tasks = recorded(TaskEvent::Executed);
    ASSERT_FALSE(tasks.empty());

    std::set<std::string> retired;
    size_t firstLoadIdx = tasks.size();
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (isUnload(tasks[i].type)) {
            retired.insert(tasks[i].name);
            EXPECT_LT(i, firstLoadIdx) << "unload of " << tasks[i].name << " ran after a load";
        } else if (i < firstLoadIdx) {
            firstLoadIdx = i;
        }
    }
    EXPECT_EQ(retired, (std::set<std::string>{"a1", "a2"}))
        << "every member of the previously active group must be unloaded on swap";
    EXPECT_EQ(groupManager->getActiveGroupName(), "groupB");
    for (const char* name : {"a1", "a2"}) {
        auto instance = mm->findModelByName(name)->getDefaultModelInstance();
        ASSERT_NE(instance, nullptr) << name << " must stay known so it can be woken up again";
        EXPECT_EQ(instance->getStatus().getState(), ModelVersionState::SLEEPING)
            << name << " must not be servable after its group was swapped out";
    }
}

TEST_F(ServableGroupSwapTest, RequestedServableIsScheduledOnlyOnce) {
    ASSERT_NO_FATAL_FAILURE(ensureLoaded("b1"));
    // Second request hits a group that is already active, so nothing should reload.
    ASSERT_NO_FATAL_FAILURE(ensureLoaded("b2"));

    auto tasks = recorded(TaskEvent::Scheduled);
    EXPECT_EQ(countLoadsOf(tasks, "b1"), 1u)
        << "loadGroup() already loads every group member, so ensureServableLoaded() "
           "must not schedule the requested servable a second time";
    EXPECT_EQ(countLoadsOf(tasks, "b2"), 1u)
        << "b2 was already loaded as part of groupB - requesting it must not reload it";
}

TEST_F(ServableGroupSwapTest, AlreadyAvailableServableDoesNotTouchLoadingQueue) {
    // we should try to load servable/push task to queue when its already loaded
    ASSERT_NO_FATAL_FAILURE(ensureLoaded("b1"));
    clearRecorded();

    for (int i = 0; i < 5; ++i) {
        ASSERT_NO_FATAL_FAILURE(ensureLoaded("b1"));
    }

    EXPECT_TRUE(recorded(TaskEvent::Scheduled).empty())
        << "requesting an already available servable must be answered without a "
           "loading queue round-trip";
    EXPECT_EQ(groupManager->getActiveGroupName(), "groupB");
}

// The requested servable must load first regardless of its position in the group's
// name ordering, so both directions are checked.
TEST_F(ServableGroupSwapTest, RequestedServableIsLoadedFirstWithinGroupLastAlphabetically) {
    expectRequestedLoadsFirst("a2");
}

TEST_F(ServableGroupSwapTest, RequestedServableIsLoadedFirstWithinGroupFirstAlphabetically) {
    expectRequestedLoadsFirst("a1");
}
