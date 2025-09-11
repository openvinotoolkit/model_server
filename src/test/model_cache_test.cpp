//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <filesystem>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../modelmanager.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "test_models_configs.hpp"
#include "test_with_temp_dir.hpp"

using namespace ovms;

class ModelCacheTest : public TestWithTempDir {
protected:
    void SetUp() override {
        TestWithTempDir::SetUp();
        modelCacheDirectory = this->directoryPath;
        dummyModelConfigWithCache = DUMMY_MODEL_CONFIG;
        dummyModelConfigWithCache.setCacheDir(modelCacheDirectory);
        dummyModelConfigWithCache.setBatchSize(std::nullopt);
        imageModelConfigWithCache = INCREMENT_1x3x4x5_MODEL_CONFIG;
        imageModelConfigWithCache.setCacheDir(modelCacheDirectory);
        imageModelConfigWithCache.setBatchSize(std::nullopt);
    }

    size_t getCachedFileCount() {
        auto it = std::filesystem::directory_iterator(this->directoryPath);
        return std::count_if(std::filesystem::begin(it), std::filesystem::end(it), [](auto& entry) { return entry.is_regular_file(); });
    }

    void prepareDummyCachedRun() {
        ModelConfig config = dummyModelConfigWithCache;
        auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
        ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    }

    void prepareImageModelCachedRun() {
        ModelConfig config = imageModelConfigWithCache;
        auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
        ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);
    }

    std::string modelCacheDirectory;
    ModelConfig dummyModelConfigWithCache;
    ModelConfig imageModelConfigWithCache;
};

// This test imitates reloading configuration at service runtime.
TEST_F(ModelCacheTest, FlowTestOnlineModifications) {
    ModelConfig config = dummyModelConfigWithCache;
    ASSERT_EQ(config.parseShapeParameter("(1,10)"), StatusCode::OK);

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // Check if cache files were created.
    size_t currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, 0);

    // Reload dummy with no change.
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK);

    // Check that no cache files were created.
    ASSERT_EQ(this->getCachedFileCount(), currentCacheFileCount);

    // Restart manager with cache directory specified.
    // Load dummy model with changed shape.
    ModelConfig config_1x100 = dummyModelConfigWithCache;
    ASSERT_EQ(config_1x100.parseShapeParameter("(1,100)"), StatusCode::OK);
    ASSERT_EQ(manager->reloadModelWithVersions(config_1x100), StatusCode::OK_RELOADED);

    // Check if new cache files were created.
    size_t previousCacheFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, previousCacheFileCount);

    // Start manager with cache directory specified.
    // Load dummy model with initial shape.
    config = dummyModelConfigWithCache;
    ASSERT_EQ(config.parseShapeParameter("(1,10)"), StatusCode::OK);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // // Check if no new cache file was created.
    // previousCacheFileCount = currentCacheFileCount;
    // currentCacheFileCount = this->getCachedFileCount();
    // ASSERT_EQ(currentCacheFileCount, previousCacheFileCount);
}

// This test imitates restarting the service.
TEST_F(ModelCacheTest, FlowTestOfflineModifications) {
    ModelConfig config = DUMMY_MODEL_CONFIG;

    // Start manager with no cache directory specified.
    auto manager = std::make_unique<ConstructorEnabledModelManager>();
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // Check if no cache files were created.
    ASSERT_EQ(this->getCachedFileCount(), 0);
    manager.reset();

    // Start manager with cache directory specified.
    manager.reset(new ConstructorEnabledModelManager(modelCacheDirectory));
    config = dummyModelConfigWithCache;
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // Check if any cache file was created.
    size_t currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GE(currentCacheFileCount, 1);
    manager.reset();

    // Restart manager with cache directory specified.
    manager.reset(new ConstructorEnabledModelManager(modelCacheDirectory));
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // Check if no cache file was created.
    ASSERT_EQ(this->getCachedFileCount(), currentCacheFileCount);
    manager.reset();

    // Restart manager with cache directory specified.
    // Load dummy model with changed shape.
    ModelConfig config_1x100 = dummyModelConfigWithCache;
    ASSERT_EQ(config_1x100.parseShapeParameter("(1,100)"), StatusCode::OK);
    manager.reset(new ConstructorEnabledModelManager(modelCacheDirectory));
    ASSERT_EQ(manager->reloadModelWithVersions(config_1x100), StatusCode::OK_RELOADED);

    // Check if new cache files were created.
    size_t previousCacheFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, previousCacheFileCount);

    // Start manager with cache directory specified.
    // Load dummy model with initial shape.
    manager.reset(new ConstructorEnabledModelManager(modelCacheDirectory));
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    // Check if no new cache file was created.
    previousCacheFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, previousCacheFileCount);
}

TEST_F(ModelCacheTest, BatchSizeChangeImpactsCache) {
    this->prepareDummyCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = dummyModelConfigWithCache;
    config.setBatchSize(5);

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    size_t lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, lastCachedFileCount);

    manager.reset();
    manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);
}

TEST_F(ModelCacheTest, ShapeChangeImpactsCache) {
    this->prepareDummyCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = dummyModelConfigWithCache;
    config.setBatchSize(std::nullopt);
    config.parseShapeParameter("(1,100)");

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    size_t lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, lastCachedFileCount);

    manager.reset();
    manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);
}

TEST_F(ModelCacheTest, NireqChangeDoesNotImpactCache) {
    this->prepareDummyCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = dummyModelConfigWithCache;
    config.setNireq(12);

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    ASSERT_EQ(this->getCachedFileCount(), currentCacheFileCount);
}

TEST_F(ModelCacheTest, LayoutChangeDoesImpactCache) {
    this->prepareImageModelCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = imageModelConfigWithCache;
    config.parseLayoutParameter("nhwc:nchw");

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    size_t lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_GT(currentCacheFileCount, lastCachedFileCount);

    manager.reset();
    manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);
}

TEST_F(ModelCacheTest, PluginConfigChangeDoesNotImpactCache) {
    this->prepareImageModelCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = imageModelConfigWithCache;
    config.setPluginConfig({{"NUM_STREAMS", "21"}});

    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    size_t lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);

    manager.reset();
    manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);
}

TEST_F(ModelCacheTest, CacheDisabledModelConfig) {
    this->prepareDummyCachedRun();
    size_t currentCacheFileCount = this->getCachedFileCount();

    ModelConfig config = dummyModelConfigWithCache;
    config.setBatchSize(ovms::Mode::AUTO);
    auto manager = std::make_unique<ConstructorEnabledModelManager>(modelCacheDirectory);
    ASSERT_EQ(manager->reloadModelWithVersions(config), StatusCode::OK_RELOADED);

    size_t lastCachedFileCount = currentCacheFileCount;
    currentCacheFileCount = this->getCachedFileCount();
    ASSERT_EQ(currentCacheFileCount, lastCachedFileCount);
}

class TestModelCacheSetting : public TestWithTempDir {
protected:
    std::unique_ptr<ov::Core> ieCore;

    void SetUp() override {
        TestWithTempDir::SetUp();
        ieCore = std::make_unique<ov::Core>();
        modelCacheDirectory = this->directoryPath;
        config = DUMMY_MODEL_CONFIG;
        config.setCacheDir(modelCacheDirectory);
        config.setBatchSize(std::nullopt);
    }
    ovms::ModelConfig config;
    std::string modelCacheDirectory;
};

TEST_F(TestModelCacheSetting, CacheNotDisabledWithDefaultConfig) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.setCacheDir("");
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_FALSE(modelInstance.isCacheDisabled());
}

TEST_F(TestModelCacheSetting, CacheDisabledWithCustomLoader) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.addCustomLoaderOption("test", "loader");
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());

    config.setAllowCache(false);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());
}

TEST_F(TestModelCacheSetting, CacheDisabledWithBatchAuto) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.setBatchingMode(ovms::Mode::AUTO);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());

    config.setAllowCache(false);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());
}

TEST_F(TestModelCacheSetting, CacheDisabledWithAnyShapeAuto) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.parseShapeParameter("auto");
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());

    config.setAllowCache(false);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_TRUE(modelInstance.isCacheDisabled());
}

TEST_F(TestModelCacheSetting, CacheCannotBeEnabledWithCustomLoader) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.addCustomLoaderOption("test", "loader");
    config.setAllowCache(true);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::ALLOW_CACHE_WITH_CUSTOM_LOADER);
}

TEST_F(TestModelCacheSetting, CacheCanBeEnabledWithBatchAuto) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.setBatchingMode(ovms::Mode::AUTO);
    config.setAllowCache(true);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_FALSE(modelInstance.isCacheDisabled());
}

TEST_F(TestModelCacheSetting, CacheCanBeEnabledWithAnyShapeAuto) {
    ovms::ModelInstance modelInstance("UNUSED_NAME", UNUSED_MODEL_VERSION, *ieCore);
    config.parseShapeParameter("auto");
    config.setAllowCache(true);
    EXPECT_EQ(modelInstance.setCacheOptions(config), ovms::StatusCode::OK);
    EXPECT_FALSE(modelInstance.isCacheDisabled());
}
