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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "../dags/custom_node_library_manager.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "platform_utils.hpp"
#include "light_test_utils.hpp"
#include "test_with_temp_dir.hpp"

using namespace ovms;

TEST(NodeLibraryManagerTest, NewManagerExpectMissingLibrary) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.getLibrary("random_name", library);
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_MISSING);
}

TEST(NodeLibraryManagerTest, UnSuccessfullLibraryLoading) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.loadLibrary("random_name", "ovms/bazel-bin/src/lib_node_mock.so");
    ASSERT_EQ(status, StatusCode::PATH_INVALID);
}

TEST(NodeLibraryManagerTest, SuccessfullLibraryLoadingAndExecution) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getLibrary("random_name", library);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(library.initialize, nullptr);
    ASSERT_NE(library.deinitialize, nullptr);
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.getInputsInfo, nullptr);
    ASSERT_NE(library.getOutputsInfo, nullptr);
    ASSERT_NE(library.release, nullptr);
    EXPECT_EQ(library.initialize(nullptr, nullptr, 0), 0);
    EXPECT_EQ(library.deinitialize(nullptr), 0);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr), 1);
    EXPECT_EQ(library.getInputsInfo(nullptr, nullptr, nullptr, 0, nullptr), 2);
    EXPECT_EQ(library.getOutputsInfo(nullptr, nullptr, nullptr, 0, nullptr), 3);
    EXPECT_EQ(library.release(nullptr, nullptr), 4);
}

TEST(NodeLibraryManagerTest, LibraryLoadingDuplicateNameAndBasePath) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_ALREADY_LOADED);
}

TEST(NodeLibraryManagerTest, LibraryReloadingDuplicateNameAndDifferentBasePath) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_add_sub.so"));
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(NodeLibraryManagerTest, LibraryLoadingDuplicatePath) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("library_A", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.loadLibrary("library_B", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(NodeLibraryManagerTest, LibraryLoadingMissingImplementation) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_missing_implementation.so"));
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM);
}

TEST(NodeLibraryManagerTest, TryLoadingCorruptedLibraryNextLoadCorrectLibrary) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_missing_implementation.so"));
    ASSERT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM);
    status = manager.loadLibrary("random_name", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/lib_node_mock.so"));
    EXPECT_EQ(status, StatusCode::OK);
}

TEST(NodeLibraryManagerTest, LibraryLoadingMissingFile) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", getGenericFullPathForTmp("/tmp/non_existing_library_file"));
    EXPECT_EQ(status, StatusCode::NODE_LIBRARY_LOAD_FAILED_OPEN);
}

TEST(NodeLibraryManagerTest, ErrorWhenLibraryPathNotEscaped) {
    CustomNodeLibraryManager manager;
    auto status = manager.loadLibrary("random_name", "/tmp/../my_dir/non_existing_library_file");
    EXPECT_EQ(status, StatusCode::PATH_INVALID);
}

TEST(NodeLibraryManagerTest, ModelZooObjectDetectionCapsAllOutputsToMaxOutputBatch) {
    CustomNodeLibraryManager manager;
    NodeLibrary library;
    auto status = manager.loadLibrary("model_zoo_object_detection", getGenericFullPathForBazelOut("/ovms/bazel-bin/src/libcustom_node_model_zoo_intel_object_detection.so"));
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getLibrary("model_zoo_object_detection", library), StatusCode::OK);

    std::array<CustomNodeParam, 7> params = {{{"original_image_height", "4"},
        {"original_image_width", "4"},
        {"target_image_height", "2"},
        {"target_image_width", "2"},
        {"confidence_threshold", "0.5"},
        {"max_output_batch", "2"},
        {"buffer_queue_size", "2"}}};

    void* customNodeLibraryInternalManager = nullptr;
    ASSERT_EQ(library.initialize(&customNodeLibraryInternalManager, params.data(), params.size()), 0);

    std::vector<float> imageData(1 * 3 * 4 * 4, 1.0f);
    std::vector<uint64_t> imageDims{1, 3, 4, 4};
    CustomNodeTensor imageTensor{
        "image",
        reinterpret_cast<uint8_t*>(imageData.data()),
        static_cast<uint64_t>(imageData.size() * sizeof(float)),
        imageDims.data(),
        imageDims.size(),
        FP32};

    const uint64_t detectionsCount = 5;
    const uint64_t featuresCount = 7;
    std::vector<float> detectionData(detectionsCount * featuresCount, 0.0f);
    for (size_t i = 0; i < detectionsCount; ++i) {
        auto* detection = detectionData.data() + i * featuresCount;
        detection[0] = 0.0f;   // image_id
        detection[1] = 1.0f;   // label_id
        detection[2] = 0.99f;  // confidence
        detection[3] = 0.1f;
        detection[4] = 0.1f;
        detection[5] = 0.9f;
        detection[6] = 0.9f;
    }
    std::vector<uint64_t> detectionDims{1, 1, detectionsCount, featuresCount};
    CustomNodeTensor detectionTensor{
        "detection",
        reinterpret_cast<uint8_t*>(detectionData.data()),
        static_cast<uint64_t>(detectionData.size() * sizeof(float)),
        detectionDims.data(),
        detectionDims.size(),
        FP32};

    std::array<CustomNodeTensor, 2> inputs{imageTensor, detectionTensor};
    CustomNodeTensor* outputs = nullptr;
    int outputsCount = 0;

    ASSERT_EQ(library.execute(inputs.data(), inputs.size(), &outputs, &outputsCount, params.data(), params.size(), customNodeLibraryInternalManager), 0);
    ASSERT_NE(outputs, nullptr);
    ASSERT_EQ(outputsCount, 4);

    constexpr uint64_t maxOutputBatch = 2;
    for (int i = 0; i < outputsCount; ++i) {
        ASSERT_NE(outputs[i].dims, nullptr);
        ASSERT_GT(outputs[i].dimsCount, 0);
        EXPECT_EQ(outputs[i].dims[0], maxOutputBatch);
    }
    EXPECT_EQ(outputs[1].dataBytes, sizeof(int32_t) * 4 * maxOutputBatch);  // coordinates
    EXPECT_EQ(outputs[2].dataBytes, sizeof(float) * maxOutputBatch);        // confidences
    EXPECT_EQ(outputs[3].dataBytes, sizeof(int32_t) * maxOutputBatch);      // label_ids

    for (int i = 0; i < outputsCount; ++i) {
        library.release(outputs[i].data, customNodeLibraryInternalManager);
        library.release(outputs[i].dims, customNodeLibraryInternalManager);
    }
    library.release(outputs, customNodeLibraryInternalManager);
    EXPECT_EQ(library.deinitialize(customNodeLibraryInternalManager), 0);
}

class ModelManagerNodeLibraryTest : public TestWithTempDir {};

TEST_F(ModelManagerNodeLibraryTest, LoadCustomNodeLibrary) {
    std::string config = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";

    adjustConfigForTargetPlatform(config);
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";
    createConfigFileWithContent(config, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary library;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getCustomNodeLibraryManager().getLibrary("lib1", library);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(library.initialize, nullptr);
    ASSERT_NE(library.deinitialize, nullptr);
    ASSERT_NE(library.execute, nullptr);
    ASSERT_NE(library.getInputsInfo, nullptr);
    ASSERT_NE(library.getOutputsInfo, nullptr);
    ASSERT_NE(library.release, nullptr);
    EXPECT_EQ(library.initialize(nullptr, nullptr, 0), 0);
    EXPECT_EQ(library.deinitialize(nullptr), 0);
    EXPECT_EQ(library.execute(nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr), 1);
    EXPECT_EQ(library.getInputsInfo(nullptr, nullptr, nullptr, 0, nullptr), 2);
    EXPECT_EQ(library.getOutputsInfo(nullptr, nullptr, nullptr, 0, nullptr), 3);
    EXPECT_EQ(library.release(nullptr, nullptr), 4);
}

TEST_F(ModelManagerNodeLibraryTest, FailLoadingCorruptedCustomNodeLibrary) {
    std::string config = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_missing_implementation.so"}
        ]})";

    adjustConfigForTargetPlatform(config);
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";
    createConfigFileWithContent(config, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary library;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    status = manager.getCustomNodeLibraryManager().getLibrary("lib1", library);
    ASSERT_EQ(status, StatusCode::NODE_LIBRARY_MISSING);
    EXPECT_EQ(library.initialize, nullptr);
    EXPECT_EQ(library.deinitialize, nullptr);
    EXPECT_EQ(library.execute, nullptr);
    EXPECT_EQ(library.getInputsInfo, nullptr);
    EXPECT_EQ(library.getOutputsInfo, nullptr);
    EXPECT_EQ(library.release, nullptr);
}

TEST_F(ModelManagerNodeLibraryTest, AddAndRemoveLibrariesInConfigReload) {
    std::string configBefore = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    std::string configAfter = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"},
            {"name": "lib2", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";

    adjustConfigForTargetPlatform(configBefore);
    adjustConfigForTargetPlatform(configAfter);

    // Start with configBefore
    createConfigFileWithContent(configBefore, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary lib1Before, lib2Before;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Before), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2Before), StatusCode::NODE_LIBRARY_MISSING);

    // Expect lib1 to be loaded but lib2 not
    EXPECT_NE(lib1Before.initialize, nullptr);
    EXPECT_NE(lib1Before.deinitialize, nullptr);
    EXPECT_NE(lib1Before.execute, nullptr);
    EXPECT_NE(lib1Before.getInputsInfo, nullptr);
    EXPECT_NE(lib1Before.getOutputsInfo, nullptr);
    EXPECT_NE(lib1Before.release, nullptr);
    EXPECT_EQ(lib2Before.initialize, nullptr);
    EXPECT_EQ(lib2Before.deinitialize, nullptr);
    EXPECT_EQ(lib2Before.execute, nullptr);
    EXPECT_EQ(lib2Before.getInputsInfo, nullptr);
    EXPECT_EQ(lib2Before.getOutputsInfo, nullptr);
    EXPECT_EQ(lib2Before.release, nullptr);

    // Reload with configAfter
    NodeLibrary lib1After, lib2After;
    createConfigFileWithContent(configAfter, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1After), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2After), StatusCode::OK);

    // Expect lib1 not to change and lib2 to be created after reload.
    EXPECT_EQ(lib1Before.initialize, lib1After.initialize);
    EXPECT_EQ(lib1Before.deinitialize, lib1After.deinitialize);
    EXPECT_EQ(lib1Before.execute, lib1After.execute);
    EXPECT_EQ(lib1Before.getInputsInfo, lib1After.getInputsInfo);
    EXPECT_EQ(lib1Before.getOutputsInfo, lib1After.getOutputsInfo);
    EXPECT_EQ(lib1Before.release, lib1After.release);
    EXPECT_NE(lib2After.initialize, nullptr);
    EXPECT_NE(lib2After.deinitialize, nullptr);
    EXPECT_NE(lib2After.execute, nullptr);
    EXPECT_NE(lib2After.getInputsInfo, nullptr);
    EXPECT_NE(lib2After.getOutputsInfo, nullptr);
    EXPECT_NE(lib2After.release, nullptr);

    // Reload with initial config (remove lib2 entry)
    NodeLibrary lib1Entry, lib2Entry;
    createConfigFileWithContent(configBefore, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    // Expect lib1 not to change and lib2 to be removed
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Entry), StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib2", lib2Entry), StatusCode::NODE_LIBRARY_MISSING);

    EXPECT_EQ(lib1After.initialize, lib1Entry.initialize);
    EXPECT_EQ(lib1After.deinitialize, lib1Entry.deinitialize);
    EXPECT_EQ(lib1After.execute, lib1Entry.execute);
    EXPECT_EQ(lib1After.getInputsInfo, lib1Entry.getInputsInfo);
    EXPECT_EQ(lib1After.getOutputsInfo, lib1Entry.getOutputsInfo);
    EXPECT_EQ(lib1After.release, lib1Entry.release);
    EXPECT_EQ(lib2Entry.initialize, nullptr);
    EXPECT_EQ(lib2Entry.deinitialize, nullptr);
    EXPECT_EQ(lib2Entry.execute, nullptr);
    EXPECT_EQ(lib2Entry.getInputsInfo, nullptr);
    EXPECT_EQ(lib2Entry.getOutputsInfo, nullptr);
    EXPECT_EQ(lib2Entry.release, nullptr);
}

TEST_F(ModelManagerNodeLibraryTest, AddRemoveAndAddLibraryInConfigReload) {
    std::string configBefore = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_mock.so"}
        ]})";
    std::string configRemove = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
        ]})";
    std::string configAfter = R"({
        "model_config_list": [],
        "custom_node_library_config_list": [
            {"name": "lib1", "base_path": "/ovms/bazel-bin/src/lib_node_add_sub.so"}
        ]})";
    std::string fileToReload = directoryPath + "/ovms_config_file1.json";

    adjustConfigForTargetPlatform(configBefore);
    adjustConfigForTargetPlatform(configRemove);
    adjustConfigForTargetPlatform(configAfter);
    // Start with configBefore
    createConfigFileWithContent(configBefore, fileToReload);
    ConstructorEnabledModelManager manager;
    NodeLibrary lib1Before;
    auto status = manager.startFromFile(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Before), StatusCode::OK);

    // Expect lib1 to be loaded
    EXPECT_TRUE(lib1Before.isValid());

    // Reload with configRemove
    NodeLibrary lib1Remove;
    createConfigFileWithContent(configRemove, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1Remove), StatusCode::NODE_LIBRARY_MISSING);

    // Expect lib1 to be removed
    EXPECT_EQ(lib1Remove.initialize, nullptr);
    EXPECT_EQ(lib1Remove.deinitialize, nullptr);
    EXPECT_EQ(lib1Remove.execute, nullptr);
    EXPECT_EQ(lib1Remove.getInputsInfo, nullptr);
    EXPECT_EQ(lib1Remove.getOutputsInfo, nullptr);
    EXPECT_EQ(lib1Remove.release, nullptr);

    // Reload with configAfter
    NodeLibrary lib1After;
    createConfigFileWithContent(configAfter, fileToReload);
    status = manager.loadConfig(fileToReload);
    ASSERT_EQ(status, StatusCode::OK);

    // Expect lib1 to be added with different library
    ASSERT_EQ(manager.getCustomNodeLibraryManager().getLibrary("lib1", lib1After), StatusCode::OK);

    EXPECT_TRUE(lib1After.isValid());
    EXPECT_NE(lib1Before.initialize, lib1After.initialize);
    EXPECT_NE(lib1Before.deinitialize, lib1After.deinitialize);
    EXPECT_NE(lib1Before.execute, lib1After.execute);
    EXPECT_NE(lib1Before.getInputsInfo, lib1After.getInputsInfo);
    EXPECT_NE(lib1Before.getOutputsInfo, lib1After.getOutputsInfo);
    EXPECT_NE(lib1Before.release, lib1After.release);
}
