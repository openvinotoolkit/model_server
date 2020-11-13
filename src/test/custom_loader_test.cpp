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
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <stdlib.h>

#include "../executinstreamidguard.hpp"
#include "../localfilesystem.hpp"
#include "../model.hpp"
#include "../model_service.hpp"
#include "../modelinstance.hpp"
#include "../modelmanager.hpp"
#include "../modelversionstatus.hpp"
#include "../prediction_service_utils.hpp"
#include "../schema.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using testing::_;
using testing::ContainerEq;
using testing::Each;
using testing::Eq;
using testing::Return;
using testing::ReturnRef;
using testing::UnorderedElementsAre;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

using namespace ovms;

namespace {

// config_model_with_customloader
const char* custom_loader_config_model = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

// config_no_model_with_customloader
const char* custom_loader_config_model_deleted = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[]
    })";

// config_2_models_with_customloader
const char* custom_loader_config_model_new = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        },
        {
          "config":{
            "name":"dummy-new",
            "base_path": "/tmp/test_cl_models/model2",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

// config_model_without_customloader_options
const char* custom_loader_config_model_customloader_options_removed = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1
          }
        }
      ]
    })";

const char* config_model_with_customloader_options_unknown_loadername = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "unknown", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

const std::string FIRST_MODEL_NAME = "dummy";
const std::string SECOND_MODEL_NAME = "dummy_new";

const std::string cl_models_path = "/tmp/test_cl_models/";
const std::string cl_model_1_path = cl_models_path + "model1/";
const std::string cl_model_2_path = cl_models_path + "model2/";

class MockModel : public ovms::Model {
public:
    MockModel() :
        Model("MOCK_NAME") {}
    MOCK_METHOD(ovms::Status, addVersion, (const ovms::ModelConfig&), (override));
};

std::shared_ptr<MockModel> modelMock;

}  // namespace

class MockModelManager : public ovms::ModelManager {
public:
    std::shared_ptr<ovms::Model> modelFactory(const std::string& name) override {
        return modelMock;
    }
};

class TestCustomLoader : public ::testing::Test {
public:
    void SetUp() {
        ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
        const int initialBatchSize = 1;
        config.setBatchSize(initialBatchSize);
        config.setNireq(2);
    }
    void TearDown() {
        SPDLOG_ERROR("TEAR_DOWN");
    }
    /**
     * @brief This function should mimic most closely predict request to check for thread safety
     */
    void performPredict(const std::string modelName,
        const ovms::model_version_t modelVersion,
        const tensorflow::serving::PredictRequest& request,
        std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance = nullptr,
        std::unique_ptr<std::future<void>> waitBeforePerformInference = nullptr);

    void deserialize(const std::vector<float>& input, InferenceEngine::InferRequest& inferRequest, std::shared_ptr<ovms::ModelInstance> modelInstance) {
        auto blob = InferenceEngine::make_shared_blob<float>(
            modelInstance->getInputsInfo().at(DUMMY_MODEL_INPUT_NAME)->getTensorDesc(),
            const_cast<float*>(reinterpret_cast<const float*>(input.data())));
        inferRequest.SetBlob(DUMMY_MODEL_INPUT_NAME, blob);
    }

    void serializeAndCheck(int outputSize, InferenceEngine::InferRequest& inferRequest) {
        std::vector<float> output(outputSize);
        ASSERT_THAT(output, Each(Eq(0.)));
        auto blobOutput = inferRequest.GetBlob(DUMMY_MODEL_OUTPUT_NAME);
        ASSERT_EQ(blobOutput->byteSize(), outputSize * sizeof(float));
        std::memcpy(output.data(), blobOutput->cbuffer(), outputSize * sizeof(float));
        EXPECT_THAT(output, Each(Eq(2.)));
    }

    ovms::Status performInferenceWithRequest(const tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response) {
        std::shared_ptr<ovms::ModelInstance> model;
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
        auto status = ovms::getModelInstance(manager, "dummy", 0, model, unload_guard);
        if (!status.ok()) {
            return status;
        }

        response.Clear();
        return ovms::inference(*model, &request, &response, unload_guard);
    }

public:
    ConstructorEnabledModelManager manager;
    ovms::ModelConfig config = DUMMY_MODEL_CONFIG;
    ~TestCustomLoader() {
        std::cout << "Destructor of TestCustomLoader()" << std::endl;
    }
};

::grpc::Status test_PerformModelStatusRequestX(ModelServiceImpl& s, tensorflow::serving::GetModelStatusRequest& req, tensorflow::serving::GetModelStatusResponse& res) {
    spdlog::info("req={} res={}", req.DebugString(), res.DebugString());
    ::grpc::Status ret = s.GetModelStatus(nullptr, &req, &res);
    spdlog::info("returned grpc status: ok={} code={} msg='{}'", ret.ok(), ret.error_code(), ret.error_details());
    return ret;
}

void TestCustomLoader::performPredict(const std::string modelName,
    const ovms::model_version_t modelVersion,
    const tensorflow::serving::PredictRequest& request,
    std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance,
    std::unique_ptr<std::future<void>> waitBeforePerformInference) {
    // only validation is skipped
    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;

    auto& tensorProto = request.inputs().find("b")->second;
    size_t batchSize = tensorProto.tensor_shape().dim(0).size();
    size_t inputSize = 1;
    for (int i = 0; i < tensorProto.tensor_shape().dim_size(); i++) {
        inputSize *= tensorProto.tensor_shape().dim(i).size();
    }

    if (waitBeforeGettingModelInstance) {
        std::cout << "Waiting before getModelInstance. Batch size: " << batchSize << std::endl;
        waitBeforeGettingModelInstance->get();
    }
    ASSERT_EQ(getModelInstance(manager, modelName, modelVersion, modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    if (waitBeforePerformInference) {
        std::cout << "Waiting before performInfernce." << std::endl;
        waitBeforePerformInference->get();
    }
    ovms::Status validationStatus = modelInstance->validate(&request);
    std::cout << validationStatus.string() << std::endl;
    ASSERT_TRUE(validationStatus == ovms::StatusCode::OK ||
                validationStatus == ovms::StatusCode::RESHAPE_REQUIRED ||
                validationStatus == ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
    ASSERT_EQ(reloadModelIfRequired(validationStatus, *modelInstance, &request, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    ovms::OVInferRequestsQueue& inferRequestsQueue = modelInstance->getInferRequestsQueue();
    ovms::ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    std::vector<float> input(inputSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));
    deserialize(input, inferRequest, modelInstance);
    auto status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    ASSERT_EQ(status, ovms::StatusCode::OK);
    size_t outputSize = batchSize * DUMMY_MODEL_OUTPUT_SIZE;
    serializeAndCheck(outputSize, inferRequest);
}

// Schema Validation

TEST_F(TestCustomLoader, CustomLoaderConfigMatchingSchema) {
    const char* customloaderConfigMatchingSchema = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "library_path": "/tmp/loader/dummyloader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"loader_name":  "dummy-loader"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMatchingSchemaParsed;
    customloaderConfigMatchingSchemaParsed.Parse(customloaderConfigMatchingSchema);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderName) {
    const char* customloaderConfigMissingLoaderName = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "library_path": "dummyloader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderNameParsed;
    customloaderConfigMissingLoaderNameParsed.Parse(customloaderConfigMissingLoaderName);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLibraryPath) {
    const char* customloaderConfigMissingLibraryPath = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "loader_config_file": "dummyloader-config"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLibraryPathParsed;
    customloaderConfigMissingLibraryPathParsed.Parse(customloaderConfigMissingLibraryPath);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLibraryPathParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderConfig) {
    const char* customloaderConfigMissingLoaderConfig = R"(
        {
           "custom_loader_config_list":[
             {
              "config":{
                "loader_name":"dummy-loader",
                "library_path": "dummyloader"
              }
             }
           ],
           "model_config_list": []
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderConfigParsed;
    customloaderConfigMissingLoaderConfigParsed.Parse(customloaderConfigMissingLoaderConfig);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderConfigInvalidCustomLoaderConfig) {
    const char* customloaderConfigInvalidCustomLoaderConfig = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options_invalid": {"loader_name":  "dummy-loader"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigInvalidCustomLoaderConfigParsed;
    customloaderConfigInvalidCustomLoaderConfigParsed.Parse(customloaderConfigInvalidCustomLoaderConfig);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigInvalidCustomLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMissingLoaderNameInCustomLoaderOptions) {
    const char* customloaderConfigMissingLoaderNameInCustomLoaderOptions = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"a": "SS"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed;
    customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed.Parse(customloaderConfigMissingLoaderNameInCustomLoaderOptions);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::JSON_INVALID);
}

TEST_F(TestCustomLoader, CustomLoaderConfigMultiplePropertiesInCustomLoaderOptions) {
    const char* customloaderConfigMultiplePropertiesInCustomLoaderOptions = R"(
        {
          "model_config_list":[
            {
              "config":{
                "name":"dummy-loader-model",
                "base_path": "/tmp/models/dummy1",
                "custom_loader_options": {"loader_name": "dummy-loader", "1": "a", "2": "b", "3": "c", "4":"d", "5":"e", "6":"f"}
              }
            }
          ]
        }
    )";

    rapidjson::Document customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed;
    customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed.Parse(customloaderConfigMultiplePropertiesInCustomLoaderOptions);
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA);
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderPrediction) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderPredictDeletePredict) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_EQ(performInferenceWithRequest(request, response), ovms::StatusCode::OK);

    createConfigFileWithContent(custom_loader_config_model_deleted, fileToReload);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    ASSERT_TRUE(performInferenceWithRequest(request, response) != ovms::StatusCode::OK);

    /*try {
        std::filesystem::remove(path);
    } catch (const std::filesystem::filesystem_error&) {
    }*/
}

TEST_F(TestCustomLoader, CustomLoaderPredictNewVersionPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);

    // Copy version 1 to version 2
    std::filesystem::create_directories(cl_model_1_path + "2");
    std::filesystem::copy(cl_model_1_path + "1", cl_model_1_path + "2", std::filesystem::copy_options::recursive);

    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 2, request);
}

TEST_F(TestCustomLoader, CustomLoaderPredictNewModelPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);

    // Copy model1 to model2
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_2_path, std::filesystem::copy_options::recursive);

    createConfigFileWithContent(custom_loader_config_model_new, fileToReload);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);
    performPredict("dummy-new", 1, request);

    auto models = manager.getModels();
}

TEST_F(TestCustomLoader, CustomLoaderPredictRemoveCustomLoaderOptionsPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);

    createConfigFileWithContent(custom_loader_config_model_customloader_options_removed, fileToReload);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, PredictNormalModelAddCustomLoaderOptionsPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model_customloader_options_removed);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    performPredict("dummy", 1, request);

    createConfigFileWithContent(custom_loader_config_model, fileToReload);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderOptionWithUnknownLibrary) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(config_model_with_customloader_options_unknown_loadername);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_TRUE(performInferenceWithRequest(request, response) != ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderWithMissingModelFiles) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    // std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);
    ASSERT_EQ(manager.startFromFile(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request = preparePredictRequest(
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::shape_t, tensorflow::DataType>{{1, 10}, tensorflow::DataType::DT_FLOAT}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_TRUE(performInferenceWithRequest(request, response) != ovms::StatusCode::OK);
}

TEST_F(TestCustomLoader, CustomLoaderSingleVersion) {
    const char* custom_loader_config_model = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/tmp/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/ovms/src/test/dummy",
            "nireq": 1,
            "custom_loader_options": {"loader_name": "sample-loader", "xml_file": "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

    std::string fileToReload = "/tmp/ovms_config_cl.json";
    createConfigFileWithContent(custom_loader_config_model, fileToReload);

    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    manager.startFromFile(fileToReload);

    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");

    ::grpc::Status ret = test_PerformModelStatusRequestX(s, req, res);
    EXPECT_EQ(ret.ok(), true);
}

TEST_F(TestCustomLoader, CustomLoaderGetStatus) {
    const char* custom_loader_config_model = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/tmp/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/ovms/src/test/dummy",
            "nireq": 1,
            "custom_loader_options": {"loader_name": "sample-loader", "xml_file": "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

    // std::string fileToReload = "/tmp/ovms_config_cl.json";
    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);

    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    manager.startFromFile(fileToReload);

    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");

    ::grpc::Status ret = test_PerformModelStatusRequestX(s, req, res);
    EXPECT_EQ(ret.ok(), true);
}

TEST_F(TestCustomLoader, CustomLoaderGetStatusDeleteModelGetStatus) {
    const char* custom_loader_config_model = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/tmp/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/ovms/src/test/dummy",
            "nireq": 1,
            "custom_loader_options": {"loader_name": "sample-loader", "xml_file": "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

    std::string fileToReload = "/tmp/ovms_config_cl.json";
    createConfigFileWithContent(custom_loader_config_model, fileToReload);

    ovms::ModelManager& manager = ovms::ModelManager::getInstance();
    manager.startFromFile(fileToReload);

    ModelServiceImpl s;
    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");

    ::grpc::Status ret = test_PerformModelStatusRequestX(s, req, res);
    EXPECT_EQ(ret.ok(), true);
}

/*
TEST_F(TestCustomLoader, CustomLoaderStartFromFile) {
    // Copy dummy model to temporary destination
    std::filesystem::remove_all(cl_models_path);
    std::filesystem::create_directories(cl_model_1_path);
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    std::string fileToReload = createConfigFileWithContent(custom_loader_config_model);

    modelMock = std::make_shared<MockModel>();
    MockModelManager manager;

    EXPECT_CALL(*modelMock, addVersion(_))
        .Times(1)
        .WillRepeatedly(Return(ovms::Status(ovms::StatusCode::OK)));
    auto status = manager.startFromFile(fileToReload);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    manager.join();
    /modelMock.reset();
}*/

#pragma GCC diagnostic pop
