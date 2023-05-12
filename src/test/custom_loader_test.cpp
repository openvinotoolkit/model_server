//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include <stdlib.h>

#include "../executingstreamidguard.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../localfilesystem.hpp"
#include "../model.hpp"
#include "../model_service.hpp"
#include "../modelinstance.hpp"
#include "../modelinstanceunloadguard.hpp"
#include "../modelmanager.hpp"
#include "../modelversionstatus.hpp"
#include "../prediction_service_utils.hpp"
#include "../schema.hpp"
#include "../sequence_processing_spec.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"

using testing::_;
using testing::ContainerEq;
using testing::Each;
using testing::Eq;
using ::testing::NiceMock;
using testing::Return;
using testing::ReturnRef;
using testing::UnorderedElementsAre;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

using namespace ovms;

namespace {

// Custom Loader Config Keys
#define ENABLE_FORCE_BLACKLIST_CHECK "ENABLE_FORCE_BLACKLIST_CHECK"

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

// config_model_with_customloader
const char* custom_loader_config_model_relative_paths = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "test_cl_models/model1",
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

// config_model_with_customloader
const char* custom_loader_config_model_multiple = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader-a",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         },
         {
          "config":{
            "loader_name":"sample-loader-b",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         },
         {
          "config":{
            "loader_name":"sample-loader-c",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy-a",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader-a", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        },
        {
          "config":{
            "name":"dummy-b",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader-b", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        },
        {
          "config":{
            "name":"dummy-c",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader-c", "model_file":  "dummy.xml", "bin_file": "dummy.bin"}
          }
        }
      ]
    })";

const char* custom_loader_config_model_blacklist = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so",
            "loader_config_file": "sample-loader-config"
          }
         }
       ],
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1",
            "nireq": 1,
            "custom_loader_options": {"loader_name":  "sample-loader", "model_file":  "dummy.xml", "bin_file": "dummy.bin", "enable_file": "dummy.status"}
          }
        }
      ]
    })";

const char* empty_config = R"({
      "custom_loader_config_list":[],
      "model_config_list":[]
    })";

const char* expected_json_available = R"({
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
)";

const char* expected_json_end = R"({
 "model_version_status": [
  {
   "version": "1",
   "state": "END",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
)";

const char* expected_json_loading_error = R"({
 "model_version_status": [
  {
   "version": "1",
   "state": "LOADING",
   "status": {
    "error_code": "UNKNOWN",
    "error_message": "UNKNOWN"
   }
  }
 ]
}
)";

}  // namespace

class TestCustomLoader : public ::testing::Test {
public:
    void SetUp() {
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

        cl_models_path = "/tmp/" + std::string(test_info->name());
        cl_model_1_path = cl_models_path + "/model1/";
        cl_model_2_path = cl_models_path + "/model2/";

        const std::string FIRST_MODEL_NAME = "dummy";
        const std::string SECOND_MODEL_NAME = "dummy_new";

        std::filesystem::remove_all(cl_models_path);
        std::filesystem::create_directories(cl_model_1_path);
    }
    void TearDown() {
        // Create config file with an empty config & reload
        std::string configStr = empty_config;
        std::string fileToReload = cl_models_path + "/cl_config.json";
        createConfigFileWithContent(configStr, fileToReload);
        ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

        // Clean up temporary destination
        std::filesystem::remove_all(cl_models_path);
    }

    /**
     * @brief This function should mimic most closely predict request to check for thread safety
     */
    void performPredict(const std::string modelName,
        const ovms::model_version_t modelVersion,
        const tensorflow::serving::PredictRequest& request,
        std::unique_ptr<std::future<void>> waitBeforeGettingModelInstance = nullptr,
        std::unique_ptr<std::future<void>> waitBeforePerformInference = nullptr);

    void deserialize(const std::vector<float>& input, ov::InferRequest& inferRequest, std::shared_ptr<ovms::ModelInstance> modelInstance) {
        try {
            ov::Tensor tensor(
                modelInstance->getInputsInfo().at(DUMMY_MODEL_INPUT_NAME)->getOvPrecision(),
                modelInstance->getInputsInfo().at(DUMMY_MODEL_INPUT_NAME)->getShape().createPartialShape().get_shape(),
                const_cast<float*>(reinterpret_cast<const float*>(input.data())));
            inferRequest.set_tensor(DUMMY_MODEL_INPUT_NAME, tensor);
        } catch (...) {
            ASSERT_TRUE(false) << "exception during deserialize";
        }
    }

    void serializeAndCheck(int outputSize, ov::InferRequest& inferRequest) {
        std::vector<float> output(outputSize);
        ASSERT_THAT(output, Each(Eq(0.)));
        auto tensorOutput = inferRequest.get_tensor(DUMMY_MODEL_OUTPUT_NAME);
        ASSERT_EQ(tensorOutput.get_byte_size(), outputSize * sizeof(float));
        std::memcpy(output.data(), tensorOutput.data(), outputSize * sizeof(float));
        EXPECT_THAT(output, Each(Eq(2.)));
    }

    ovms::Status performInferenceWithRequest(const tensorflow::serving::PredictRequest& request, tensorflow::serving::PredictResponse& response) {
        std::shared_ptr<ovms::ModelInstance> model;
        std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
        auto status = manager.getModelInstance("dummy", 0, model, unload_guard);
        if (!status.ok()) {
            return status;
        }

        response.Clear();
        return model->infer(&request, &response, unload_guard);
    }

public:
    ConstructorEnabledModelManager manager;

    ~TestCustomLoader() {
        std::cout << "Destructor of TestCustomLoader()" << std::endl;
    }

    std::string cl_models_path;
    std::string cl_model_1_path;
    std::string cl_model_2_path;
};

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", 42, ieCore) {}
    const ovms::Status mockValidate(const tensorflow::serving::PredictRequest* request) {
        return validate(request);
    }
};

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
    ASSERT_EQ(manager.getModelInstance(modelName, modelVersion, modelInstance, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    if (waitBeforePerformInference) {
        std::cout << "Waiting before performInfernce." << std::endl;
        waitBeforePerformInference->get();
    }
    ovms::Status validationStatus = (std::static_pointer_cast<MockModelInstance>(modelInstance))->mockValidate(&request);
    std::cout << validationStatus.string() << std::endl;
    ASSERT_TRUE(validationStatus == ovms::StatusCode::OK ||
                validationStatus == ovms::StatusCode::RESHAPE_REQUIRED ||
                validationStatus == ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
    auto bsPositionIndex = 0;
    auto requestBatchSize = ovms::getRequestBatchSize(&request, bsPositionIndex);
    auto requestShapes = ovms::getRequestShapes(&request);
    ASSERT_EQ(modelInstance->reloadModelIfRequired(validationStatus, requestBatchSize, requestShapes, modelInstanceUnloadGuard), ovms::StatusCode::OK);

    ovms::ExecutingStreamIdGuard executingStreamIdGuard(modelInstance->getInferRequestsQueue(), modelInstance->getMetricReporter());
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    std::vector<float> input(inputSize);
    std::generate(input.begin(), input.end(), []() { return 1.; });
    ASSERT_THAT(input, Each(Eq(1.)));
    deserialize(input, inferRequest, modelInstance);
    auto status = modelInstance->performInference(inferRequest);
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMatchingSchemaParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLibraryPathParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigInvalidCustomLoaderConfigParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMissingLoaderNameInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
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
    auto result = ovms::validateJsonAgainstSchema(customloaderConfigMultiplePropertiesInCustomLoaderOptionsParsed, ovms::MODELS_CONFIG_SCHEMA.c_str());
    EXPECT_EQ(result, ovms::StatusCode::OK);
}

// Functional Validation

TEST_F(TestCustomLoader, CustomLoaderPrediction) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderPredictionRelativePath) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);
    std::filesystem::copy("/ovms/bazel-bin/src/libsampleloader.so", cl_models_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_relative_paths;
    configStr.replace(configStr.find("test_cl_models"), std::string("test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderGetStatus) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const = res;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);
}

TEST_F(TestCustomLoader, CustomLoaderPredictDeletePredict) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_EQ(performInferenceWithRequest(request, response), ovms::StatusCode::OK);

    // Re-create config file
    createConfigFileWithContent(custom_loader_config_model_deleted, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    ASSERT_EQ(performInferenceWithRequest(request, response), ovms::StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(TestCustomLoader, CustomLoaderPredictNewVersionPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);

    // Copy version 1 to version 2
    std::filesystem::create_directories(cl_model_1_path + "2");
    std::filesystem::copy(cl_model_1_path + "1", cl_model_1_path + "2", std::filesystem::copy_options::recursive);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 2, request);
}

TEST_F(TestCustomLoader, CustomLoaderPredictNewModelPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);

    // Copy model1 to model2
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_2_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    configStr = custom_loader_config_model_new;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Re-create config file
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);
    performPredict("dummy-new", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderPredictRemoveCustomLoaderOptionsPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);

    // Replace model path in the config string
    configStr = custom_loader_config_model_customloader_options_removed;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Re-create config file
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, PredictNormalModelAddCustomLoaderOptionsPredict) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_customloader_options_removed;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);

    // Replace model path in the config string
    configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderOptionWithUnknownLibrary) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = config_model_with_customloader_options_unknown_loadername;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_EQ(performInferenceWithRequest(request, response), ovms::StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(TestCustomLoader, CustomLoaderWithMissingModelFiles) {
    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    tensorflow::serving::PredictResponse response;
    ASSERT_EQ(performInferenceWithRequest(request, response), ovms::StatusCode::MODEL_VERSION_MISSING);
}

TEST_F(TestCustomLoader, CustomLoaderGetStatusDeleteModelGetStatus) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const = res;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);

    // Re-create config file
    createConfigFileWithContent(custom_loader_config_model_deleted, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest reqx;
    tensorflow::serving::GetModelStatusResponse resx;

    auto model_specx = reqx.mutable_model_spec();
    model_specx->Clear();
    model_specx->set_name("dummy");
    model_specx->mutable_version()->set_value(1);

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&reqx, &resx, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_constx = resx;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_constx, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_end);
}

TEST_F(TestCustomLoader, CustomLoaderPredictionUsingManyCustomLoaders) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_multiple;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});

    performPredict("dummy-a", 1, request);
    performPredict("dummy-b", 1, request);
    performPredict("dummy-c", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderGetMetaData) {
    const char* expected_json = R"({
 "modelSpec": {
  "name": "dummy",
  "signatureName": "",
  "version": "1"
 },
 "metadata": {
  "signature_def": {
   "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
   "signatureDef": {
    "serving_default": {
     "inputs": {
      "b": {
       "dtype": "DT_FLOAT",
       "tensorShape": {
        "dim": [
         {
          "size": "1",
          "name": ""
         },
         {
          "size": "10",
          "name": ""
         }
        ],
        "unknownRank": false
       },
       "name": "b"
      }
     },
     "outputs": {
      "a": {
       "dtype": "DT_FLOAT",
       "tensorShape": {
        "dim": [
         {
          "size": "1",
          "name": ""
         },
         {
          "size": "10",
          "name": ""
         }
        ],
        "unknownRank": false
       },
       "name": "a"
      }
     },
     "methodName": ""
    }
   }
  }
 }
}
)";

    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    std::shared_ptr<ovms::ModelInstance> model;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> unload_guard;
    ASSERT_EQ(manager.getModelInstance("dummy", 1, model, unload_guard), ovms::StatusCode::OK);

    tensorflow::serving::GetModelMetadataResponse response;
    ovms::GetModelMetadataImpl::buildResponse(model, &response);

    std::string json_output = "";
    ovms::GetModelMetadataImpl::serializeResponse2Json(&response, &json_output);

    EXPECT_TRUE(response.has_model_spec());
    EXPECT_EQ(response.model_spec().name(), "dummy");

    tensorflow::serving::SignatureDefMap def;
    response.metadata().at("signature_def").UnpackTo(&def);

    const auto& inputs = ((*def.mutable_signature_def())["serving_default"]).inputs();
    const auto& outputs = ((*def.mutable_signature_def())["serving_default"]).outputs();

    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(json_output, expected_json);
}

TEST_F(TestCustomLoader, CustomLoaderMultipleLoaderWithSameLoaderName) {
    const char* custom_loader_config_model_xx = R"({
       "custom_loader_config_list":[
         {
          "config":{
            "loader_name":"sample-loader",
            "library_path": "/ovms/bazel-bin/src/libsampleloader.so"
          }
         },
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

    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_xx;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::PredictRequest request;
    preparePredictRequest(request,
        {{DUMMY_MODEL_INPUT_NAME,
            std::tuple<ovms::signed_shape_t, ovms::Precision>{{1, 10}, ovms::Precision::FP32}}});
    performPredict("dummy", 1, request);
}

TEST_F(TestCustomLoader, CustomLoaderBlackListingModel) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Create Sample Custom Loader Config
    std::string cl_config_file_path = cl_models_path;
    std::string cl_config_str = ENABLE_FORCE_BLACKLIST_CHECK;
    std::string cl_config_file = cl_config_file_path + "/customloader_config";
    createConfigFileWithContent(cl_config_str, cl_config_file);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_blacklist;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("sample-loader-config"), std::string("sample-loader-config").size(), cl_config_file);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    tensorflow::serving::GetModelStatusResponse response_const = res;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);

    // copy status file
    std::string status_file_path = cl_model_1_path + "1";
    std::string status_str = "DISABLED";
    std::string status_file = status_file_path + "/dummy.status";
    createConfigFileWithContent(status_str, status_file);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest reqx;
    tensorflow::serving::GetModelStatusResponse resx;

    auto model_specx = reqx.mutable_model_spec();
    model_specx->Clear();
    model_specx->set_name("dummy");
    model_specx->mutable_version()->set_value(1);

    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&reqx, &resx, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_constx = resx;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_constx, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_end);
}

TEST_F(TestCustomLoader, CustomLoaderBlackListingRevoke) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Create Sample Custom Loader Config
    std::string cl_config_file_path = cl_models_path;
    std::string cl_config_str = ENABLE_FORCE_BLACKLIST_CHECK;
    std::string cl_config_file = cl_config_file_path + "/customloader_config";
    createConfigFileWithContent(cl_config_str, cl_config_file);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_blacklist;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("sample-loader-config"), std::string("sample-loader-config").size(), cl_config_file);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const = res;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);

    // copy status file
    std::string status_file_path = cl_model_1_path + "1";
    std::string status_str = "DISABLED";
    std::string status_file = status_file_path + "/dummy.status";
    createConfigFileWithContent(status_str, status_file);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req1;
    tensorflow::serving::GetModelStatusResponse res1;

    auto model_spec1 = req1.mutable_model_spec();
    model_spec1->Clear();
    model_spec1->set_name("dummy");
    model_spec1->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req1, &res1, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const1 = res1;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_const1, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_end);

    // Remove status file
    std::filesystem::remove(status_file);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req2;
    tensorflow::serving::GetModelStatusResponse res2;

    auto model_spec2 = req2.mutable_model_spec();
    model_spec2->Clear();
    model_spec2->set_name("dummy");
    model_spec2->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req2, &res2, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const2 = res2;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_const2, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);
}

TEST_F(TestCustomLoader, CustomLoaderBlackListModelReloadError) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Create Sample Custom Loader Config
    std::string cl_config_file_path = cl_models_path;
    std::string cl_config_str = ENABLE_FORCE_BLACKLIST_CHECK;
    std::string cl_config_file = cl_config_file_path + "/customloader_config";
    createConfigFileWithContent(cl_config_str, cl_config_file);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_blacklist;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("sample-loader-config"), std::string("sample-loader-config").size(), cl_config_file);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req;
    tensorflow::serving::GetModelStatusResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const = res;
    std::string json_output;
    Status error_status = GetModelStatusImpl::serializeResponse2Json(&response_const, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);

    // copy status file
    std::string status_file_path = cl_model_1_path + "1";
    std::string status_str = "DISABLED";
    std::string status_file = status_file_path + "/dummy.status";
    createConfigFileWithContent(status_str, status_file);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req1;
    tensorflow::serving::GetModelStatusResponse res1;

    auto model_spec1 = req1.mutable_model_spec();
    model_spec1->Clear();
    model_spec1->set_name("dummy");
    model_spec1->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req1, &res1, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const1 = res1;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_const1, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_end);

    // Remove status file & the Dummy.bin file
    std::filesystem::remove(status_file);
    std::string bin_file = status_file_path + "/dummy.bin";
    std::filesystem::remove(bin_file);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::FILE_INVALID);

    tensorflow::serving::GetModelStatusRequest req2;
    tensorflow::serving::GetModelStatusResponse res2;

    auto model_spec2 = req2.mutable_model_spec();
    model_spec2->Clear();
    model_spec2->set_name("dummy");
    model_spec2->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req2, &res2, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const2 = res2;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_const2, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_loading_error);

    // Copy back the model files & try reload
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req3;
    tensorflow::serving::GetModelStatusResponse res3;

    auto model_spec3 = req3.mutable_model_spec();
    model_spec3->Clear();
    model_spec3->set_name("dummy");
    model_spec3->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req3, &res3, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const3 = res3;
    json_output = "";
    error_status = GetModelStatusImpl::serializeResponse2Json(&response_const3, &json_output);
    ASSERT_EQ(error_status, StatusCode::OK);
    EXPECT_EQ(json_output, expected_json_available);
}

TEST_F(TestCustomLoader, CustomLoaderLoadBlackListedModel) {
    // Copy dummy model to temporary destination
    std::filesystem::copy("/ovms/src/test/dummy", cl_model_1_path, std::filesystem::copy_options::recursive);

    // Create Sample Custom Loader Config
    std::string cl_config_file_path = cl_models_path;
    std::string cl_config_str = ENABLE_FORCE_BLACKLIST_CHECK;
    std::string cl_config_file = cl_config_file_path + "/customloader_config";
    createConfigFileWithContent(cl_config_str, cl_config_file);

    // Replace model path in the config string
    std::string configStr = custom_loader_config_model_blacklist;
    configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    configStr.replace(configStr.find("sample-loader-config"), std::string("sample-loader-config").size(), cl_config_file);

    // Create config file
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);

    // Create status file
    std::string status_file_path = cl_model_1_path + "1";
    std::string status_str = "DISABLED";
    std::string status_file = status_file_path + "/dummy.status";
    createConfigFileWithContent(status_str, status_file);
    ovms::Status status1 = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status1 == ovms::StatusCode::INTERNAL_ERROR);

    tensorflow::serving::GetModelStatusRequest req1;
    tensorflow::serving::GetModelStatusResponse res1;

    auto model_spec1 = req1.mutable_model_spec();
    model_spec1->Clear();
    model_spec1->set_name("dummy");
    model_spec1->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req1, &res1, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const1 = res1;
    std::string json_output1;
    Status error_status1 = GetModelStatusImpl::serializeResponse2Json(&response_const1, &json_output1);
    ASSERT_EQ(error_status1, StatusCode::OK);
    EXPECT_EQ(json_output1, expected_json_loading_error);

    // remove enable_file from config file
    std::string status_config = ", \"enable_file\": \"dummy.status\"";
    configStr.replace(configStr.find(status_config), std::string(status_config).size(), "");
    createConfigFileWithContent(configStr, fileToReload);

    ovms::Status status2 = manager.loadConfig(fileToReload);
    ASSERT_TRUE(status2 == ovms::StatusCode::OK);

    tensorflow::serving::GetModelStatusRequest req2;
    tensorflow::serving::GetModelStatusResponse res2;

    auto model_spec2 = req2.mutable_model_spec();
    model_spec2->Clear();
    model_spec2->set_name("dummy");
    model_spec2->mutable_version()->set_value(1);
    ASSERT_EQ(GetModelStatusImpl::getModelStatus(&req2, &res2, manager, DEFAULT_TEST_CONTEXT), StatusCode::OK);

    const tensorflow::serving::GetModelStatusResponse response_const2 = res2;
    std::string json_output2;
    Status error_status2 = GetModelStatusImpl::serializeResponse2Json(&response_const2, &json_output2);
    ASSERT_EQ(error_status2, StatusCode::OK);
    EXPECT_EQ(json_output2, expected_json_available);
}

#pragma GCC diagnostic pop
