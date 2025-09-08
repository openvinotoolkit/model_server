//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include <chrono>
#include <future>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>

#if (MEDIAPIPE_DISABLE == 0)
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipegraphexecutor.hpp"
#endif
#include "../executingstreamidguard.hpp"
#include "../get_model_metadata_impl.hpp"
#include "../grpcservermodule.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
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
#include "../server.hpp"
#include "mockmodelinstancechangingstates.hpp"
#include "test_utils.hpp"
#include "light_test_utils.hpp"
#include "platform_utils.hpp"

using namespace ovms;
using namespace rapidjson;

class GetMediapipeGraphMetadataResponse : public ::testing::Test {
protected:
    KFSModelMetadataResponse response;
    ConstructorEnabledModelManager manager;
};

TEST_F(GetMediapipeGraphMetadataResponse, BasicResponseMetadata) {
    std::string testPbtxt = R"(
        input_stream: "TEST:in"
        input_stream: "TEST33:in2"
        output_stream: "TEST0:out"
        output_stream: "TEST1:out2"
        output_stream: "TEST3:out3"
            node {
            calculator: "OVMSOVCalculator"
            input_stream: "B:in"
            output_stream: "A:out"
              node_options: {
                  [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
                    servable_name: "dummyUpper"
                    servable_version: "1"
                  }
              }
            }
        )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeGraphDefinition("mediaDummy", mgc, testPbtxt);
    mediapipeGraphDefinition.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeGraphDefinition.validate(manager), StatusCode::OK);

    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(mediapipeGraphDefinition, &response), ovms::StatusCode::OK);
    EXPECT_EQ(response.name(), "mediaDummy");

    EXPECT_EQ(response.versions_size(), 1);
    EXPECT_EQ(response.versions().at(0), "1");

    EXPECT_EQ(response.platform(), "OpenVINO");

    EXPECT_EQ(response.inputs_size(), 2);
    auto firstInput = response.inputs().at(0);
    EXPECT_EQ(firstInput.name(), "in");
    EXPECT_EQ(firstInput.datatype(), "INVALID");
    EXPECT_EQ(firstInput.shape_size(), 0);
    auto secondInput = response.inputs().at(1);
    EXPECT_EQ(secondInput.name(), "in2");
    EXPECT_EQ(secondInput.datatype(), "INVALID");
    EXPECT_EQ(secondInput.shape_size(), 0);

    EXPECT_EQ(response.outputs_size(), 3);
    auto firstOutput = response.outputs().at(0);
    EXPECT_EQ(firstOutput.name(), "out");
    EXPECT_EQ(firstOutput.datatype(), "INVALID");
    EXPECT_EQ(firstOutput.shape_size(), 0);
    auto secondOutput = response.outputs().at(1);
    EXPECT_EQ(secondOutput.name(), "out2");
    EXPECT_EQ(secondOutput.datatype(), "INVALID");
    EXPECT_EQ(secondOutput.shape_size(), 0);

    auto thirdOutput = response.outputs().at(2);
    EXPECT_EQ(thirdOutput.name(), "out3");
    EXPECT_EQ(thirdOutput.datatype(), "INVALID");
    EXPECT_EQ(thirdOutput.shape_size(), 0);
}

class MediapipeGraphDefinitionMetadataResponseBuild : public ::testing::Test {
protected:
    class MockMediapipeGraphDefinitionGetInputsOutputsInfo : public ovms::MediapipeGraphDefinition {
        ovms::Status status = ovms::StatusCode::OK;

    public:
        MockMediapipeGraphDefinitionGetInputsOutputsInfo() :
            MediapipeGraphDefinition("mediaDummy", {}, {}) {
            MediapipeGraphDefinition::status.handle(ovms::ValidationPassedEvent());
        }

        void mockStatus(ovms::Status status) {
            this->status = status;
        }

        ovms::PipelineDefinitionStatus& getGraphDefinitionStatus() {
            return ovms::MediapipeGraphDefinition::status;
        }
    };

    MockMediapipeGraphDefinitionGetInputsOutputsInfo graphDefinition;
    KFSModelMetadataResponse response;
    ConstructorEnabledModelManager manager;
};

TEST_F(MediapipeGraphDefinitionMetadataResponseBuild, GraphNotLoadedAnymore) {
    graphDefinition.getGraphDefinitionStatus().handle(ovms::RetireEvent());
    auto status = ovms::KFSInferenceServiceImpl::buildResponse(graphDefinition, &response);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE) << status.string();
}

TEST_F(MediapipeGraphDefinitionMetadataResponseBuild, GraphNotLoadedYet) {
    graphDefinition.getGraphDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    graphDefinition.getGraphDefinitionStatus().handle(ovms::ValidationFailedEvent());
    auto status = ovms::KFSInferenceServiceImpl::buildResponse(graphDefinition, &response);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET) << status.string();
    graphDefinition.getGraphDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    ASSERT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(graphDefinition, &response), ovms::StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET);
}

TEST_F(MediapipeGraphDefinitionMetadataResponseBuild, GraphAvailableOrAvailableRequiringRevalidation) {
    graphDefinition.getGraphDefinitionStatus().handle(ovms::UsedModelChangedEvent());
    EXPECT_EQ(ovms::KFSInferenceServiceImpl::buildResponse(graphDefinition, &response), ovms::StatusCode::OK);
}

const char* dummy_config = R"({
      "model_config_list":[
        {
          "config":{
            "name":"dummy",
            "base_path": "/tmp/test_cl_models/model1"
          }
        }
      ]
    })";

class TestImplGetModelStatus : public ::testing::Test {
public:
    ConstructorEnabledModelManager manager;
    std::string cl_models_path;
    std::string cl_model_1_path;
    std::string cl_model_2_path;

    void SetUpSingleModel(std::string modelPath, std::string modelName) {
        std::string port{"9000"};
        randomizeAndEnsureFree(port);
        char* n_argv[] = {(char*)"ovms", (char*)"--model_path", (char*)modelPath.data(), (char*)"--model_name", (char*)modelName.data(), (char*)"--file_system_poll_wait_seconds", (char*)"0", (char*)"--port", (char*)port.c_str()};
        int arg_count = 9;
        ovms::Config::instance().parse(arg_count, n_argv);
    }

    void SetUp() {
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

        cl_models_path = getGenericFullPathForTmp("/tmp/" + std::string(test_info->name()));
        cl_model_1_path = cl_models_path + "/model1/";
        cl_model_2_path = cl_models_path + "/model2/";

        const std::string FIRST_MODEL_NAME = "dummy";
        const std::string SECOND_MODEL_NAME = "dummy_new";

        std::filesystem::remove_all(cl_models_path);
        std::filesystem::create_directories(cl_model_1_path);
    }

    void TearDown() {
        // Clean up temporary destination
        std::filesystem::remove_all(cl_models_path);
    }
};

TEST_F(TestImplGetModelStatus, NegativeTfsGetModelStatus) {
    // Create config file with an empty config & reload
    std::string configStr = dummy_config;
    configStr = configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);
    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    tensorflow::serving::GetModelMetadataRequest req;
    tensorflow::serving::GetModelMetadataResponse res;

    auto model_spec = req.mutable_model_spec();
    model_spec->Clear();
    model_spec->set_name("dummy2");
    model_spec->mutable_version()->set_value(2);
    ASSERT_EQ(GetModelMetadataImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::MODEL_NAME_MISSING);

    model_spec->Clear();
    model_spec->set_name("dummy");
    model_spec->mutable_version()->set_value(2);
    ASSERT_EQ(GetModelMetadataImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::MODEL_VERSION_MISSING);

    model_spec->Clear();
    model_spec->set_name("dummy");
    ASSERT_EQ(GetModelMetadataImpl::getModelStatus(&req, &res, manager, DEFAULT_TEST_CONTEXT), StatusCode::MODEL_VERSION_MISSING);
}

class ServerShutdownGuard {
    ovms::Server& ovmsServer;

public:
    ServerShutdownGuard(ovms::Server& ovmsServer) :
        ovmsServer(ovmsServer) {}
    ~ServerShutdownGuard() {
        ovmsServer.shutdownModules();
    }
};

class TestEnabledConfig : public ovms::Config {
public:
    TestEnabledConfig() :
        Config() {
        std::string port{"9000"};
        randomizeAndEnsureFree(port);
        this->serverSettings.grpcPort = std::stoul(port);
    }
};
TEST_F(TestImplGetModelStatus, NegativeKfsGetModelStatus) {
    // Create config file with an empty config & reload
    std::string configStr = dummy_config;
    configStr = configStr.replace(configStr.find("/tmp/test_cl_models"), std::string("/tmp/test_cl_models").size(), cl_models_path);
    std::string fileToReload = cl_models_path + "/cl_config.json";
    createConfigFileWithContent(configStr, fileToReload);

    // Copy dummy model to temporary destination
    std::filesystem::copy(getGenericFullPathForSrcTest("/ovms/src/test/dummy"), cl_model_1_path, std::filesystem::copy_options::recursive);

    ASSERT_EQ(manager.loadConfig(fileToReload), ovms::StatusCode::OK);

    KFSModelMetadataRequest req;
    KFSModelMetadataResponse res;
    KFSModelExtraMetadata extraMetadata;

    req.Clear();
    req.set_name("dummy2");
    req.set_version("2");

    std::unique_ptr<ServerShutdownGuard> serverGuard;
    ovms::Server& server = ovms::Server::instance();
    SetUpSingleModel(cl_models_path, "dummy");
    TestEnabledConfig config;
    auto retCode = server.startModules(config);
    EXPECT_TRUE(retCode.ok()) << retCode.string();
    serverGuard = std::make_unique<ServerShutdownGuard>(server);

    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();

    ASSERT_EQ(impl.ModelMetadataImpl(nullptr, &req, &res, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata), extraMetadata), StatusCode::MODEL_NAME_MISSING);
    req.Clear();
    req.set_name("dummy");
    req.set_version("2");
    ASSERT_EQ(impl.ModelMetadataImpl(nullptr, &req, &res, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata), extraMetadata), StatusCode::MODEL_VERSION_MISSING);

    req.Clear();
    req.set_name("dummy");
    ASSERT_EQ(impl.ModelMetadataImpl(nullptr, &req, &res, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata), extraMetadata), StatusCode::MODEL_VERSION_MISSING);

    req.Clear();
    req.set_name("dummy");
    req.set_version("$$");
    ASSERT_EQ(impl.ModelMetadataImpl(nullptr, &req, &res, ovms::ExecutionContext(ovms::ExecutionContext::Interface::GRPC, ovms::ExecutionContext::Method::GetModelMetadata), extraMetadata), StatusCode::MODEL_VERSION_INVALID_FORMAT);

#ifdef _WIN32
    // Unload model to allow folder delete on Windows
    std::shared_ptr<ovms::ModelInstance> modelInstance1;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    manager.getModelInstance("dummy", 1, modelInstance1, modelInstanceUnloadGuard);
    // Release guard
    modelInstanceUnloadGuard.reset();
    // Unload model
    modelInstance1->retireModel();
#endif
}
