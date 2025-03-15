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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "../config.hpp"
#include "../dags/pipelinedefinition.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/outputstreamobserver.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "mediapipe/framework/thread_pool_executor.h"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

/***
 * Test assumptions about MP framework
 * */
class MediapipeFrameworkTest : public TestWithTempDir {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
    }

    void SetUp() override {
    }
    void TearDown() {
        if (server.isLive(CAPI_MODULE_NAME)) {
            server.setShutdownRequest(1);
            t->join();
            server.setShutdownRequest(0);
        }
    }
};

class MediapipeNegativeFrameworkTest : public MediapipeFrameworkTest {
};

using mediapipe::Adopt;
using mediapipe::CalculatorGraphConfig;
using mediapipe::Packet;
using mediapipe::ParseTextProtoOrDie;
using mediapipe::Timestamp;

#define MP_ERROR_STOP(A)                                         \
    {                                                            \
        absStatus = A;                                           \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{}", absMessage);                      \
            ASSERT_TRUE(false);                                  \
        }                                                        \
    }
TEST_F(MediapipeFrameworkTest, HotReloadOutputStreamHandlerPOC) {
    // we need it only so that dummy is available via C-API
//    ServerGuard servGuard("/ovms/src/test/configs/config_standard_dummy.json");
    ServerGuard servGuard("/ovms/src/test/configs/config_benchmark.json");
    std::string graph_proto = R"(
      input_stream: "IN:input"
      output_stream: "OUT:output"
      node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "dummy"
            }
          }
      }
      node {
        calculator: "OpenVINOInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "OVTENSOR:input"
        output_stream: "OVTENSOR:output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                tag_to_input_tensor_names {
                    key: "OVTENSOR"
                    value: "b"
                }
                tag_to_output_tensor_names {
                    key: "OVTENSOR"
                    value: "a"
                }
            }
        }
      }
    )";
    CalculatorGraphConfig graphConfig =
        ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
    const std::string inputStreamName = "input";
    const std::string outputStreamName = "output";
    // avoid creating pollers, retreiving packets etc.
    //////////////////
    // model mgmt thread
    //////////////////
    //std::shared_ptr<ovms::GraphQueue> queue;
    //queue = std::make_shared<GraphQueue>(graphConfig, 1);
    ::mediapipe::CalculatorGraph graph;
    EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
    // Install NullObserver
    // its not per graph but per output
    std::shared_ptr<ovms::OutputStreamObserverI> perGraphObserverFunctor = std::make_shared<NullOutputStreamObserver>();
    const std::string outputName{"output"};
    absl::Status absStatus;
    MP_ERROR_STOP(graph.ObserveOutputStream(outputStreamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); }));
    // Here ends model management
    // Here starts mp graph executor
    //ovms::GraphIdGuard graphIdGuard(queue); // TODO timeout?
    // get graphIdGuard from queue
    // create FrontendAppropriateObserver
    float expVal = 13.5;
    struct MyFunctor : public OutputStreamObserverI {
        float expVal;
        MyFunctor(float expVal) :
            expVal(expVal) {
            SPDLOG_ERROR("MyFunctor observer constructed:{}", (void*)this);
        }
        absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
            SPDLOG_ERROR("ER my functor:{}", (void*)this);
            const ov::Tensor& outputTensor =
                packet.Get<ov::Tensor>();
            auto datatype = ov::element::Type_t::f32;
            EXPECT_EQ(datatype, outputTensor.get_element_type());
            EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
            const void* outputData = outputTensor.data();
            EXPECT_EQ(*((float*)outputData), expVal);
            return absl::OkStatus();
        }
    };
    perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
    auto copyOfMyFunctor = perGraphObserverFunctor;
    // now start execution
    absStatus = graph.StartRun({});
    auto datatype = ov::element::Type_t::f32;
    ov::Shape shape{1, 10};
    int timestamp{0};
    std::vector<float> data{expVal - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
    MP_ERROR_STOP(graph.AddPacketToInputStream(
        inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
    MP_ERROR_STOP(graph.WaitUntilIdle());
    SPDLOG_ERROR("Now swap Functor, we don't have to call ObserverOutputStream");
    expVal = 42;
    data[0] = expVal - 1;
    perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
    // now add second packet
    auto inputTensor2 = std::make_unique<ov::Tensor>(datatype, shape, data.data());
    MP_ERROR_STOP(graph.AddPacketToInputStream(
        inputStreamName, Adopt(inputTensor2.release()).At(Timestamp(timestamp++))));
    MP_ERROR_STOP(graph.WaitUntilIdle());
}
TEST_F(MediapipeFrameworkTest, HotReloadOutputStreamHandlerPOCCompare) {
    // we need it only so that dummy is available via C-API
    ServerGuard servGuard("/ovms/src/test/configs/config_standard_dummy.json");
    std::string graph_proto = R"(
      input_stream: "IN:input"
      output_stream: "OUT:output"
      node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "dummy"
            }
          }
      }
      node {
        calculator: "OpenVINOInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "OVTENSOR:input"
        output_stream: "OVTENSOR:output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                tag_to_input_tensor_names {
                    key: "OVTENSOR"
                    value: "b"
                }
                tag_to_output_tensor_names {
                    key: "OVTENSOR"
                    value: "a"
                }
            }
        }
      }
    )";
    CalculatorGraphConfig graphConfig =
        ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
    const std::string inputStreamName = "input";
    const std::string outputStreamName = "output";
    // avoid creating pollers, retreiving packets etc.
    //////////////////
    // model mgmt thread
    //////////////////
    //std::shared_ptr<ovms::GraphQueue> queue;
    //queue = std::make_shared<GraphQueue>(graphConfig, 1);
    auto datatype = ov::element::Type_t::f32;
    ov::Shape shape{1, 10};
    int timestamp{0};
    float expVal = 13.5;
    std::vector<float> data{expVal - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ovms::Timer<3> timer;
    const std::string outputName{"output"};
    int N = 1000;

    absl::Status absStatus;
    // here starts new case of ovms
    {  // new case of ovms
        ::mediapipe::CalculatorGraph graph;
        EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
        auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
        // Install NullObserver
        // its not per graph but per output
        std::shared_ptr<ovms::OutputStreamObserverI> perGraphObserverFunctor = std::make_shared<NullOutputStreamObserver>();
        MP_ERROR_STOP(graph.ObserveOutputStream(outputStreamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); }));
        // Here ends model management
        // Here starts mp graph executor
        //ovms::GraphIdGuard graphIdGuard(queue); // TODO timeout?
        // get graphIdGuard from queue
        // create FrontendAppropriateObserver
        struct MyFunctor : public OutputStreamObserverI {
            float expVal;
            MyFunctor(float expVal) :
                expVal(expVal) {
                SPDLOG_ERROR("MyFunctor observer constructed:{}", (void*)this);
            }
            absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
                SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY Getting output tensor:{}", (void*)this);
                const ov::Tensor& outputTensor =
                    packet.Get<ov::Tensor>();
                auto datatype = ov::element::Type_t::f32;
                EXPECT_EQ(datatype, outputTensor.get_element_type());
                EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
                const void* outputData = outputTensor.data();
                EXPECT_EQ(*((float*)outputData), expVal);
                return absl::OkStatus();
            }
        };
        absStatus = graph.StartRun({});
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX warmup");
        {
            perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
            auto copyOfMyFunctor = perGraphObserverFunctor;
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX warmup end");
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX new");
        timer.start(0);
        for (auto i = 0; i < N; ++i) {  // iter begin
            perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
            auto copyOfMyFunctor = perGraphObserverFunctor;
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY AddingPacket");
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY WaitUntilIdle");
            MP_ERROR_STOP(graph.WaitUntilIdle());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY After WaitUntilIdle");
        }  // iter end
        timer.stop(0);
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX end:{}", timer.elapsed<std::chrono::microseconds>(0) / 1000);
    }  // end of new case ovms
    {  // current ovms case
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ovms");
        timer.start(1);
        for (auto i = 0; i < N; ++i) {  // iter begin
            ::mediapipe::CalculatorGraph graph;
            EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
            auto absStatusOrPoller = graph.AddOutputStreamPoller(outputName);
            MP_ERROR_STOP(graph.StartRun({}));
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY AddingPacket");
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            ::mediapipe::Packet packet;
            absStatusOrPoller.value().Next(&packet);
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY Getting output tensor");
            const ov::Tensor& outputTensor =
                packet.Get<ov::Tensor>();
            auto datatype = ov::element::Type_t::f32;
            EXPECT_EQ(datatype, outputTensor.get_element_type());
            EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
            const void* outputData = outputTensor.data();
            EXPECT_EQ(*((float*)outputData), expVal);
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY WaitUntilIdle");
            MP_ERROR_STOP(graph.WaitUntilIdle());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY After WaitUntilIdle");
            MP_ERROR_STOP(graph.CloseAllPacketSources());
            MP_ERROR_STOP(graph.WaitUntilDone());
        }  // iter end
        timer.stop(1);
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX end:{}", timer.elapsed<std::chrono::microseconds>(1) / 1000);
    } 
    { // thread pool case
        //auto sharedThreadPool = std::make_shared<mediapipe::ThreadPoolExecutor>(std::thread::hardware_concurrency());
        auto sharedThreadPool = std::make_shared<mediapipe::ThreadPoolExecutor>(24);
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX thread");
        timer.start(2);
        for (auto i = 0; i < N; ++i) {  // iter begin
            ::mediapipe::CalculatorGraph graph;
            MP_ERROR_STOP(graph.SetExecutor("", sharedThreadPool));
            EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
            auto absStatusOrPoller = graph.AddOutputStreamPoller(outputName);
            MP_ERROR_STOP(graph.StartRun({}));
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY AddingPacket");
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            ::mediapipe::Packet packet;
            absStatusOrPoller.value().Next(&packet);
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY Getting output tensor");
            const ov::Tensor& outputTensor =
                packet.Get<ov::Tensor>();
            auto datatype = ov::element::Type_t::f32;
            EXPECT_EQ(datatype, outputTensor.get_element_type());
            EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
            const void* outputData = outputTensor.data();
            EXPECT_EQ(*((float*)outputData), expVal);
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY WaitUntilIdle");
            MP_ERROR_STOP(graph.WaitUntilIdle());
            SPDLOG_ERROR("YYYYYYYYYYYYYYYYYYYYYYYYYYYYY After WaitUntilIdle");
            MP_ERROR_STOP(graph.CloseAllPacketSources());
            MP_ERROR_STOP(graph.WaitUntilDone());
        }  // iter end
        timer.stop(2);
        SPDLOG_ERROR("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX end:{}", timer.elapsed<std::chrono::microseconds>(2) / 1000);
    } // end of thread pool case
    double ms = timer.elapsed<std::chrono::microseconds>(0) / 1000;
    SPDLOG_ERROR("{} iterations of new flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    ms = timer.elapsed<std::chrono::microseconds>(1) / 1000;
    SPDLOG_ERROR("{} iterations of old flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    ms = timer.elapsed<std::chrono::microseconds>(2) / 1000;
    SPDLOG_ERROR("{} iterations of thread pool flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    SPDLOG_ERROR("Threads: {}", std::thread::hardware_concurrency());
}

TEST_F(MediapipeNegativeFrameworkTest, NoOutputPacketProduced) {
    // purpose of this test is to ensure there is no hang in case of one of the graph nodes
    // not producing output packet
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_no_calc_output_stream.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_no_calc_output_stream"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    auto status = impl.ModelInfer(nullptr, &request, &response);
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
}

TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringProcess) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_process.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_process"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringGetContract) {
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_getcontract.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_getcontract"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringGetOpen) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_open.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_open"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
TEST_F(MediapipeNegativeFrameworkTest, ExceptionDuringClose) {
    GTEST_SKIP() << "Terminate called otherwise";
    SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/negative/config_exception_during_close.json").c_str());
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName{"graph_exception_during_close"};
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    std::vector<float> requestData{13.5, 0., 0, 0., 0., 0., 0., 0, 3., 67.};
    preparePredictRequest(request, inputsMeta, requestData);
    request.mutable_model_name()->assign(modelName);
    try {
        auto status = impl.ModelInfer(nullptr, &request, &response);
        ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT) << status.error_message();
    } catch (std::exception& e) {
        SPDLOG_ERROR("ERs");
    } catch (...) {
        SPDLOG_ERROR("ER");
    }
}
