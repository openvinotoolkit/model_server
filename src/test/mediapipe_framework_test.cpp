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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/outputstreamobserver.hpp"
#include "../mediapipe_internal/mediapipefactory.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../mediapipe_internal/node_initializer.hpp"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "src/metrics/metric_config.hpp"
#include "src/metrics/metric_module.hpp"
#include "../precision.hpp"
#include "src/servable_management/servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "../stringutils.hpp"
#include "src/timer.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "c_api_test_utils.hpp"
#include "environment.hpp"
#include "test_utils.hpp"
#include "platform_utils.hpp"
#include "test_with_temp_dir.hpp"

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
TEST_F(MediapipeFrameworkTest, HotReloadOutputStreamHandlerCheckNoInputPackets) {
    // we need it only so that dummy is available via C-API
    ServerGuard servGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_benchmark.json"));
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
    // avoid creating pollers, retrieving packets etc.
    //////////////////
    // model mgmt thread
    //////////////////
    // std::shared_ptr<ovms::GraphQueue> queue;
    // queue = std::make_shared<GraphQueue>(graphConfig, 1);
    ::mediapipe::CalculatorGraph graph;
    EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
    // Install NullObserver
    // its not per graph but per output
    std::shared_ptr<OutputStreamObserverI> perGraphObserverFunctor = std::make_shared<NullOutputStreamObserver>();
    const std::string outputName{"output"};
    absl::Status absStatus;
    MP_ERROR_STOP(graph.ObserveOutputStream(outputStreamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); }));
    // Here ends model management
    // Here starts mp graph executor
    // ovms::GraphIdGuard graphIdGuard(queue); // TODO timeout?
    // get graphIdGuard from queue
    // create FrontendAppropriateObserver
    float expVal = 13.5;
    struct MyFunctor : public OutputStreamObserverI {
        float expVal;
        MyFunctor(float expVal) :
            expVal(expVal) {
            SPDLOG_TRACE("MyFunctor observer constructed:{}", (void*)this);
        }
        absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
            SPDLOG_TRACE("my functor:{}", (void*)this);
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
    SPDLOG_TRACE("Now swap Functor, we don't have to call ObserverOutputStream");
    expVal = 42;
    data[0] = expVal - 1;
    perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
    // now add second packet
    auto inputTensor2 = std::make_unique<ov::Tensor>(datatype, shape, data.data());
    // MP_ERROR_STOP(graph.AddPacketToInputStream(
    //    inputStreamName, Adopt(inputTensor2.release()).At(Timestamp(timestamp++))));
    // MP_ERROR_STOP(graph.WaitUntilIdle());
    MP_ERROR_STOP(graph.CloseAllPacketSources());
    MP_ERROR_STOP(graph.WaitUntilDone());
}
TEST_F(MediapipeFrameworkTest, HotReloadOutputStreamHandlerPOC) {
    // we need it only so that dummy is available via C-API
    ServerGuard servGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_benchmark.json"));
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
    // avoid creating pollers, retrieving packets etc.
    //////////////////
    // model mgmt thread
    //////////////////
    // std::shared_ptr<ovms::GraphQueue> queue;
    // queue = std::make_shared<GraphQueue>(graphConfig, 1);
    ::mediapipe::CalculatorGraph graph;
    EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
    // Install NullObserver
    // its not per graph but per output
    std::shared_ptr<OutputStreamObserverI> perGraphObserverFunctor = std::make_shared<NullOutputStreamObserver>();
    const std::string outputName{"output"};
    absl::Status absStatus;
    MP_ERROR_STOP(graph.ObserveOutputStream(outputStreamName, [&perGraphObserverFunctor](const ::mediapipe::Packet& packet) -> absl::Status { return perGraphObserverFunctor->handlePacket(packet); }));
    // Here ends model management
    // Here starts mp graph executor
    // ovms::GraphIdGuard graphIdGuard(queue); // TODO timeout?
    // get graphIdGuard from queue
    // create FrontendAppropriateObserver
    float expVal = 13.5;
    struct MyFunctor : public OutputStreamObserverI {
        float expVal;
        MyFunctor(float expVal) :
            expVal(expVal) {
        }
        absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
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
    SKIP_AND_EXIT_IF_NOT_RUNNING_UNSTABLE();
    // we need it only so that dummy is available via C-API
    ServerGuard servGuard(getGenericFullPathForSrcTest("/ovms/src/test/configs/config_standard_dummy.json"));
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
    // avoid creating pollers, retrieving packets etc.
    //////////////////
    // model mgmt thread
    //////////////////
    // std::shared_ptr<ovms::GraphQueue> queue;
    // queue = std::make_shared<GraphQueue>(graphConfig, 1);
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
        // ovms::GraphIdGuard graphIdGuard(queue); // TODO timeout?
        // get graphIdGuard from queue
        // create FrontendAppropriateObserver
        struct MyFunctor : public OutputStreamObserverI {
            float expVal;
            MyFunctor(float expVal) :
                expVal(expVal) {
            }
            absl::Status handlePacket(const ::mediapipe::Packet& packet) override {
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
        {
            perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
            auto copyOfMyFunctor = perGraphObserverFunctor;
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        timer.start(0);
        for (auto i = 0; i < N; ++i) {  // iter begin
            perGraphObserverFunctor = std::make_shared<MyFunctor>(expVal);
            auto copyOfMyFunctor = perGraphObserverFunctor;
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            MP_ERROR_STOP(graph.WaitUntilIdle());
        }  // iter end
        timer.stop(0);
    }  // end of new case ovms
    {  // current ovms case
        timer.start(1);
        for (auto i = 0; i < N; ++i) {  // iter begin
            ::mediapipe::CalculatorGraph graph;
            EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
            auto absStatusOrPoller = graph.AddOutputStreamPoller(outputName);
            MP_ERROR_STOP(graph.StartRun({}));
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            ::mediapipe::Packet packet;
            absStatusOrPoller.value().Next(&packet);
            const ov::Tensor& outputTensor =
                packet.Get<ov::Tensor>();
            auto datatype = ov::element::Type_t::f32;
            EXPECT_EQ(datatype, outputTensor.get_element_type());
            EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
            const void* outputData = outputTensor.data();
            EXPECT_EQ(*((float*)outputData), expVal);
            MP_ERROR_STOP(graph.WaitUntilIdle());
            MP_ERROR_STOP(graph.CloseAllPacketSources());
            MP_ERROR_STOP(graph.WaitUntilDone());
        }  // iter end
        timer.stop(1);
    }
    {  // thread pool case
        // auto sharedThreadPool = std::make_shared<mediapipe::ThreadPoolExecutor>(std::thread::hardware_concurrency());
        auto sharedThreadPool = std::make_shared<mediapipe::ThreadPoolExecutor>(24);
        timer.start(2);
        for (auto i = 0; i < N; ++i) {  // iter begin
            ::mediapipe::CalculatorGraph graph;
            MP_ERROR_STOP(graph.SetExecutor("", sharedThreadPool));
            EXPECT_EQ(graph.Initialize(graphConfig).code(), absl::StatusCode::kOk);
            auto absStatusOrPoller = graph.AddOutputStreamPoller(outputName);
            MP_ERROR_STOP(graph.StartRun({}));
            auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
            MP_ERROR_STOP(graph.AddPacketToInputStream(
                inputStreamName, Adopt(inputTensor.release()).At(Timestamp(timestamp++))));
            ::mediapipe::Packet packet;
            absStatusOrPoller.value().Next(&packet);
            const ov::Tensor& outputTensor =
                packet.Get<ov::Tensor>();
            auto datatype = ov::element::Type_t::f32;
            EXPECT_EQ(datatype, outputTensor.get_element_type());
            EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1, 10));
            const void* outputData = outputTensor.data();
            EXPECT_EQ(*((float*)outputData), expVal);
            MP_ERROR_STOP(graph.WaitUntilIdle());
            MP_ERROR_STOP(graph.CloseAllPacketSources());
            MP_ERROR_STOP(graph.WaitUntilDone());
        }  // iter end
        timer.stop(2);
    }  // end of thread pool case
    double ms = timer.elapsed<std::chrono::microseconds>(0) / 1000;
    SPDLOG_DEBUG("{} iterations of new flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    ms = timer.elapsed<std::chrono::microseconds>(1) / 1000;
    SPDLOG_DEBUG("{} iterations of old flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    ms = timer.elapsed<std::chrono::microseconds>(2) / 1000;
    SPDLOG_DEBUG("{} iterations of thread pool flow took:{} ms. FPS:{}", N, ms, N / ms * 1000);
    SPDLOG_DEBUG("Threads: {}", std::thread::hardware_concurrency());
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
        SPDLOG_ERROR("ER: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("ER: unknown exception");
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
        SPDLOG_ERROR("ER: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("ER: unknown exception");
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
        SPDLOG_ERROR("ER: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("ER: unknown exception");
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
        SPDLOG_ERROR("ER: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("ER: unknown exception");
    }
}

TEST(NodeInitializerRegistryTest, AllExpectedInitializersRegistered) {
    const auto& initializers = ovms::NodeInitializerRegistry::instance().all();
    const std::vector<std::string> knownCalculators = {
        "LLMCalculator",
        "EmbeddingsCalculatorOV",
        "ImageGenCalculator",
        "RerankCalculatorOV",
        "S2tCalculator",
        "T2sCalculator",
        "PythonExecutorCalculator",
    };
    for (const auto& name : knownCalculators) {
        bool found = std::any_of(initializers.begin(), initializers.end(),
            [&](const auto& init) { return init->matches(name); });
        EXPECT_TRUE(found) << "No NodeInitializer registered for calculator: " << name;
    }

    constexpr size_t expectedInitializersCount = 7;
    EXPECT_EQ(initializers.size(), expectedInitializersCount)
        << "Unexpected number of registered NodeInitializers from static registration/link-time inclusion.";
}

// Regression tests for issue #4717: the local stou32() copy in
// OpenVINOModelServerSessionCalculator accepted partial parses ("3.9" -> 3,
// "12abc" -> 12) because it called std::stoul without capturing the consumed
// length.  The fix replaces the local copy with ovms::stou32() and returns an
// explicit InvalidArgument error when servable_version is supplied
// (has_servable_version() is true) but not a valid non-negative integer
// (including explicit empty string "").  These tests exercise the validation path through
// Open() so that no server is required for the failing inputs (the error fires
// before any model load attempt).
class OpenVINOModelServerSessionCalculatorVersionTest : public ::testing::Test {
    // Helper: build a minimal graph proto with the given servable_version
    // string, initialize the graph, and return the status from StartRun().
    // The session calculator will call Open() during StartRun(); if the version
    // string is malformed it returns immediately with kInvalidArgument before
    // touching OVMS, so no ServerGuard is needed for the error path.
protected:
    absl::Status RunWithVersion(const std::string& version_str, bool omit_version = false) {
        std::string version_field = omit_version ? "" : ("servable_version: \"" + version_str + "\"");
        std::string graph_proto = R"(
          node {
              calculator: "OpenVINOModelServerSessionCalculator"
              output_side_packet: "SESSION:session"
              node_options: {
                [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
                  servable_name: "dummy"
                  )" + version_field +
                                  R"(
                }
              }
          }
        )";
        CalculatorGraphConfig graphConfig =
            ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
        ::mediapipe::CalculatorGraph graph;
        auto initStatus = graph.Initialize(graphConfig);
        if (!initStatus.ok()) {
            return initStatus;
        }
        auto runStatus = graph.StartRun({});
        if (!runStatus.ok()) {
            return runStatus;
        }
        return graph.WaitUntilDone();
    }
};

// Malformed or invalid inputs (including explicit empty string) must be rejected
// with kInvalidArgument.  These tests do not
// require a running OVMS server because the calculator returns before attempting
// any model load.
TEST_F(OpenVINOModelServerSessionCalculatorVersionTest, MalformedDecimalRejected) {
    auto status = RunWithVersion("3.9");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.ToString();
    EXPECT_THAT(status.ToString(), testing::HasSubstr("3.9"));
}

TEST_F(OpenVINOModelServerSessionCalculatorVersionTest, MalformedTrailingAlphaRejected) {
    auto status = RunWithVersion("12abc");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.ToString();
    EXPECT_THAT(status.ToString(), testing::HasSubstr("12abc"));
}

TEST_F(OpenVINOModelServerSessionCalculatorVersionTest, MalformedHexPrefixRejected) {
    auto status = RunWithVersion("0x3");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.ToString();
    EXPECT_THAT(status.ToString(), testing::HasSubstr("0x3"));
}

TEST_F(OpenVINOModelServerSessionCalculatorVersionTest, ExplicitEmptyStringRejected) {
    auto status = RunWithVersion("");
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status.ToString();
    EXPECT_THAT(status.ToString(), testing::HasSubstr("is not a valid non-negative integer"));
}

// When servable_version is valid or omitted, validation passes and execution proceeds
// to session->loadModel(). Without a live OVMS server running, loadModel() fails with
// kFailedPrecondition ("failed to load the model") rather than kInvalidArgument.
TEST_F(OpenVINOModelServerSessionCalculatorVersionTest, ValidVersionsPassValidation) {
    for (const auto& [version, omit] : std::vector<std::pair<std::string, bool>>{
             {"0", false},
             {"12", false},
             {"", true},
         }) {
        auto status = RunWithVersion(version, omit);
        EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition)
            << "Expected kFailedPrecondition (model load failure past validation) for version: "
            << (omit ? "<omitted>" : version) << ", got: " << status.ToString();
        EXPECT_THAT(status.ToString(), testing::HasSubstr("failed to load the model"));
    }
}
