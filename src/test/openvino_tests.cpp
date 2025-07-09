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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>

#include "../ov_utils.hpp"
#include "../ovms.h"  // NOLINT
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ov;

class OpenVINO : public ::testing::Test {
};

TEST_F(OpenVINO, TensorCopyDoesNotCopyUnderlyingData) {
    std::vector<float> data{1, 2, 3};
    ov::Shape shape{3};
    ov::Tensor t(ov::element::f32, shape, data.data());
    ov::Tensor t2(ov::element::f32, shape);
    EXPECT_NE(t2.data(), t.data());
    t2 = t;
    EXPECT_EQ(t2.data(), t.data());
}

TEST_F(OpenVINO, String) {
    std::vector<std::string> data{{"Intel"}, {"CCGafawfaw"}, {"aba"}};
    ov::Shape shape{3};
    ov::Tensor t(ov::element::string, shape, data.data());
    EXPECT_EQ(t.get_byte_size(), data.size() * sizeof(std::string));
}
TEST_F(OpenVINO, CallbacksTest) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    ov::element::Type_t dtype = ov::element::Type_t::f32;
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(10000);
    std::map<std::string, ov::PartialShape> inputShapes;
    inputShapes[inputName] = ovShape;
    model->reshape(inputShapes);
    auto cpuCompiledModel = core.compile_model(model, "CPU");
    auto cpuInferRequest = cpuCompiledModel.create_infer_request();
    // prepare ov::Tensor data
    std::vector<ov::Tensor> inputOvTensors, outputOvTensors;
    inputOvTensors.emplace_back(dtype, ovShape);
    outputOvTensors.emplace_back(dtype, ovShape);
    cpuInferRequest.set_tensor(inputName, inputOvTensors[0]);

    uint32_t callbackUsed = 31;
    OVMS_InferenceResponse* response{nullptr};
    cpuInferRequest.set_callback([&cpuInferRequest, &response, &callbackUsed](std::exception_ptr exception) {
        if (exception) {
            try {
                std::rethrow_exception(exception);
            } catch (const std::exception& e) {
                std::cout << "Caught exception: '" << e.what() << "'\n";
            } catch (...) {
                return;
            }
        }
        SPDLOG_INFO("Using OV callback");
        // here will go OVMS C-API serialization code
        callbackMarkingItWasUsedWith42(response, 1, reinterpret_cast<void*>(&callbackUsed));
        cpuInferRequest.set_callback([](std::exception_ptr exception) {});
    });
    bool callbackCalled{false};
    cpuInferRequest.start_async();
    EXPECT_FALSE(callbackCalled);
    cpuInferRequest.wait();
    ov::Tensor outOvTensor = cpuInferRequest.get_tensor("a");
    auto outAutoTensor = cpuInferRequest.get_tensor("a");
    EXPECT_TRUE(outOvTensor.is<ov::Tensor>());
    EXPECT_TRUE(outAutoTensor.is<ov::Tensor>());
}
TEST_F(OpenVINO, StressInferTest) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    auto input = model->get_parameters().at(0);
    ov::element::Type_t dtype = ov::element::Type_t::f32;
    ov::Shape ovShape;
    ovShape.emplace_back(1);
    ovShape.emplace_back(100000);
    std::map<std::string, ov::PartialShape> inputShapes;
    inputShapes[inputName] = ovShape;
    model->reshape(inputShapes);
    auto cpuCompiledModel = core.compile_model(model, "CPU");
    std::vector<ov::InferRequest> inferRequests;
    SPDLOG_INFO("Starting vector size:{}, vector capacity:{}", inferRequests.size(), inferRequests.capacity());
    inferRequests.resize(0);
    SPDLOG_INFO("Starting vector size:{}, vector capacity:{}", inferRequests.size(), inferRequests.capacity());
    inferRequests.reserve(2);
    SPDLOG_INFO("Starting vector size:{}, vector capacity:{}", inferRequests.size(), inferRequests.capacity());
    //inferRequests.shrink_to_fit();
    // we want to test workload when we increase number of infer requests vector during workload
    // so we start with vector of 1, start workload on it
    // then after 1s we start another thread which will add another infer request to the vector
    // ideally we ensure that vector does realocate memory so it forces move of the objects inside it

    // first write function that will be done in thread. It will get reference to inferRequests vector
    // it will create ov::Tensor with passed dtype and ovShape. It will set the content of that vector to i-th
    // so that we will check content of response each time. It will perform workload until it gets signal by future
    auto loadFunction = [&cpuCompiledModel, &inferRequests, inputName, dtype, ovShape](size_t i, std::future<void> stopSignal) {
        SPDLOG_INFO("Starting loadFunction:{}", i);
        inferRequests.emplace_back(cpuCompiledModel.create_infer_request());
        SPDLOG_INFO("Starting shrinkToFit:{} vector size:{}, vector capacity:{}", i, inferRequests.size(), inferRequests.capacity());
        inferRequests.shrink_to_fit();
        SPDLOG_INFO("After shrinkToFit:{} vector size:{}, vector capacity:{}", i, inferRequests.size(), inferRequests.capacity());
        auto& inferRequest = inferRequests[i];
        // prepare ov::Tensor data
        ov::Tensor inputOvTensor(dtype, ovShape);
        ov::Tensor outputOvTensor(dtype, ovShape);
        for (size_t j = 0; j < 100000; j++) {
            reinterpret_cast<float*>(inputOvTensor.data())[j] = i;
            reinterpret_cast<float*>(outputOvTensor.data())[j] = (i + 1);
            if (j<10 || j > 99990) {
            SPDLOG_ERROR("input data: {}, expected: {}, i:{}, j:{}", reinterpret_cast<float*>(inputOvTensor.data())[j], reinterpret_cast<float*>(outputOvTensor.data())[j], i, j);
            }
        }

        // now while loop that stops only if we get stop signal
        SPDLOG_INFO("Running infer request {}", i);
        size_t k = 0;
        while (stopSignal.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout) {
            inferRequest.set_tensor(inputName, inputOvTensor);
            inferRequest.start_async();
            inferRequest.wait();
            auto outOvTensor = inferRequest.get_tensor("a");
            for (size_t j = 0; j < 100000; j++) {
                 if (j<10 || j > 99990) {
                 SPDLOG_ERROR("infReqRef:{} infReq[i]:{} outTensor data: {}, expected: {} i:{} j:{} k:{}", (void*)(&inferRequest), (void*)(&inferRequests[i]),reinterpret_cast<float*>(outOvTensor.data())[j], reinterpret_cast<float*>(outputOvTensor.data())[j], i, j , k);
                 }
            }
            ASSERT_EQ(0, std::memcmp(outOvTensor.data(), outputOvTensor.data(), outOvTensor.get_byte_size())) << "i: " << i;
            ASSERT_EQ(0, std::memcmp(outOvTensor.data(), outputOvTensor.data(), outOvTensor.get_byte_size())) << "i: " << i;
            k++;
        }
    };
    size_t n = 2;
    std::vector<std::promise<void>> stopSignal(n);
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        // create thread that will run loadFunction
        SPDLOG_INFO("Starting thread {}", i);
        threads.emplace_back(loadFunction, i, stopSignal[i].get_future());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    for (size_t i = 0; i < n; ++i) {
        // create thread that will run loadFunction
        SPDLOG_INFO("Stopping thread {}", i);
        stopSignal[i].set_value();
    }
    for (size_t i = 0; i < n; ++i) {
        SPDLOG_INFO("Joining thread {}", i);
        threads[i].join();
    }
}
TEST_F(OpenVINO, ResetOutputTensors) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
        ov::auto_batch_timeout(0)};
    auto compiledModel = core.compile_model(model, "CPU", config);
    std::vector<float> in1(DUMMY_MODEL_INPUT_SIZE, 0.1);
    ov::Shape shape;
    shape.emplace_back(1);
    shape.emplace_back(DUMMY_MODEL_INPUT_SIZE);
    ov::element::Type_t dtype = ov::element::Type_t::f32;
    ov::Tensor input1(dtype, shape, in1.data());
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(DUMMY_MODEL_INPUT_NAME, input1);
    // keep original output tensor
    ov::Tensor originalOutput = inferRequest.get_tensor(DUMMY_MODEL_OUTPUT_NAME);
    // set output
    std::vector<float> out(DUMMY_MODEL_INPUT_SIZE, 15124.1);
    ov::Tensor output(dtype, shape, out.data());
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, output);
    inferRequest.infer();
    for (size_t i = 0; i < DUMMY_MODEL_INPUT_SIZE; ++i) {
        EXPECT_NEAR(in1[i] + 1, out[i], 0.0004) << "i:" << i;
    }
    std::vector<float> in2(DUMMY_MODEL_INPUT_SIZE, 42);
    ov::Tensor input2(dtype, shape, in2.data());
    inferRequest.set_tensor(DUMMY_MODEL_INPUT_NAME, input2);
    inferRequest.set_tensor(DUMMY_MODEL_OUTPUT_NAME, originalOutput);
    inferRequest.infer();
    auto secondOutput = inferRequest.get_tensor(DUMMY_MODEL_OUTPUT_NAME);
    float* data2nd = reinterpret_cast<float*>(secondOutput.data());
    for (size_t i = 1; i < DUMMY_MODEL_INPUT_SIZE; ++i) {
        EXPECT_NEAR(in2[i] + 1, data2nd[i], 0.0004) << "i:" << i;
    }
    // now check if first output didn't change content
    for (size_t i = 0; i < DUMMY_MODEL_INPUT_SIZE; ++i) {
        EXPECT_NEAR(in1[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}
