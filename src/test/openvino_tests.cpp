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

TEST_F(OpenVINO, String) {
    std::vector<std::string> data{{"Intel"}, {"CCGafawfaw"}, {"aba"}};
    ov::Shape shape{3};
    ov::Tensor t(ov::element::string, shape, data.data());
    EXPECT_EQ(t.get_byte_size() + 1, data.size() * sizeof(std::string));
}
TEST_F(OpenVINO, CallbacksTest) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    auto input = model->get_parameters().at(0);
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
