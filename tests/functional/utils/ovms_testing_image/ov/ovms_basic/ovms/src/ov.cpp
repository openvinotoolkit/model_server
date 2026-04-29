//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <thread>
#include <string>
#include <iostream>

#include <openvino/openvino.hpp>

int main(int argc, char *argv[]){
    ov::Core ieCore;
    auto model = ieCore.read_model(argv[1]);

    // Print info of first input layer
    std::cout << model->input(0).get_partial_shape() << "\n";

    // Compile model
    ov::CompiledModel compiledModel = ieCore.compile_model(model, argv[2]);

    // Create an inference request
    ov::InferRequest infer_request = compiledModel.create_infer_request();

    // Get input port for model with one input
    auto input_port = compiledModel.input();

    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape());

    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    infer_request.wait();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    compiledModel = {};
    std::cout << "Hello\n" << std::flush;
}
