//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                              \
    {                                                                    \
        tc::Error err = (X);                                             \
        if (!err.IsOk()) {                                               \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }

namespace {

void ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result) {
    std::vector<int64_t> shape;
    FAIL_IF_ERR(
        result->Shape(name, &shape), "unable to get shape for '" + name + "'");
    // Validate shape
    if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 10)) {
        std::cerr << "error: received incorrect shapes for '" << name << "'"
                  << std::endl;
        exit(1);
    }
    std::string datatype;
    FAIL_IF_ERR(
        result->Datatype(name, &datatype),
        "unable to get datatype for '" + name + "'");
    // Validate datatype
    if (datatype.compare("FP32") != 0) {
        std::cerr << "error: received incorrect datatype for '" << name
                  << "': " << datatype << std::endl;
        exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options opt("grpc_infer_dummy", "Sends requests via KServe gRPC API.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("grpc_address", "Specify url to grpc service. default:localhost", cxxopts::value<std::string>()->default_value("localhost"))
    ("grpc_port", "Specify port to grpc service. default:9000", cxxopts::value<std::string>()->default_value("9000"))
    ("input_name", "Specify input tensor name. default: input", cxxopts::value<std::string>()->default_value("input"))
    ("output_name", "Specify input tensor name. default: output", cxxopts::value<std::string>()->default_value("output"))
    ("model_name", "Define model name, must be same as is in service. default: dummy", cxxopts::value<std::string>()->default_value("dummy"))
    ("model_version", "Define model version.")
    ("timeout", "", cxxopts::value<int>()->default_value("0"))
    ;
    // clang-format on

    auto args = opt.parse(argc, argv);

    if (args.count("help")) {
        std::cout << opt.help() << std::endl;
        exit(0);
    }

    std::string url(args["grpc_address"].as<std::string>() + ":" + args["grpc_port"].as<std::string>());
    std::string model_name = args["model_name"].as<std::string>();

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;

    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url),
        err);

    std::vector<float> inputb_data(10);
    for (size_t i = 0; i < 10; ++i) {
        inputb_data[i] = i;
    }

    std::vector<int64_t> shape{1, 10};

    // Initialize the inputs with the data.
    tc::InferInput* inputb;

    FAIL_IF_ERR(
        tc::InferInput::Create(&inputb, "b", shape, "FP32"),
        "unable to get b");
    std::shared_ptr<tc::InferInput> inputb_ptr;
    inputb_ptr.reset(inputb);

    FAIL_IF_ERR(
        inputb_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&inputb_data[0]),
            inputb_data.size() * sizeof(float)),
        "unable to set data for b");

    // Generate the outputs to be requested.
    tc::InferRequestedOutput* outputa;

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&outputa, "a"),
        "unable to get 'a'");
    std::shared_ptr<tc::InferRequestedOutput> outputa_ptr;
    outputa_ptr.reset(outputa);

    tc::InferOptions options(model_name);
    if (args.count("model_version"))
        options.model_version_ = args["model_version"].as<std::string>();
    options.client_timeout_ = args["timeout"].as<int>();

    std::vector<tc::InferInput*> inputs = {inputb_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {outputa_ptr.get()};

    tc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(&results, options, inputs, outputs),
        "unable to run model");
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    // Validate the results...
    ValidateShapeAndDatatype("a", results_ptr);

    // Get pointers to the result returned...
    float* outputa_data;
    size_t outputa_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "a", (const uint8_t**)&outputa_data, &outputa_byte_size),
        "unable to get result data for 'a'");
    if (outputa_byte_size != 40) {
        std::cerr << "error: received incorrect byte size for 'a': "
                  << outputa_byte_size << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < 10; ++i) {
        std::cout << inputb_data[i] << " => "
                  << *(outputa_data + i) << std::endl;

        if ((inputb_data[i] + 1) != *(outputa_data + i)) {
            std::cerr << "error: Incorrect sum" << std::endl;
        }
    }

    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "completed_request_count "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "cumulative_total_request_time_ns "
              << infer_stat.cumulative_total_request_time_ns << std::endl;
    std::cout << "cumulative_send_time_ns "
              << infer_stat.cumulative_send_time_ns << std::endl;
    std::cout << "cumulative_receive_time_ns "
              << infer_stat.cumulative_receive_time_ns << std::endl;


    return 0;
}