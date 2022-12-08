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
    ("grpc_address", "Specify url to grpc service. ", cxxopts::value<std::string>()->default_value("localhost"), "GRPC_ADDRESS")
    ("grpc_port", "Specify port to grpc service. ", cxxopts::value<std::string>()->default_value("9000"), "PORT")
    ("input_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("b"), "INPUT_NAME")
    ("output_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("a"), "OUTPUT_NAME")
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("dummy"), "MODEL_NAME")
    ("model_version", "Define model version.", cxxopts::value<std::string>(), "MODEL_VERSION")
    ("timeout", "Request timeout.", cxxopts::value<int>()->default_value("0"), "TIMEOUT")
    ;
    // clang-format on

    auto args = opt.parse(argc, argv);

    if (args.count("help")) {
        std::cout << opt.help() << std::endl;
        exit(0);
    }

    std::string input_name(args["input_name"].as<std::string>());
    std::string output_name(args["output_name"].as<std::string>());

    std::string url(args["grpc_address"].as<std::string>() + ":" + args["grpc_port"].as<std::string>());
    std::string model_name = args["model_name"].as<std::string>();

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;

    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url),
        err);

    std::vector<float> input_data(10);
    for (size_t i = 0; i < 10; ++i) {
        input_data[i] = i;
    }

    std::vector<int64_t> shape{1, 10};

    // Initialize the inputs with the data.
    tc::InferInput* input;

    FAIL_IF_ERR(
        tc::InferInput::Create(&input, input_name, shape, "FP32"),
        "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);

    FAIL_IF_ERR(
        input_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input_data[0]),
            input_data.size() * sizeof(float)),
        "unable to set data for input");


    tc::InferOptions options(model_name);
    if (args.count("model_version"))
        options.model_version_ = args["model_version"].as<std::string>();
    options.client_timeout_ = args["timeout"].as<int>();

    std::vector<tc::InferInput*> inputs = {input_ptr.get()};

    tc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(&results, options, inputs),
        "unable to run model");
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    // Validate the results...
    ValidateShapeAndDatatype(output_name, results_ptr);

    // Get pointer to the result returned...
    float* output_data;
    size_t output_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            output_name, (const uint8_t**)&output_data, &output_byte_size),
        "unable to get result data for output");
    if (output_byte_size != 40) {
        std::cerr << "error: received incorrect byte size for output: "
                  << output_byte_size << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < 10; ++i) {
        std::cout << input_data[i] << " => "
                  << *(output_data + i) << std::endl;

        if ((input_data[i] + 1) != *(output_data + i)) {
            std::cerr << "error: Incorrect sum" << std::endl;
        }
    }

    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "Number of requests: "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "Total processing time: "
              << double(infer_stat.cumulative_total_request_time_ns)/1.0e+6 << " ms" << std::endl;
    std::cout << "Latency: "
              << double(infer_stat.cumulative_total_request_time_ns/infer_stat.completed_request_count)/1.0e+6 << " ms" << std::endl;
    std::cout << "Requests per second: "
              << double(1.0e+9/infer_stat.cumulative_total_request_time_ns/infer_stat.completed_request_count) << std::endl;

    return 0;
}