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
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>

#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "grpc_client.h"

namespace tc = triton::client;

#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define IMG_C 3

#define FAIL_IF_ERR(X, MSG)                                              \
    {                                                                    \
        tc::Error err = (X);                                             \
        if (!err.IsOk()) {                                               \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }

std::vector<uint8_t> Load(std::string fileName) {
    std::ifstream input(fileName, std::ios::binary);

    std::vector<uint8_t> bytes(
         (std::istream_iterator<uint8_t>(input)),
         (std::istream_iterator<uint8_t>()));

    input.close();
    return bytes;
}

int main(int argc, char** argv) {
    cxxopts::Options opt("grpc_infer_resnet", "Sends requests via KServe gRPC API.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("images_list", "Path to a file with a list of labeled images. ", cxxopts::value<std::string>())
    ("labels_list", "Path to a file with a list of labels. ", cxxopts::value<std::string>())
    ("grpc_address", "Specify url to grpc service. ", cxxopts::value<std::string>()->default_value("localhost"))
    ("grpc_port", "Specify port to grpc service. ", cxxopts::value<std::string>()->default_value("9000"))
    ("input_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("0"))
    ("output_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("1463"))
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("resnet"))
    ("model_version", "Define model version.", cxxopts::value<std::string>())
    ("timeout", "Request timeout.", cxxopts::value<int>()->default_value("0"))
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

    std::string img;
    int label;
    std::vector<std::string> imgs;
    std::vector<int> labels;
    std::ifstream images(args["images_list"].as<std::string>());
    while (images >> img >> label) {
        imgs.push_back(img);
        labels.push_back(label);
    }


    std::vector<int64_t> shape{1};

    // Initialize the inputs with the data.
    tc::InferInput* input;

    FAIL_IF_ERR(
        tc::InferInput::Create(&input, input_name, shape, "BYTES"),
        "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);

    tc::InferOptions options(model_name);
    if (args.count("model_version"))
        options.model_version_ = args["model_version"].as<std::string>();
    options.client_timeout_ = args["timeout"].as<int>();

    std::vector<tc::InferInput*> inputs = {input_ptr.get()};

    std::vector<tc::InferResult*> results;
    results.resize(imgs.size());
    for (int i = 0; i < imgs.size(); i++) {
        std::vector<uint8_t> input_data = Load(imgs[i]);
        FAIL_IF_ERR(
            input_ptr->AppendRaw(input_data),
            "unable to set data for input");
        FAIL_IF_ERR(
            client->Infer(&(results[i]), options, inputs),
            "unable to run model");
        input->Reset();
    }

    std::vector<std::string> classes;
    std::ifstream lb_f(args["labels_list"].as<std::string>());
    std::string tmp;
    while (std::getline(lb_f, tmp)) {
        classes.push_back(tmp);
    }

    float acc = 0;
    for (int i = 0; i < imgs.size(); i++) {
        std::shared_ptr<tc::InferResult> results_ptr;
        results_ptr.reset(results[i]);
        // Get pointers to the result returned...
        float* output_data;
        size_t output_byte_size;
        FAIL_IF_ERR(
            results_ptr->RawData(
                output_name, (const uint8_t**)&output_data, &output_byte_size),
            "unable to get result data for output");

        int lb = std::distance(output_data, std::max_element(output_data, output_data + 1000));
        std::cout << imgs[i] << " classified as "
                  << lb << " " << classes[lb] << " ";
        if (lb != labels[i]) {
            std::cout << "should be " << labels[i] << " " << classes[labels[i]];
        } else {
            acc++;
        }
        std::cout << std::endl;
    }

    std::cout << "Accuracy " << acc / imgs.size() * 100 << "%\n";

    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "Completed request count "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "Cumulative total request time "
              << double(infer_stat.cumulative_total_request_time_ns) / 1000000 << " ms" << std::endl;
    std::cout << "Cumulative send time "
              << double(infer_stat.cumulative_send_time_ns) / 1000000 << " ms" << std::endl;
    std::cout << "Cumulative receive time "
              << double(infer_stat.cumulative_receive_time_ns) / 1000000 << " ms" << std::endl;

    return 0;
}