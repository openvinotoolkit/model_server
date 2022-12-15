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

#include "http_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                              \
    {                                                                    \
        tc::Error err = (X);                                             \
        if (!err.IsOk()) {                                               \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }

std::vector<uint8_t> Load(const std::string& fileName) {
    std::ifstream fileImg(fileName, std::ios::binary);
    fileImg.seekg(0, std::ios::end);
    int bufferLength = fileImg.tellg();
    fileImg.seekg(0, std::ios::beg);

    char* buffer = new char[bufferLength];
    fileImg.read(buffer, bufferLength);

    return std::vector<uint8_t>(buffer, buffer + bufferLength);
}

int main(int argc, char** argv) {
    cxxopts::Options opt("http_infer_resnet", "Sends requests via KServe rest API.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("images_list", "Path to a file with a list of labeled images. ", cxxopts::value<std::string>(), "IMAGES")
    ("labels_list", "Path to a file with a list of labels. ", cxxopts::value<std::string>(), "LABELS")
    ("http_address", "Specify url to REST service. ", cxxopts::value<std::string>()->default_value("localhost"), "GRPC_ADDRESS")
    ("http_port", "Specify port to REST service. ", cxxopts::value<std::string>()->default_value("9000"), "PORT")
    ("input_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("0"), "INPUT_NAME")
    ("output_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("1463"), "OUTPUT_NAME")
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("resnet"), "MODEL_NAME")
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

    std::string url(args["http_address"].as<std::string>() + ":" + args["http_port"].as<std::string>());
    std::string model_name = args["model_name"].as<std::string>();

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerHttpClient> client;

    FAIL_IF_ERR(
        tc::InferenceServerHttpClient::Create(&client, url),
        err);

    std::string img;
    int label = -1;
    std::vector<std::string> imgs;
    std::vector<int> labels;
    if(!args.count("image_list")){
        std::cout<<"error: option \"images_list\" has no value\n";
        exit(1);
    }
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
    try{
        options.client_timeout_ = args["timeout"].as<int>();
    } catch (cxxopts::argument_incorrect_type e) {
        std::cout << "The provided argument is of a wrong type" << std::endl;
        exit(1);
    }

    tc::InferRequestedOutput* output;

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, output_name),
        "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);     

    std::vector<tc::InferInput*> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

    std::vector<tc::InferResult*> results;
    results.resize(imgs.size());
    for (int i = 0; i < imgs.size(); i++) {
        std::vector<uint8_t> input_data = Load(imgs[i]);
        FAIL_IF_ERR(
            input_ptr->AppendRaw(input_data),
            "unable to set data for input");
        FAIL_IF_ERR(
            client->Infer(&(results[i]), options, inputs, outputs),
            "unable to run model");
        input->Reset();
    }

    std::vector<std::string> classes;
    if(!args.count("labels_list")){
        std::cout<<"error: option \"labels_list\" has no value\n";
        exit(1);
    }
    std::ifstream lb_f(args["labels_list"].as<std::string>());
    std::string tmp;
    while (std::getline(lb_f, tmp)) {
        classes.push_back(tmp);
    }

    int acc = 0;
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

    std::cout << "Accuracy " << float(acc) / imgs.size() * 100 << "%\n";

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
              << double(1.0e+9/(infer_stat.cumulative_total_request_time_ns/infer_stat.completed_request_count)) << std::endl;

    return 0;
}