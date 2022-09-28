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
#include <fstream>
#include <string>

#include <cxxopts.hpp>

#include "grpc_client.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                              \
    {                                                                    \
        tc::Error err = (X);                                             \
        if (!err.IsOk()) {                                               \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }

void Load(std::string fileName, uint8_t* input){
    std::vector<uint8_t> data;


    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "error: unable to decode image " << fileName << std::endl;
        exit(1);
    }
    img.convertTo(img, CV_32FC3);
    std::cout<<img.size();
    cv::resize(img, img, cv::Size(224,224));
    memcpy(input, img.data, img.total()*sizeof(float));
}

int main(int argc, char** argv) {
    cxxopts::Options opt("grpc_infer_resnet", "Sends requests via KServe gRPC API.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("images_list", "Path to a file with a list of labeled images", cxxopts::value<std::string>())
    ("grpc_address", "Specify url to grpc service. ", cxxopts::value<std::string>()->default_value("localhost"))
    ("grpc_port", "Specify port to grpc service. ", cxxopts::value<std::string>()->default_value("9000"))
    ("input_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("b"))
    ("output_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("a"))
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("dummy"))
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

    std::vector<uint8_t> input_data;
    std::string img;
    int label;
    std::vector<std::string> imgs;
    std::vector<int> labels;
    std::ifstream images(args["images_list"].as<std::string>());
    while(images>>img>>label){
        imgs.push_back(img);
        labels.push_back(label);
    }
    
    input_data.resize(224*224 * 4*3);


    std::vector<int64_t> shape{1, 224, 224, 3};

    // Initialize the inputs with the data.
    tc::InferInput* input;

    FAIL_IF_ERR(
        tc::InferInput::Create(&input, input_name, shape, "FP32"),
        "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);


    // Generate the outputs to be requested.
    tc::InferRequestedOutput* output;

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, output_name),
        "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);

    tc::InferOptions options(model_name);
    if (args.count("model_version"))
        options.model_version_ = args["model_version"].as<std::string>();
    options.client_timeout_ = args["timeout"].as<int>();

    std::vector<tc::InferInput*> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

    std::vector<tc::InferResult*> results;
    results.resize(10);
    for(int i = 0; i < imgs.size();i++){
        Load(imgs[i], input_data.data());
        FAIL_IF_ERR(
            input_ptr->AppendRaw(input_data),
            "unable to set data for input");
        FAIL_IF_ERR(
            client->Infer(&(results[i]), options, inputs, outputs),
            "unable to run model");
        //input_ptr.reset(input);
        input->Reset();
    }

    for(int i = 0; i < imgs.size(); i++){
        std::shared_ptr<tc::InferResult> results_ptr;
        results_ptr.reset(results[i]);
        // Get pointers to the result returned...
        float* output_data;
        size_t output_byte_size;
        FAIL_IF_ERR(
            results_ptr->RawData(
                output_name, (const uint8_t**)&output_data, &output_byte_size),
            "unable to get result data for output");

        std::cout << "Index of max element: "
        << std::distance(output_data, std::max_element(output_data, output_data + 1001))
        << " <=>" << labels[i]
        << std::endl;
    }


    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "Completed request count "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "Cumulative total request time "
              << double(infer_stat.cumulative_total_request_time_ns)/1000000 << " ms" << std::endl;
    std::cout << "Cumulative send time "
              << double(infer_stat.cumulative_send_time_ns)/1000000 << " ms" << std::endl;
    std::cout << "Cumulative receive time "
              << double(infer_stat.cumulative_receive_time_ns)/1000000 << " ms" << std::endl;

    return 0;
}