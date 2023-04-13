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
// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

#include <cxxopts.hpp>

#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                              \
    {                                                                    \
        tc::Error err = (X);                                             \
        if (!err.IsOk()) {                                               \
            std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
            return 1;                                                    \
        }                                                                \
    }

std::string load(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary);
    file.unsetf(std::ios::skipws);
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::ostringstream oss;
    oss << file.rdbuf();

    return oss.str();
}


int main(int argc, char** argv) {
    cxxopts::Options opt("grpc_async_infer_resnet", "Sends requests via KServe gRPC API.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("images_list", "Path to a file with a list of labeled images. ", cxxopts::value<std::string>(), "IMAGES")
    ("labels_list", "Path to a file with a list of labels. ", cxxopts::value<std::string>(), "LABELS")
    ("grpc_address", "Specify url to grpc service. ", cxxopts::value<std::string>()->default_value("localhost"), "GRPC_ADDRESS")
    ("grpc_port", "Specify port to grpc service. ", cxxopts::value<std::string>()->default_value("9000"), "PORT")
    ("input_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("0"), "INPUT_NAME")
    ("output_name", "Specify input tensor name. ", cxxopts::value<std::string>()->default_value("1463"), "OUTPUT_NAME")
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("resnet"), "MODEL_NAME")
    ("model_version", "Define model version.", cxxopts::value<std::string>(), "MODEL_VERSION")
    ("timeout", "Request timeout.", cxxopts::value<int>()->default_value("0"), "TIMEOUT")
    ;
    // clang-format on

    cxxopts::ParseResult args;
    try {
        args = opt.parse(argc, argv);
    }
    catch(cxxopts::option_not_exists_exception e) {
        std::cerr << "error: cli options parsing failed - " << e.what();
        return 1;
    }
    
    if (args.count("help")) {
        std::cout << opt.help() << std::endl;
        return 0;
    }
    if (!args.count("images_list")) {
        std::cerr << "error: option \"images_list\" has no value\n";
        return 1;
    }
    if (!args.count("labels_list")) {
        std::cerr << "error: option \"labels_list\" has no value\n";
        return 1;
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
    int label = -1;
    std::vector<std::string> imgs;
    std::vector<int> labels;
    std::ifstream images(args["images_list"].as<std::string>());
    while (images >> img >> label) {
        imgs.push_back(img);
        labels.push_back(label);
    }

    if(imgs.size() == 0) {
        std::cerr << "error: Path to image_list file is invalid or the file does not contain valid image paths. \n";
        return 1;
    }

    std::vector<int64_t> shape{1};

    std::vector<tc::InferInput*> inputs;
    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    for (int i = 0; i < imgs.size(); i++) {
        tc::InferInput* input;
        inputs.push_back(input);

        FAIL_IF_ERR(
            tc::InferInput::Create(&input, input_name, shape, "BYTES"),
            "unable to get input");
        std::shared_ptr<tc::InferInput> input_ptr;
        input_ptr.reset(input);
        input_ptrs.push_back(input_ptr);
    }

    tc::InferOptions options(model_name);
    if (args.count("model_version"))
        options.model_version_ = args["model_version"].as<std::string>();
    try {
        options.client_timeout_ = args["timeout"].as<int>();
    } catch (cxxopts::argument_incorrect_type e) {
        std::cerr << "The provided argument is of a wrong type" << std::endl;
        return 1;
    }

    std::vector<std::string> classes;
    std::ifstream lb_f(args["labels_list"].as<std::string>());
    std::string tmp;
    while (std::getline(lb_f, tmp)) {
        classes.push_back(tmp);
    }

    std::vector<std::string> input_data;
    input_data.reserve(imgs.size());
    for (int i = 0; i < imgs.size(); i++) {
        try {
            input_data.push_back(load(imgs[i]));
        }
        catch(const std::bad_alloc&) {
            std::cerr<< "error: Loading image:" + imgs[i] + " failed. \n";
            return 1;
        }
    }

    int acc = 0;
    std::mutex mtx;
    std::condition_variable cv;
    int completedRequestCount = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < imgs.size(); i++) {
        std::vector<tc::InferInput*> inputs = {input_ptrs[i].get()};
        FAIL_IF_ERR(
            input_ptrs[i]->AppendFromString({input_data[i]}),
            "unable to set data for input");
        client->AsyncInfer(
            [&, i](tc::InferResult* result) -> int {
                {
                    std::shared_ptr<tc::InferResult> result_ptr;
                    result_ptr.reset(result);
                    std::lock_guard<std::mutex> lk(mtx);
                    completedRequestCount++;
                    // Get pointers to the result returned...
                    float* output_data;
                    size_t output_byte_size;
                    FAIL_IF_ERR(
                        result_ptr->RequestStatus(),
                        "unable to get result");
                    FAIL_IF_ERR(
                        result_ptr->RawData(
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
                cv.notify_all();
                return 0;
            },
            options, inputs);
    }
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]() {
            if (completedRequestCount >= imgs.size()) {
                return true;
            } else {
                return false;
            }
        });
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Accuracy " << float(acc) / imgs.size() * 100 << "%\n";

    tc::InferStat infer_stat;
    client->ClientInferStat(&infer_stat);
    std::cout << "======Client Statistics======" << std::endl;
    std::cout << "Number of requests: "
              << infer_stat.completed_request_count << std::endl;
    std::cout << "Total processing time: "
              << duration.count() << " ms" << std::endl;
    std::cout << "Latency: "
              << double(infer_stat.cumulative_total_request_time_ns / infer_stat.completed_request_count) / 1.0e+6 << " ms" << std::endl;
    std::cout << "Requests per second: "
              << double(1.0e+9 / (infer_stat.cumulative_total_request_time_ns / infer_stat.completed_request_count)) << std::endl;

    return 0;
}