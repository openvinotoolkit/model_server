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


int main(int argc, char** argv) {
    cxxopts::Options opt("http_ready", "Sends requests via KServe rest API to get model metadata.");

    // clang-format off
    opt.add_options()
    ("h,help", "Show this help message and exit")
    ("http_address", "Specify url to grpc service. ", cxxopts::value<std::string>()->default_value("localhost"), "HTTP_ADDRESS")
    ("http_port", "Specify port to grpc service. ", cxxopts::value<std::string>()->default_value("8000"), "PORT")
    ("model_name", "Define model name, must be same as is in service. ", cxxopts::value<std::string>()->default_value("dummy"), "MODEL_NAME")
    ("model_version", "Define model version.", cxxopts::value<std::string>()->default_value(""), "MODEL_VERSION")
    ("timeout", "Request timeout.", cxxopts::value<int>()->default_value("0"), "TIMEOUT")
    ;
    // clang-format on

    auto args = opt.parse(argc, argv);

    if (args.count("help")) {
        std::cout << opt.help() << std::endl;
        exit(0);
    }


    std::string url(args["http_address"].as<std::string>() + ":" + args["http_port"].as<std::string>());

    // Create a InferenceServerHttpClient instance to communicate with the
    // server using http protocol.
    std::unique_ptr<tc::InferenceServerHttpClient> client;

    FAIL_IF_ERR(
        tc::InferenceServerHttpClient::Create(&client, url),
        err);

    std::string model_metadata;
    FAIL_IF_ERR(
        client->ModelMetadata(&model_metadata, args["model_name"].as<std::string>(), args["model_version"].as<std::string>()),
        "unable to get model metadata");
    std::cout<<model_metadata<< std::endl;


    return 0;
}