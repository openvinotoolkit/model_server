/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;

using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

using tensorflow::serving::GetModelMetadataRequest;
using tensorflow::serving::GetModelMetadataResponse;
using tensorflow::serving::SignatureDefMap;

using proto_signature_map_t = google::protobuf::Map<std::string, tensorflow::TensorInfo>;
using proto_tensor_map_t = google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>;

struct AsyncClientCall {
    PredictResponse reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> response_reader;
    tensorflow::int64 id;
};

struct Configuration {
    tensorflow::string address = "localhost";
    tensorflow::string port = "9000";
    tensorflow::string modelName = "resnet";
    tensorflow::int64 iterations = 10;
    tensorflow::int64 producers = 1;
    tensorflow::int64 consumers = 8;
    tensorflow::int64 max_parallel_requests = 100;

    bool validate() const {
        if (iterations <= 0)
            return false;
        if (producers <= 0 || consumers <= 0)
            return false;
        if (max_parallel_requests < 0)
            return false;
        return true;
    }
};

void prepareSyntheticData(tensorflow::TensorInfo& info, tensorflow::TensorProto& tensor) {
    tensor.set_dtype(info.dtype());
    *tensor.mutable_tensor_shape() = info.tensor_shape();
    size_t expectedValueCount = 1;
    for (int i = 0; i < info.tensor_shape().dim_size(); i++) {
        expectedValueCount *= info.tensor_shape().dim(i).size();
    }
    expectedValueCount *= tensorflow::DataTypeSize(info.dtype());
    *tensor.mutable_tensor_content() = std::string(expectedValueCount, '1');
}

template <typename T>
class ResourceGuard {
    T* ptr;

public:
    ResourceGuard(T* ptr) :
        ptr(ptr) {}
    ~ResourceGuard() { delete ptr; }
};

class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;
    CompletionQueue cq_;

public:
    ServingClient(std::shared_ptr<Channel> channel, const Configuration& config) :
        stub_(PredictionService::NewStub(channel)) {
        this->config = config;
    }

    bool prepareRequest() {
        this->predictRequest.mutable_model_spec()->set_name(this->config.modelName);
        this->predictRequest.mutable_model_spec()->set_signature_name("serving_default");
        proto_tensor_map_t& inputs = *this->predictRequest.mutable_inputs();

        return this->prepareBatchedInputs(inputs);
    }

    // Pre-processing function for synthetic data.
    // gRPC request proto is generated with synthetic data with shape/precision matching endpoint metadata.
    bool prepareBatchedInputs(proto_tensor_map_t& inputs) {
        proto_signature_map_t inputsMetadata;
        if (!getEndpointInputsMetadata(config, inputsMetadata)) {
            return false;
        }

        bool isMetadataValid = true;
        std::cout << "Synthetic inputs:" << std::endl;
        for (auto& [name, input] : inputsMetadata) {
            std::cout << "\t" << input.name() << ": (";
            for (size_t i = 0; i < input.tensor_shape().dim_size(); i++) {
                if (input.tensor_shape().dim(i).size() <= 0) {
                    isMetadataValid = false;
                }
                std::cout << input.tensor_shape().dim(i).size();
                if (i < input.tensor_shape().dim_size() - 1) {
                    std::cout << ",";
                }
            }
            std::cout << "); " << tensorflow::DataType_Name(input.dtype()) << std::endl;

            auto& inputTensor = inputs[input.name()];
            prepareSyntheticData(input, inputTensor);
        }

        if (!isMetadataValid) {
            std::cout << "[ERROR] Input metadata cannot contain negative shape" << std::endl;
            return false;
        }
        return true;
    }

    bool schedulePredict(tensorflow::int64 iteration) {
        PredictResponse response;
        ClientContext context;

        auto* call = new AsyncClientCall;
        call->id = iteration + 1;

        call->response_reader = stub_->PrepareAsyncPredict(&call->context, this->predictRequest, &this->cq_);
        call->response_reader->StartCall();
        call->response_reader->Finish(&call->reply, &call->status, (void*)call);

        return true;
    }

    void asyncCompleteRpc() {
        void* got_tag;
        bool ok = false;

        while (cq_.Next(&got_tag, &ok)) {
            if (++this->finishedIterations >= this->config.iterations * this->config.producers) {
                cq_.Shutdown();
            }
            cv.notify_one();
            auto* call = static_cast<AsyncClientCall*>(got_tag);
            ResourceGuard guard(call);

            auto& response = call->reply;

            if (!ok) {
                std::cerr << "Request is not ok" << std::endl;
                failedIterations++;
                continue;
            }

            if (!call->status.ok()) {
                std::cout << "gRPC call return code: " << call->status.error_code() << ": "
                          << call->status.error_message() << std::endl;
                failedIterations++;
                continue;
            }
        }
    }

    tensorflow::int64 getFailedIterations() const {
        return this->failedIterations;
    }

    size_t getRequestBatchSize() {
        return this->predictRequest.mutable_inputs()->begin()->second.mutable_tensor_shape()->dim(0).size();
    }

    void scheduler() {
        for (tensorflow::int64 i = 0; i < config.iterations; i++) {
            if (config.max_parallel_requests > 0 && i - finishedIterations + 1 > config.max_parallel_requests) {
                std::unique_lock<std::mutex> lck(cv_m);
                cv.wait(lck);
            }
            if (!this->schedulePredict(i)) {
                return;
            }
        }
    }

    bool getEndpointInputsMetadata(const Configuration& config, proto_signature_map_t& inputsMetadata) {
        GetModelMetadataRequest request;
        GetModelMetadataResponse response;
        ClientContext context;
        request.mutable_metadata_field()->Add("signature_def");
        *request.mutable_model_spec()->mutable_name() = config.modelName;
        auto status = stub_->GetModelMetadata(&context, request, &response);

        if (!status.ok()) {
            std::cout << "gRPC call return code: " << status.error_code() << ": "
                      << status.error_message() << std::endl;
            return false;
        }

        auto it = response.mutable_metadata()->find("signature_def");
        if (it == response.metadata().end()) {
            std::cout << "error reading metadata response" << std::endl;
            return false;
        }
        SignatureDefMap def;
        def.ParseFromString(*it->second.mutable_value());
        inputsMetadata = *(*def.mutable_signature_def())["serving_default"].mutable_inputs();
        return true;
    }

    static void start(const tensorflow::string& address, const Configuration& config) {
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(-1);
        ServingClient client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args), config);
        if (!client.prepareRequest()) {
            return;
        }
        std::vector<std::thread> threads;
        std::cout << "\nRunning the workload..." << std::endl;
        auto begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.consumers; i++) {
            threads.emplace_back(std::thread(&ServingClient::asyncCompleteRpc, &client));
        }
        for (int i = 0; i < config.producers; i++) {
            threads.emplace_back(std::thread(&ServingClient::scheduler, &client));
        }
        for (auto& t : threads) {
            t.join();
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin);
        auto totalTime = (duration.count() / 1000);
        float avgFps = (1000 / ((float)totalTime / (float)(config.iterations * config.producers * client.getRequestBatchSize())));

        std::cout << "========================\n        Summary\n========================" << std::endl;
        std::cout << "Total time: " << totalTime << "ms" << std::endl;
        std::cout << "Total iterations: " << config.iterations * config.producers << std::endl;
        std::cout << "Producer threads: " << config.producers << std::endl;
        std::cout << "Consumer threads: " << config.consumers << std::endl;
        std::cout << "Max parallel requests: " << config.max_parallel_requests << std::endl;
        std::cout << "Avg FPS: " << avgFps << std::endl;
        if (client.getFailedIterations() > 0) {
            std::cout << "\n[WARNING] " << client.getFailedIterations() << " requests have failed." << std::endl;
        }
    }

private:
    Configuration config;
    std::atomic<tensorflow::int64> finishedIterations = 0;
    std::atomic<tensorflow::int64> failedIterations = 0;
    std::condition_variable cv;
    std::mutex cv_m;

    PredictRequest predictRequest;
};

int main(int argc, char** argv) {
    Configuration config;
    std::vector<tensorflow::Flag> flagList = {
        tensorflow::Flag("grpc_address", &config.address, "url to grpc service"),
        tensorflow::Flag("grpc_port", &config.port, "port to grpc service"),
        tensorflow::Flag("model_name", &config.modelName, "model name to request"),
        tensorflow::Flag("iterations", &config.iterations, "number of requests to be send by each producer thread"),
        tensorflow::Flag("producers", &config.producers, "number of threads asynchronously scheduling prediction"),
        tensorflow::Flag("consumers", &config.consumers, "number of threads receiving responses"),
        tensorflow::Flag("max_parallel_requests", &config.max_parallel_requests, "maximum number of parallel inference requests; 0=no limit")};

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
    const bool result = tensorflow::Flags::Parse(&argc, argv, flagList);

    if (!result || !config.validate()) {
        std::cout << usage;
        return -1;
    }

    const tensorflow::string host = config.address + ":" + config.port;

    std::cout
        << "Address: " << host << std::endl
        << "Model name: " << config.modelName << std::endl;

    ServingClient::start(host, config);
    return 0;
}
