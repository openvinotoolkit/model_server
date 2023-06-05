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

#include "common.hpp"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "opencv2/opencv.hpp"

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

template <typename T>
struct AsyncClientCall {
    PredictResponse reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> response_reader;
    std::vector<T> selectedEntries;
    tensorflow::int64 id;
};

struct Configuration {
    tensorflow::string address = "localhost";
    tensorflow::string port = "9000";
    tensorflow::string modelName = "resnet";
    tensorflow::string inputName = "0";
    tensorflow::string outputName = "1463";
    tensorflow::int64 iterations = 10;
    tensorflow::int64 batchSize = 1;
    tensorflow::string imagesListPath = "input_images.txt";
    tensorflow::string layout = "nchw";
    tensorflow::int64 producers = 1;
    tensorflow::int64 consumers = 8;
    tensorflow::int64 max_parallel_requests = 100;
    tensorflow::int64 benchmark_mode = 0;
    tensorflow::int64 width = 224;
    tensorflow::int64 height = 224;

    bool validate() const {
        if (imagesListPath.empty())
            return false;
        if (batchSize <= 0)
            return false;
        if (iterations <= 0)
            return false;
        if (producers <= 0 || consumers <= 0)
            return false;
        if (max_parallel_requests < 0)
            return false;
        if (benchmark_mode < 0 || benchmark_mode > 1)
            return false;
        if (layout != "binary" && layout != "nchw" && layout != "nhwc")
            return false;
        if (width <= 0 || height <= 0)
            return false;
        return true;
    }
};

template <typename T>
std::vector<T> selectEntries(const std::vector<T>& entries, tensorflow::int64 batchSize, tensorflow::int64 iteration) {
    size_t startPoint = (iteration * batchSize) % entries.size();
    if (batchSize > entries.size()) {
        std::vector<T> selected;
        while (selected.size() < batchSize) {
            auto remainingBatches = batchSize - selected.size();
            if (entries.size() - startPoint >= remainingBatches) {
                selected.insert(selected.end(), entries.begin() + startPoint, entries.begin() + startPoint + remainingBatches);
                break;
            }
            selected.insert(selected.end(), entries.begin() + startPoint, entries.end());
            startPoint = 0;
        }
        return selected;
    } else {
        if (startPoint + batchSize > entries.size()) {
            std::vector<T> selected;
            selected.insert(selected.end(), entries.begin() + startPoint, entries.end());
            selected.insert(selected.end(), entries.begin(), entries.begin() + (batchSize - (entries.size() - startPoint)));
            return selected;
        } else {
            return std::vector<T>(entries.begin() + startPoint, entries.begin() + startPoint + batchSize);
        }
    }
}

std::vector<tensorflow::int64> argmax(const tensorflow::Tensor& tensor) {
    const auto& shape = tensor.shape();
    assert(shape.dims() == 2);
    size_t batchSize = shape.dim_size(0);
    size_t elements = shape.dim_size(1);
    std::vector<tensorflow::int64> labels;
    labels.reserve(batchSize);
    for (size_t j = 0; j < batchSize; j++) {
        float topConfidence = 0;
        tensorflow::int64 topLabel = -1;
        for (size_t i = 0; i < elements; i++) {
            float confidence = ((float*)tensor.data())[j * elements + i];
            if (topLabel == -1 || topConfidence < confidence) {
                topLabel = i;
                topConfidence = confidence;
            }
        }
        labels.push_back(topLabel);
    }
    return labels;
}

template <typename T>
class ResourceGuard {
    T* ptr;

public:
    ResourceGuard(T* ptr) :
        ptr(ptr) {}
    ~ResourceGuard() { delete ptr; }
};

template <typename T>
class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;
    CompletionQueue cq_;

public:
    ServingClient(std::shared_ptr<Channel> channel, const Configuration& config, const std::vector<T>& entries = {}) :
        stub_(PredictionService::NewStub(channel)) {
        this->config = config;
        this->entries = entries;
    }

    bool prepareRequest() {
        this->predictRequest.mutable_model_spec()->set_name(this->config.modelName);
        this->predictRequest.mutable_model_spec()->set_signature_name("serving_default");
        proto_tensor_map_t& inputs = *this->predictRequest.mutable_inputs();

        tensorflow::int64 iteration = 1;
        this->entries = selectEntries(this->entries, this->config.batchSize, iteration);
        return this->prepareBatchedInputs(inputs, this->entries);
    }

    // Pre-processing function for binary images.
    // Images loaded from disk are packed into gRPC request proto.
    bool prepareBatchedInputs(proto_tensor_map_t& inputs, const std::vector<BinaryData>& entries) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        for (const auto& entry : entries) {
            proto.add_string_val(entry.imageData.get(), entry.fileSize);
        }
        proto.mutable_tensor_shape()->add_dim()->set_size(entries.size());
        inputs[this->config.inputName] = proto;
        return true;
    }

    // Pre-processing function for images in array format.
    // Images loaded from disk are packed into tensor_content in plain array format (using OpenCV) either in NCHW or NHWC layout.
    bool prepareBatchedInputs(proto_tensor_map_t& inputs, const std::vector<CvMatData>& entries) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);

        std::string* content = proto.mutable_tensor_content();

        // We are already ensured that each cv::Mat contains 224x224 data.
        size_t byteSize = entries[0].image.total() * entries[0].image.elemSize();

        content->resize(byteSize * entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            std::memcpy(content->data() + i * byteSize, entries[i].image.data, byteSize);
        }

        proto.mutable_tensor_shape()->add_dim()->set_size(entries.size());
        if (entries[0].layout == "nchw") {
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.channels());
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.cols);
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.rows);
        } else {
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.cols);
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.rows);
            proto.mutable_tensor_shape()->add_dim()->set_size(entries[0].image.channels());
        }
        inputs[this->config.inputName] = proto;
        return true;
    }

    // Post-processing function for classification.
    // Most probable label is selected from the output.
    bool interpretOutputs(proto_tensor_map_t& outputs, std::vector<tensorflow::int64>& predictedLabels) {
        auto it = outputs.find(this->config.outputName);
        if (it == outputs.end()) {
            std::cout << "cannot find output " << this->config.outputName << std::endl;
            return false;
        }
        tensorflow::TensorProto& resultTensorProto = it->second;
        if (resultTensorProto.dtype() != tensorflow::DataType::DT_FLOAT) {
            std::cout << "result has non-float datatype" << std::endl;
            return false;
        }
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(resultTensorProto);
        if (!converted) {
            std::cout << "the result tensor[" << it->first << "] convert failed." << std::endl;
            return false;
        }
        predictedLabels = argmax(tensor);
        return true;
    }

    void reportPredictionCorrectness(tensorflow::serving::PredictResponse& response, const std::vector<T>& selectedEntries) {
        std::vector<tensorflow::int64> predictedLabels;
        if (!this->interpretOutputs(*response.mutable_outputs(), predictedLabels)) {
            std::cout << "error interpreting outputs" << std::endl;
            this->failedIterations++;
            return;
        }

        size_t numberOfCorrectLabels = 0;
        for (size_t i = 0; i < predictedLabels.size(); i++) {
            if (predictedLabels[i] == selectedEntries[i].expectedLabel) {
                numberOfCorrectLabels++;
            } else {
                std::cout << "incorrect prediction; expected " << selectedEntries[i].expectedLabel << ", got " << predictedLabels[i] << std::endl;
            }
        }
        this->numberOfCorrectLabels += numberOfCorrectLabels;
    }

    bool schedulePredict(tensorflow::int64 iteration) {
        PredictResponse response;
        ClientContext context;

        auto* call = new AsyncClientCall<T>;
        call->id = iteration + 1;

        if (this->config.benchmark_mode == 0) {
            // Pre-processing step.
            // Packing image into gRPC message.
            PredictRequest request;
            request.mutable_model_spec()->set_name(this->config.modelName);
            request.mutable_model_spec()->set_signature_name("serving_default");
            google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *request.mutable_inputs();
            call->selectedEntries = selectEntries(this->entries, this->config.batchSize, iteration);
            this->prepareBatchedInputs(inputs, call->selectedEntries);
            call->response_reader = stub_->PrepareAsyncPredict(&call->context, request, &this->cq_);
            call->response_reader->StartCall();
            call->response_reader->Finish(&call->reply, &call->status, (void*)call);
        } else {
            // No pre-processing step.
            // Re-use previously prepared gRPC message.
            call->response_reader = stub_->PrepareAsyncPredict(&call->context, this->predictRequest, &this->cq_);
            call->response_reader->StartCall();
            call->response_reader->Finish(&call->reply, &call->status, (void*)call);
        }

        if (config.benchmark_mode == 0) {
            std::cout << "Scheduled request no. " << call->id << std::endl;
        }
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
            auto* call = static_cast<AsyncClientCall<T>*>(got_tag);
            ResourceGuard guard(call);

            if (config.benchmark_mode == 0) {
                std::cout << "Received response no. " << call->id << std::endl;
            }
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

            // Postprocessing
            if (this->config.benchmark_mode == 0) {
                this->reportPredictionCorrectness(response, call->selectedEntries);
            }
        }
    }

    size_t getNumberOfCorrectLabels() const {
        return this->numberOfCorrectLabels;
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

    static void start(const tensorflow::string& address, const Configuration& config, const std::vector<T>& entries = {}) {
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(-1);
        ServingClient<T> client(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args), config, entries);
        if (config.benchmark_mode == 1) {
            if (!client.prepareRequest()) {
                return;
            }
        }
        std::vector<std::thread> threads;
        std::cout << "\nRunning the workload..." << std::endl;
        auto begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.consumers; i++) {
            threads.emplace_back(std::thread(&ServingClient<T>::asyncCompleteRpc, &client));
        }
        for (int i = 0; i < config.producers; i++) {
            threads.emplace_back(std::thread(&ServingClient<T>::scheduler, &client));
        }
        for (auto& t : threads) {
            t.join();
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin);
        auto totalTime = (duration.count() / 1000);
        float accuracy = (client.getNumberOfCorrectLabels() * 100.0f) / (config.iterations * config.producers * config.batchSize);
        float avgFps = (1000 / ((float)totalTime / (float)(config.iterations * config.producers * config.batchSize)));

        std::cout << "========================\n        Summary\n========================" << std::endl;
        if (config.benchmark_mode == 0) {
            std::cout << "Benchmark mode: False\nAccuracy: " << accuracy << "%" << std::endl;
        } else {
            std::cout << "Benchmark mode: True\nAccuracy: N/A" << std::endl;
        }
        std::cout << "Total time: " << totalTime << "ms" << std::endl;
        std::cout << "Total iterations: " << config.iterations * config.producers << std::endl;
        std::cout << "Layout: " << config.layout << std::endl;
        std::cout << "Batch size: " << config.batchSize << std::endl;
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
    std::vector<T> entries;
    std::atomic<size_t> numberOfCorrectLabels = 0;
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
        tensorflow::Flag("input_name", &config.inputName, "input tensor name with image"),
        tensorflow::Flag("output_name", &config.outputName, "output tensor name with classification result"),
        tensorflow::Flag("iterations", &config.iterations, "number of requests to be send by each producer thread"),
        tensorflow::Flag("batch_size", &config.batchSize, "batch size of each iteration"),
        tensorflow::Flag("images_list", &config.imagesListPath, "path to a file with a list of labeled images"),
        tensorflow::Flag("layout", &config.layout, "binary, nhwc or nchw"),
        tensorflow::Flag("producers", &config.producers, "number of threads asynchronously scheduling prediction"),
        tensorflow::Flag("consumers", &config.consumers, "number of threads receiving responses"),
        tensorflow::Flag("max_parallel_requests", &config.max_parallel_requests, "maximum number of parallel inference requests; 0=no limit"),
        tensorflow::Flag("benchmark_mode", &config.benchmark_mode, "when enabled, there is no pre/post-processing step"),
        tensorflow::Flag("width", &config.width, "input images width will be resized to this value; not applied to binary input"),
        tensorflow::Flag("height", &config.height, "input images height will be resized to this value; not applied to binary input")};

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
    const bool result = tensorflow::Flags::Parse(&argc, argv, flagList);

    if (!result || !config.validate()) {
        std::cout << usage;
        return -1;
    }

    std::vector<Entry> entries;
    if (!readImagesList(config.imagesListPath, entries)) {
        std::cout << "Error parsing images_list" << std::endl;
        return -1;
    }

    if (entries.empty()) {
        std::cout << "Empty images_list" << std::endl;
        return -1;
    }

    const tensorflow::string host = config.address + ":" + config.port;

    std::cout
        << "Address: " << host << std::endl
        << "Model name: " << config.modelName << std::endl;

    std::cout << "Images list path: " << config.imagesListPath << std::endl;

    if (config.layout == "binary") {
        std::vector<BinaryData> images;
        if (!readImagesBinary(entries, images)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient<BinaryData>::start(host, config, images);
    } else {
        std::vector<CvMatData> images;
        if (!readImagesCvMat(entries, images, config.layout, config.width, config.height)) {
            std::cout << "Error reading opencv images" << std::endl;
            return -1;
        }
        ServingClient<CvMatData>::start(host, config, images);
    }
    return 0;
}
