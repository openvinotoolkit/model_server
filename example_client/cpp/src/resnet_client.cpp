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

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

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

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

struct AsyncClientCall {
    PredictResponse reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> response_reader;
};

struct Entry {
    tensorflow::string imagePath;
    tensorflow::int64 expectedLabel;
};

struct BinaryData {
    std::shared_ptr<char[]> imageData;
    std::streampos fileSize;
    tensorflow::int64 expectedLabel;
};

struct CvMatData {
    cv::Mat image;
    tensorflow::int64 expectedLabel;
    tensorflow::string layout;
};

struct Configuration  {
    tensorflow::string address = "localhost";
    tensorflow::string port = "9000";
    tensorflow::string modelName = "resnet";
    tensorflow::string inputName = "0";
    tensorflow::string outputName = "1463";
    tensorflow::int64 iterations = 1;
    tensorflow::int64 batchSize = 1;
    tensorflow::string imagesListPath = "";
    tensorflow::string layout = "binary";

    bool validate() const {
        if (imagesListPath.empty())
            return false;
        if (batchSize < 0)
            return false;
        if (iterations < 0)
            return false;
        if (layout != "binary" && layout != "nchw" && layout != "nhwc")
            return false;
        return true;
    }
};

template <typename T>
std::vector<T> reorderVectorToNchw(const T* nhwcVector, int rows, int cols, int channels) {
    std::vector<T> nchwVector(rows * cols * channels);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                nchwVector[c * (rows * cols) + y * cols + x] = reinterpret_cast<const T*>(nhwcVector)[y * channels * cols + x * channels + c];
            }
        }
    }
    return nchwVector;
}

const cv::Mat reorderMatToNchw(cv::Mat* mat) {
    uint64_t channels = mat->channels();
    uint64_t rows = mat->rows;
    uint64_t cols = mat->cols;
    auto nchwVector = reorderVectorToNchw<float>((float*)mat->data, rows, cols, channels);

    cv::Mat image(rows, cols, CV_32FC3);
    std::memcpy(image.data, nchwVector.data(), nchwVector.size() * sizeof(float));
    return image;
}

bool readImagesList(const tensorflow::string& path, std::vector<Entry>& entries) {
    entries.clear();
    std::ifstream infile(path);
    tensorflow::string image = "";
    tensorflow::int64 label = 0;

    if (!infile.is_open()) {
        std::cout << "Failed to open " << path << std::endl;
        return false;
    }

    while (infile >> image >> label) {
        entries.emplace_back(Entry{image, label});
    }

    return true;
}

bool readImagesBinary(const std::vector<Entry>& entriesIn, std::vector<BinaryData>& entriesOut) {
    entriesOut.clear();

    for (const auto& entry : entriesIn) {
        std::ifstream imageFile(entry.imagePath, std::ios::binary);
        if (!imageFile.is_open()) {
            std::cout << "Failed to open " << entry.imagePath << std::endl;
            return false;
        }

        std::filebuf* pbuf = imageFile.rdbuf();
        auto fileSize = pbuf->pubseekoff(0, std::ios::end, std::ios::in);

        auto image = std::unique_ptr<char[]>(new char[fileSize]());

        pbuf->pubseekpos(0, std::ios::in);
        pbuf->sgetn(image.get(), fileSize);
        imageFile.close();

        entriesOut.emplace_back(BinaryData{std::move(image), fileSize, entry.expectedLabel});
    }

    return true;
}

bool readImagesCvMat(const std::vector<Entry>& entriesIn, std::vector<CvMatData>& entriesOut, const tensorflow::string& layout) {
    entriesOut.clear();

    for (const auto& entryIn : entriesIn) {
        CvMatData entryOut;
        entryOut.layout = layout;
        entryOut.expectedLabel = entryIn.expectedLabel;
        try {
            entryOut.image = cv::imread(entryIn.imagePath);
            if (entryOut.image.data == nullptr) {
                return false;
            }
        } catch (cv::Exception& ex) {
            return false;
        }
        entryOut.image.convertTo(entryOut.image, CV_32F);
        cv::resize(entryOut.image, entryOut.image, cv::Size(224, 224));
        if (layout == "nchw") {
            entryOut.image = reorderMatToNchw(&entryOut.image);
        }
        entriesOut.emplace_back(entryOut);
    }

    return true;
}

template <typename T>
std::vector<T> selectEntries(const std::vector<T>& entries, tensorflow::int64 batchSize) {
    if (batchSize > entries.size()) {
        std::vector<T> selected;
        while (batchSize > entries.size()) {
            selected.insert(selected.end(), entries.begin(), entries.end());
            batchSize -= entries.size();
        }
        if (batchSize > 0) {
            selected.insert(selected.end(), entries.begin(), entries.begin() + batchSize);
        }
        return selected;
    } else {
        return std::vector<T>(entries.begin(), entries.begin() + batchSize);
    }
}

std::vector<tensorflow::int64> argmax(const tensorflow::Tensor& tensor) {
    const auto& shape = tensor.shape();
    assert(shape.dims() == 2);
    size_t batchSize = shape.dim_size(0);
    size_t elements = shape.dim_size(1);
    std::vector<tensorflow::int64> labels;
    for (size_t j = 0 ; j < batchSize; j++) {
        float topConfidence = 0;
        tensorflow::int64 topLabel = -1;
        for (tensorflow::int64 i = 0; i < elements; i++) {
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
class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;
    CompletionQueue cq_;

public:
    ServingClient(std::shared_ptr<Channel> channel, const std::vector<T>& entries, const Configuration& config) :
        stub_(PredictionService::NewStub(channel)) {
        this->config = config;
        this->entries = selectEntries(entries, this->config.batchSize);  // TODO: Use other images than 0th for bs=1.
    }

    // Pre-processing function for binary images.
    // Images loaded from disk are packed into gRPC request proto.
    // TODO: To remove
    static bool prepareInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const BinaryData& entry, const tensorflow::string& inputName) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        proto.add_string_val(entry.imageData.get(), entry.fileSize);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        inputs[inputName] = proto;
        return true;
    }

    static bool prepareBatchedInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const std::vector<BinaryData>& entries, const tensorflow::string& inputName) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        for (const auto& entry : entries) {
            proto.add_string_val(entry.imageData.get(), entry.fileSize);
        }
        proto.mutable_tensor_shape()->add_dim()->set_size(entries.size());
        inputs[inputName] = proto;
        return true;
    }

    // Pre-processing function for images in array format.
    // Images loaded from disk are packed into tensor_content in plain array format (using OpenCV) either in NCHW or NHWC layout.
    // TODO: To remove
    static bool prepareInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const CvMatData& entry, const tensorflow::string& inputName) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_FLOAT);
        size_t byteSize = entry.image.total() * entry.image.elemSize();
        proto.mutable_tensor_content()->assign((char*)entry.image.data, byteSize);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        if (entry.layout == "nchw") {
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.channels());
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.cols);
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.rows);
        } else {
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.cols);
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.rows);
            proto.mutable_tensor_shape()->add_dim()->set_size(entry.image.channels());
        }
        inputs[inputName] = proto;
        return true;
    }

    static bool prepareBatchedInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const std::vector<CvMatData>& entries, const tensorflow::string& inputName) {
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
        inputs[inputName] = proto;
        return true;
    }

    // Post-processing function for resnet classification.
    // Most probable label is selected from the output.
    static bool interpretOutputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& outputs, const tensorflow::string& outputName, std::vector<tensorflow::int64>& predictedLabels) {
        auto it = outputs.find(outputName);
        if (it == outputs.end()) {
            std::cout << "cannot find output " << outputName << std::endl;
            return false;
        }
        tensorflow::TensorProto& resultTensorProto = it->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(resultTensorProto);
        if (!converted) {
            std::cout << "the result tensor[" << it->first << "] convert failed." << std::endl;
            return false;
        }
        predictedLabels = argmax(tensor);
        return true;
    }

    bool schedulePredict() {
        PredictRequest predictRequest;
        PredictResponse response;
        ClientContext context;

        predictRequest.mutable_model_spec()->set_name(this->config.modelName);
        predictRequest.mutable_model_spec()->set_signature_name("serving_default");

        google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *predictRequest.mutable_inputs();

        // Pre-processing step.
        // Packing image into gRPC message.
        //this->prepareInputs(inputs, entry, inputName);
        prepareBatchedInputs(inputs, this->entries, this->config.inputName);

        // Actual predict request.
        //auto start = std::chrono::high_resolution_clock::now();
        
        //Status status = stub_->Predict(&context, predictRequest, &response);
        AsyncClientCall* call = new AsyncClientCall;
        call->response_reader = stub_->PrepareAsyncPredict(&(call->context), predictRequest, &this->cq_);
        call->response_reader->StartCall();

        call->response_reader->Finish(&(call->reply), &(call->status), (void*)call);

        return true;
    }

    void AsyncCompleteRpc() {
        std::cout << "AsyncCompleteRpc start" << std::endl;

        void* got_tag;
        bool ok = false;

        int finishedIterations = 0;
        while (cq_.Next(&got_tag, &ok)) {
            std::cout << "Received iteration " << finishedIterations + 1 << std::endl;

            std::cout << "Got reply" << std::endl;
            AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
            auto& response = call->reply;
            // The tag in this example is the memory location of the call object

            // Verify that the request was completed successfully. Note that "ok"
            // corresponds solely to the request for updates introduced by Finish().
            //GPR_ASSERT(ok);
            if (!ok) {
                std::cerr << "Request is not ok" << std::endl;
                finishedIterations++;
                delete call;
                if (finishedIterations == this->config.iterations) {
                    break;
                } else {
                    continue;
                }
            }

            if (!call->status.ok()) {
                std::cout << "gRPC call return code: " << call->status.error_code() << ": "
                          << call->status.error_message() << std::endl;
                finishedIterations++;
                delete call;
                if (finishedIterations == this->config.iterations) {
                    break;
                } else {
                    continue;
                }
            }

            std::cout << "call predict ok" << std::endl;
            //std::cout << "call predict time: " << duration.count() / 1000 << "ms" << std::endl;
            std::cout << "outputs size is " << response.outputs_size() << std::endl;

            // Post-processing step.
            // Extracting most probable label from resnet output.
            std::vector<tensorflow::int64> predictedLabels;
            if (!interpretOutputs(*response.mutable_outputs(), this->config.outputName, predictedLabels)) {
                std::cout << "error interpreting outputs" << std::endl;
                finishedIterations++;
                delete call;
                if (finishedIterations == this->config.iterations) {
                    break;
                } else {
                    continue;
                }
            }
            // int numberOfCorrectLabels = 0;  // TODO: save
            assert(predictedLabels.size() == this->entries.size());
            for (size_t i = 0; i < predictedLabels.size(); i++) {
                if (predictedLabels[i] == this->entries[i].expectedLabel) {
                    this->numberOfCorrectLabels++;
                }
            }

            // Once we're complete, deallocate the call object.
            finishedIterations++;
            delete call; // add to other
            if (finishedIterations == this->config.iterations) {
                break;
            } else {
                continue;
            }
        }
    }

    size_t getNumberOfCorrectLabels() const {
        return this->numberOfCorrectLabels;
    }

    static void start(const tensorflow::string& address, const std::vector<T>& entries, const Configuration& config) {
        auto begin = std::chrono::high_resolution_clock::now();  // avg, median, max time per predict
        ServingClient<T> client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()), entries, config);
        std::thread thread_ = std::thread(&ServingClient<T>::AsyncCompleteRpc, &client);
        for (tensorflow::int64 i = 0; i < config.iterations; i++) {
            if (!client.schedulePredict()) {
                return;
            }
            std::cout << "Finished scheduling " << i << " request" << std::endl;
        }
        thread_.join();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin);
        std::cout << "Overall accuracy: " << (client.getNumberOfCorrectLabels() * 100) / (config.iterations * config.batchSize) << "%" << std::endl;
        std::cout << "Total time: " << (duration.count() / 1000) << "ms" << std::endl;
    }

private:
    Configuration config;
    std::vector<T> entries;
    size_t numberOfCorrectLabels = 0;
};

int main(int argc, char** argv) {
    Configuration config;
    std::vector<tensorflow::Flag> flagList = {
        tensorflow::Flag("grpc_address", &config.address, "url to grpc service"),
        tensorflow::Flag("grpc_port", &config.port, "port to grpc service"),
        tensorflow::Flag("model_name", &config.modelName, "model name to request"),
        tensorflow::Flag("input_name", &config.inputName, "input tensor name with image"),
        tensorflow::Flag("output_name", &config.outputName, "output tensor name with classification result"),
        tensorflow::Flag("iterations", &config.iterations, "number of images per thread, by default each thread will use all images from list"),
        tensorflow::Flag("batch_size", &config.batchSize, "batch size of each iteration"),
        tensorflow::Flag("images_list", &config.imagesListPath, "path to a file with a list of labeled images"),
        tensorflow::Flag("layout", &config.layout, "binary, nhwc or nchw")};

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

    std::cout
        << "Address: " << config.address << std::endl
        << "Port: " << config.port << std::endl
        << "Images list path: " << config.imagesListPath << std::endl
        << "Layout: " << config.layout << std::endl;

    const tensorflow::string host = config.address + ":" + config.port;
    if (config.iterations == 0) {
        config.iterations = entries.size();
    }

    if (config.layout == "binary") {
        std::vector<BinaryData> images;
        if (!readImagesBinary(entries, images)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient<BinaryData>::start(host, images, config);
    } else {
        std::vector<CvMatData> images;
        if (!readImagesCvMat(entries, images, config.layout)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient<CvMatData>::start(host, images, config);
    }

    return 0;
}
