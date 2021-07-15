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

#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

typedef tensorflow::string string;
typedef tensorflow::int64 int64;

struct Entry {
    string imagePath;
    int64 expectedLabel;
};

struct BinaryData {
    std::unique_ptr<char[]> imageData;
    std::streampos fileSize;
    int64 expectedLabel;
};

bool readImagesList(const string& path, std::vector<Entry>& entries) {
    entries.clear();
    std::ifstream infile(path);
    string image = "";
    int64 label = 0;

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

class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;
public:
    ServingClient(std::shared_ptr<Channel> channel) : stub_(PredictionService::NewStub(channel)) {}

    static int64 argmax(const tensorflow::Tensor& tensor) {
        float topConfidence = 0;
        int64 topLabel = -1;
        for (int64 i = 0; i < tensor.NumElements(); i++) {
            float confidence = ((float*)tensor.data())[i];
            if (topLabel == -1 || topConfidence < confidence) {
                topLabel = i;
                topConfidence = confidence;
            }
        }
        return topLabel;
    }

    bool predict(
        const string& modelName,
        const string& inputName,
        const string& outputName,
        const BinaryData& entry,
        bool& isLabelCorrect) {

        PredictRequest predictRequest;
        PredictResponse response;
        ClientContext context;

        predictRequest.mutable_model_spec()->set_name(modelName);
        predictRequest.mutable_model_spec()->set_signature_name("serving_default");

        google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *predictRequest.mutable_inputs();

        tensorflow::TensorProto proto;

        proto.set_dtype(tensorflow::DataType::DT_STRING);
        proto.add_string_val(entry.imageData.get(), entry.fileSize);

        proto.mutable_tensor_shape()->add_dim()->set_size(1);

        inputs[inputName] = proto;

        auto start = std::chrono::high_resolution_clock::now();
        Status status = stub_->Predict(&context, predictRequest, &response);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

        if (status.ok()) {
            std::cout << "call predict ok" << std::endl;
            std::cout << "call predict time: " << duration.count() / 1000 << "ms" << std::endl;
            std::cout << "outputs size is " << response.outputs_size() << std::endl;

            auto it = response.mutable_outputs()->find(outputName);
            if (it == response.mutable_outputs()->end()) {
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

            auto label = this->argmax(tensor);
            std::cout << "most probable label: " << label << "; expected: " << entry.expectedLabel;
            if (entry.expectedLabel == label) {
                std::cout << "; OK" << std::endl;
                isLabelCorrect = true;
                return true;
            } else {
                std::cout << "; Incorrect" << std::endl;
                isLabelCorrect = false;
                return true;
            }
        } else {
            std::cout << "gRPC call return code: " << status.error_code() << ": "
                        << status.error_message() << std::endl;
            return false;
        }
        return true;
    }

    static void start(
        const string& address,
        const string& modelName,
        const string& inputName,
        const string& outputName,
        const std::vector<BinaryData>& entries,
        int64 iterations) {
    
        auto begin = std::chrono::high_resolution_clock::now();
        int64 correctLabels = 0;
        for (int64 i = 0; i < iterations; i++) {
            ServingClient client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
            bool isLabelCorrect = false;
            if (!client.predict(modelName, inputName, outputName, entries[i % entries.size()], isLabelCorrect)) {
                return;
            }
            if (isLabelCorrect) {
                correctLabels++;
            }
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin);

        std::cout << "Overall accuracy: " << (correctLabels * 100) / iterations << "%" << std::endl;
        std::cout << "Average predict time: " << (duration.count() / 1000) / iterations << "ms" << std::endl;
    }
};

int main(int argc, char** argv) {
    string address = "localhost";
    string port = "9000";
    string modelName = "resnet";
    string inputName = "0";
    string outputName = "1463";
    int64 iterations = 0;
    string imagesListPath = "";
    std::vector<tensorflow::Flag> flagList = {
        tensorflow::Flag("grpc_address", &address, "url to grpc service"),
        tensorflow::Flag("grpc_port", &port, "port to grpc service"),
        tensorflow::Flag("model_name", &modelName, "model name to request"),
        tensorflow::Flag("input_name", &inputName, "input tensor name with image"),
        tensorflow::Flag("output_name", &outputName, "output tensor name with classification result"),
        tensorflow::Flag("iterations", &iterations, "number of images per thread, by default each thread will use all images from list"),
        tensorflow::Flag("images_list", &imagesListPath, "path to a file with a list of labeled images")
    };

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
    const bool result = tensorflow::Flags::Parse(&argc, argv, flagList);

    if (!result || imagesListPath.empty() || iterations < 0) {
        std::cout << usage;
        return -1;
    }

    std::vector<Entry> entries;
    if (!readImagesList(imagesListPath, entries)) {
        std::cout << "Error parsing images_list" << std::endl;
        return -1;
    }

    if (entries.empty()) {
        std::cout << "Empty images_list" << std::endl;
        return -1;
    }

    std::vector<BinaryData> binaryImages;
    if (!readImagesBinary(entries, binaryImages)) {
        std::cout << "Error reading binary images" << std::endl;
        return -1;
    }

    std::cout
        << "Address: " << address << std::endl
        << "Port: " << port << std::endl
        << "Images list path: " << imagesListPath << std::endl;

    if (iterations == 0) {
        iterations = binaryImages.size();
    }

    std::cout << "Processing images..." << std::endl;

    const string host = address + ":" + port;
    ServingClient::start(host, modelName, inputName, outputName, binaryImages, iterations);

    return 0;
}
