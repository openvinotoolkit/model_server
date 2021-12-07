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

#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "common.hpp"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "opencv2/opencv.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;

public:
    ServingClient(std::shared_ptr<Channel> channel) :
        stub_(PredictionService::NewStub(channel)) {}

    static tensorflow::int64 argmax(const tensorflow::Tensor& tensor) {
        float topConfidence = 0;
        tensorflow::int64 topLabel = -1;
        for (tensorflow::int64 i = 0; i < tensor.NumElements(); i++) {
            float confidence = ((float*)tensor.data())[i];
            if (topLabel == -1 || topConfidence < confidence) {
                topLabel = i;
                topConfidence = confidence;
            }
        }
        return topLabel;
    }

    // Pre-processing function for binary images.
    // Images loaded from disk are packed into gRPC request proto.
    bool prepareInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const BinaryData& entry, const tensorflow::string& inputName) {
        tensorflow::TensorProto proto;
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        proto.add_string_val(entry.imageData.get(), entry.fileSize);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        inputs[inputName] = proto;
        return true;
    }

    // Pre-processing function for images in array format.
    // Images loaded from disk are packed into tensor_content in plain array format (using OpenCV) either in NCHW or NHWC layout.
    bool prepareInputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs, const CvMatData& entry, const tensorflow::string& inputName) {
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

    // Post-processing function for classification.
    // Most probable label is selected from the output.
    bool interpretOutputs(google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& outputs, const tensorflow::string& outputName, tensorflow::int64& predictedLabel) {
        auto it = outputs.find(outputName);
        if (it == outputs.end()) {
            std::cout << "cannot find output " << outputName << std::endl;
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
        predictedLabel = this->argmax(tensor);
        return true;
    }

    template <class T>
    bool predict(
        const tensorflow::string& modelName,
        const tensorflow::string& inputName,
        const tensorflow::string& outputName,
        const T& entry,
        bool& isLabelCorrect) {
        PredictRequest predictRequest;
        PredictResponse response;
        ClientContext context;

        predictRequest.mutable_model_spec()->set_name(modelName);
        predictRequest.mutable_model_spec()->set_signature_name("serving_default");

        google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *predictRequest.mutable_inputs();

        // Pre-processing step.
        // Packing image into gRPC message.
        this->prepareInputs(inputs, entry, inputName);

        // Actual predict request.
        auto start = std::chrono::high_resolution_clock::now();
        Status status = stub_->Predict(&context, predictRequest, &response);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

        // gRPC error handling.
        if (!status.ok()) {
            std::cout << "gRPC call return code: " << status.error_code() << ": "
                      << status.error_message() << std::endl;
            return false;
        }

        std::cout << "call predict ok" << std::endl;
        std::cout << "call predict time: " << duration.count() / 1000 << "ms" << std::endl;
        std::cout << "outputs size is " << response.outputs_size() << std::endl;

        // Post-processing step.
        // Extracting most probable label from classification model output.
        tensorflow::int64 predictedLabel = -1;
        if (!this->interpretOutputs(*response.mutable_outputs(), outputName, predictedLabel)) {
            return false;
        }
        isLabelCorrect = predictedLabel == entry.expectedLabel;

        return true;
    }

    template <class T>
    static void start(
        const tensorflow::string& address,
        const tensorflow::string& modelName,
        const tensorflow::string& inputName,
        const tensorflow::string& outputName,
        const std::vector<T>& entries,
        tensorflow::int64 iterations) {
        auto begin = std::chrono::high_resolution_clock::now();
        tensorflow::int64 correctLabels = 0;
        ServingClient client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
        for (tensorflow::int64 i = 0; i < iterations; i++) {
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
        std::cout << "Total time divided by number of requests: " << (duration.count() / 1000) / iterations << "ms" << std::endl;
    }
};

int main(int argc, char** argv) {
    tensorflow::string address = "localhost";
    tensorflow::string port = "9000";
    tensorflow::string modelName = "resnet";
    tensorflow::string inputName = "0";
    tensorflow::string outputName = "1463";
    tensorflow::int64 iterations = 0;
    tensorflow::string imagesListPath = "input_images.txt";
    tensorflow::string layout = "binary";
    tensorflow::int64 width = 224;
    tensorflow::int64 height = 224;
    std::vector<tensorflow::Flag> flagList = {
        tensorflow::Flag("grpc_address", &address, "url to grpc service"),
        tensorflow::Flag("grpc_port", &port, "port to grpc service"),
        tensorflow::Flag("model_name", &modelName, "model name to request"),
        tensorflow::Flag("input_name", &inputName, "input tensor name with image"),
        tensorflow::Flag("output_name", &outputName, "output tensor name with classification result"),
        tensorflow::Flag("iterations", &iterations, "number of images per thread, by default each thread will use all images from list"),
        tensorflow::Flag("images_list", &imagesListPath, "path to a file with a list of labeled images"),
        tensorflow::Flag("layout", &layout, "binary, nhwc or nchw"),
        tensorflow::Flag("width", &width, "input images width will be resized to this value; not applied to binary input"),
        tensorflow::Flag("height", &height, "input images height will be resized to this value; not applied to binary input")};

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
    const bool result = tensorflow::Flags::Parse(&argc, argv, flagList);

    if (!result || imagesListPath.empty() || iterations < 0 || (layout != "binary" && layout != "nchw" && layout != "nhwc") || width <= 0 || height <= 0) {
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

    std::cout
        << "Address: " << address << std::endl
        << "Port: " << port << std::endl
        << "Images list path: " << imagesListPath << std::endl
        << "Layout: " << layout << std::endl;

    const tensorflow::string host = address + ":" + port;
    if (iterations == 0) {
        iterations = entries.size();
    }

    if (layout == "binary") {
        std::vector<BinaryData> images;
        if (!readImagesBinary(entries, images)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient::start(host, modelName, inputName, outputName, images, iterations);
    } else {
        std::vector<CvMatData> images;
        if (!readImagesCvMat(entries, images, layout, width, height)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient::start(host, modelName, inputName, outputName, images, iterations);
    }

    return 0;
}
