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
#include "opencv2/opencv.hpp"
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


struct Entry {
    tensorflow::string imagePath;
    tensorflow::int64 expectedLabel;
};

struct BinaryData {
    std::unique_ptr<char[]> imageData;
    std::streampos fileSize;
    tensorflow::int64 expectedLabel;
};

struct CvMatData {
    cv::Mat image;
    tensorflow::int64 expectedLabel;
    tensorflow::string layout;
};

template <typename T>
std::vector<T> reorderToNchw(const T* nhwcVector, int rows, int cols, int channels) {
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
    auto nchwVector = reorderToNchw<float>((float*)mat->data, rows, cols, channels);

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
        entryOut.image = cv::imread(entryIn.imagePath);
        entryOut.image.convertTo(entryOut.image, CV_32F);
        cv::resize(entryOut.image, entryOut.image, cv::Size(224, 224));
        if (layout == "nchw") {
            entryOut.image = reorderMatToNchw(&entryOut.image);
        }
        entriesOut.emplace_back(entryOut);
    }

    return true;
}

class ServingClient {
    std::unique_ptr<PredictionService::Stub> stub_;
public:
    ServingClient(std::shared_ptr<Channel> channel) : stub_(PredictionService::NewStub(channel)) {}

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

    bool prepareInput(tensorflow::TensorProto& proto, const BinaryData& entry) {
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        proto.add_string_val(entry.imageData.get(), entry.fileSize);
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        return true;
    }

    bool prepareInput(tensorflow::TensorProto& proto, const CvMatData& entry) {
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

        tensorflow::TensorProto proto;

        this->prepareInput(proto, entry);

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
        for (tensorflow::int64 i = 0; i < iterations; i++) {
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
    tensorflow::string address = "localhost";
    tensorflow::string port = "9000";
    tensorflow::string modelName = "resnet";
    tensorflow::string inputName = "0";
    tensorflow::string outputName = "1463";
    tensorflow::int64 iterations = 0;
    tensorflow::string imagesListPath = "";
    tensorflow::string layout = "binary";
    std::vector<tensorflow::Flag> flagList = {
        tensorflow::Flag("grpc_address", &address, "url to grpc service"),
        tensorflow::Flag("grpc_port", &port, "port to grpc service"),
        tensorflow::Flag("model_name", &modelName, "model name to request"),
        tensorflow::Flag("input_name", &inputName, "input tensor name with image"),
        tensorflow::Flag("output_name", &outputName, "output tensor name with classification result"),
        tensorflow::Flag("iterations", &iterations, "number of images per thread, by default each thread will use all images from list"),
        tensorflow::Flag("images_list", &imagesListPath, "path to a file with a list of labeled images"),
        tensorflow::Flag("layout", &layout, "binary, nhwc or nchw")
    };

    tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
    const bool result = tensorflow::Flags::Parse(&argc, argv, flagList);

    if (!result || imagesListPath.empty() || iterations < 0 || (layout != "binary" && layout != "nchw" && layout != "nhwc")) {
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
        if (!readImagesCvMat(entries, images, layout)) {
            std::cout << "Error reading binary images" << std::endl;
            return -1;
        }
        ServingClient::start(host, modelName, inputName, outputName, images, iterations);
    }

    return 0;
}
