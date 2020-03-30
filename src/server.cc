//*****************************************************************************
// Copyright 2018-2020 Intel Corporation
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
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>

#include <inference_engine.hpp>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ResourceQuota;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

using namespace InferenceEngine;

using std::cout;
using std::endl;

std::vector<int> get_shape(tensorflow::TensorShape shape) {
    std::vector<int> dims;
    for (const auto& dim : shape) {
        dims.push_back(dim.size);
    }
    return dims;
}

void printShape(const tensorflow::TensorShape& shape) {
    cout << "Tensor_shape: (" << endl;
    for (auto& i : get_shape(shape)) {
            cout << i << ", ";
    }
    cout << ")" << endl;
}

int getNumOfElements(const tensorflow::TensorProto& tensorProto) {
    tensorflow::TensorShape shape(tensorProto.tensor_shape());
    // printShape(shape);
    return shape.num_elements();
}

template<typename T>
void printTensor(std::vector<T> t) {
    unsigned int i = 0;
    cout << "Vector: ";
    for (auto it=t.begin(); it != t.end(); ++it, ++i) {
        if (i > 6 && i < t.size() - 10)
             continue;
        cout << *it << " ";
    }
    cout << endl;
}

template<typename T>
void printTensor(T ptr, unsigned int N) {
    unsigned int i = 0;
    cout << "Vector: ";
    for (; i < N; ++i) {
        if (i > 6 && i < N - 4)
             continue;
        cout << *(ptr+i) << " ";
    }
    cout << endl;
}

#define CASE(TYPE) \
    case tensorflow::DataTypeToEnum<TYPE>::value: {\
        const TYPE* proto_tensor_content = reinterpret_cast<const TYPE*>( \
            tensor.tensor_content().data()); \
        std::vector<TYPE> t(proto_tensor_content, \
            proto_tensor_content + num_of_elements); \
        printTensor(t); \
    }

void deserializePredict(const PredictRequest * const request) {
    for (auto& inputs : request->inputs()) {
        auto& tensor = inputs.second;
        auto num_of_elements = getNumOfElements(tensor);
        switch (tensor.dtype()) {
            CASE(int)
            CASE(float)
            CASE(double)
            default:
                return;
        }
    }
}

// alternative deserialize
class TensorBuffer {
 public:
    TensorBuffer(const int& dataType,
                 const int& numberOfElements,
                 const void* content) :
        dataType_(dataType),
        numberOfElements_(numberOfElements),
        content_(content) {}
    const void * const getContent() { return content_;}
    int getNumberOfElements() { return numberOfElements_;}

 private:
    const int dataType_;
    const int numberOfElements_;
    const void* content_;
};

auto deserializePredict2(const PredictRequest * const request) {
    for (auto& inputs : request->inputs()) {
        auto& tensor = inputs.second;
        auto dataType = tensor.dtype();
        auto numOfElements = getNumOfElements(tensor);
        auto proto_tensor_content = tensor.tensor_content().data();
        auto tensorBuffer = TensorBuffer(dataType,
                numOfElements,
                proto_tensor_content);
        printTensor(
                reinterpret_cast<const float*>(tensorBuffer.getContent()),
                tensorBuffer.getNumberOfElements());
        return tensorBuffer;
    }
}

std::string timeStamp() {
    using std::chrono::system_clock;
    auto currentTime = std::chrono::system_clock::now();
    char buffer[80];
    auto transformed = currentTime.time_since_epoch().count() / 1000000;
    auto millis = transformed % 1000;
    std::time_t tt;

    tt = system_clock::to_time_t(currentTime);
    auto timeinfo = localtime(&tt);
    strftime(buffer, 80, "%F %H:%M:%S", timeinfo);
    sprintf(buffer, "%s:%03d", buffer, static_cast<int>(millis));

    return std::string(buffer);
}

<<<<<<< HEAD
class OV
{
public:
    Core m_core;
    CNNNetwork m_network;
    ExecutableNetwork m_exec_network;

    std::string m_input_name;
    std::string m_output_name;

    OV(std::string path) : 
        m_core(), 
        m_network(m_core.ReadNetwork(path)),
        m_exec_network(m_core.LoadNetwork(m_network, "CPU"))
    {
        m_network.setBatchSize(1);
        m_network.getInputsInfo().begin()->second->setPrecision(Precision::FP32);
        m_network.getOutputsInfo().begin()->second->setPrecision(Precision::FP32);
        m_input_name = m_network.getInputsInfo().begin()->first;
        m_output_name = m_network.getOutputsInfo().begin()->first;
    }
};

OV ov("/models/resnet50/1/resnet_50_i8.xml");

class PredictionServiceImpl final : public PredictionService::Service
{
    Status Predict(
                ServerContext*      context,
        const   PredictRequest*     request,
                PredictResponse*    response)
    {
        std::cout << timeStamp() << " Received Predict() request\n";
        std::vector<float> tensor = deserializePredict(request);
=======
class PredictionServiceImpl final : public PredictionService::Service {
    Status Predict(
                ServerContext*      context,
        const   PredictRequest*     request,
                PredictResponse*    response) {
        std::cout << timeStamp() << " Received Predict() request" << std::endl;
        deserializePredict(request);
>>>>>>> Fix most of the cpplint errors.
        deserializePredict2(request);

        InferRequest infer_request = ov.m_exec_network.CreateInferRequest();

        TensorDesc tensorDesc(Precision::FP32, {1, 3, 224, 224}, Layout::NHWC);
        Blob::Ptr blob = make_shared_blob<float>(tensorDesc, tensor.data());

        infer_request.SetBlob(ov.m_input_name, blob);
        infer_request.Infer();

        Blob::Ptr output = infer_request.GetBlob(ov.m_output_name);
        const float* buffer = (const float*)output->buffer();

        // std::cout << "Infer output:\n" << std::setprecision(2) << std::fixed;;
        // for (int i = 0; i < 1000; i++)
        //     std::cout << (double) buffer[i] << " ";
        // std::cout << std::endl;

        return Status::OK;
    }
};

<<<<<<< HEAD
int main()
{
    std::cout << "Initializing gRPC\n";
=======

int main() {
    std::cout << "Initializing\n";
>>>>>>> Fix most of the cpplint errors.

    PredictionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:9000", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    const int SERVER_COUNT = 24;
    std::vector<std::unique_ptr<Server>> servers;
    for (int i = 0; i < SERVER_COUNT; i++) {
        servers.push_back(std::unique_ptr<Server>(builder.BuildAndStart()));
    }

    std::cout << "Servers started on port 9178" << std::endl;
    servers[0]->Wait();
    return 0;
}
