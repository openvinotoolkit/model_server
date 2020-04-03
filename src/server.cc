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
#include <chrono> // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <tuple>

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

using tensorflow::DataType;
using tensorflow::DataTypeToEnum;
using tensorflow::TensorProto;
using tensorflow::TensorShape;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

using namespace InferenceEngine;

using std::cout;
using std::endl;

const std::vector<int> getShape(const TensorShape& shape) {
    std::vector<int> dims(shape.dims());
    std::transform(shape.begin(), shape.end(), dims.begin(), [](const auto& dim) { return dim.size;});
    return dims;
}

void printShape(const TensorShape& shape) {
    cout << "Tensor_shape: (";
    for (auto& i : getShape(shape)) {
            cout << i << ", ";
    }
    cout << ")" << endl;
}

int getNumOfElements(const TensorProto& tensorProto) {
    TensorShape shape(tensorProto.tensor_shape());
    // printShape(shape);
    return shape.num_elements();
}

template<typename T>
void printTensor(T ptr, unsigned int N) {
    cout << "Vector: ";
    for (unsigned int i = 0; i < N; ++i) {
        if (i > 6 && i < N - 4)
             continue;
        cout << *(ptr+i) << " ";
    }
    cout << endl;
}

class TensorBuffer {
 public:
    TensorBuffer(const DataType& dataType,
                 const int& numberOfElements,
                 const void* content) :
        dataType_(dataType),
        numberOfElements_(numberOfElements),
        data_(content) {}
    const void * const data() const { return data_;}
    int getNumberOfElements() const { return numberOfElements_;}
    DataType getDataType() const { return dataType_;}

 private:
    const DataType dataType_;
    const int numberOfElements_;
    const void* data_;
};

// TODO(atobisze) Return multiple inputs
std::unique_ptr<TensorBuffer> deserializePredict(const PredictRequest * const request) {
    for (auto& inputs : request->inputs()) {
        auto& tensor = inputs.second;
        auto dataType = tensor.dtype();
        auto numOfElements = getNumOfElements(tensor);
        auto proto_tensor_content = tensor.tensor_content().data();
        return std::make_unique<TensorBuffer>(dataType, numOfElements, proto_tensor_content);
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

class OV {
 public:
    Core m_core;
    CNNNetwork m_network;
    ExecutableNetwork m_exec_network;

    std::string m_input_name;
    std::string m_output_name;

    explicit OV(std::string path) :
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


#define CASE(TYPE) \
    case DataTypeToEnum<TYPE>::value: {\
        return make_shared_blob<TYPE>(tensorDesc, const_cast<TYPE*>(reinterpret_cast<const TYPE*>(tensorBuffer.data()))); \
    }

Blob::Ptr create_input_blob(TensorDesc tensorDesc, const TensorBuffer& tensorBuffer) {
    switch(tensorBuffer.getDataType()) {
    CASE(float)
    CASE(double)
    CASE(int)
    }
}

class PredictionServiceImpl final : public PredictionService::Service {
    Status Predict(
                ServerContext*      context,
        const   PredictRequest*     request,
                PredictResponse*    response) {
        // std::cout << timeStamp() << " Received Predict() request\n";
        std::unique_ptr<TensorBuffer> tensor_buffer = deserializePredict(request);
        //printTensor(reinterpret_cast<const float*>(tensorBuffer->data()), tensorBuffer->getNumberOfElements());

        InferRequest infer_request = ov.m_exec_network.CreateInferRequest();
        TensorDesc tensorDesc(Precision::FP32, {1, 3, 224, 224}, Layout::NHWC);
        Blob::Ptr blob = create_input_blob(tensorDesc, *tensor_buffer);

        infer_request.SetBlob(ov.m_input_name, blob);
        infer_request.Infer();
        /*for (auto output : ov.m_exec_network.GetOutputsInfo()) {
            cout << "<<Output name:" << output.first << endl;
        }*/
        auto outputsInfo = ov.m_exec_network.GetOutputsInfo();
        // TODO(atobisze) use DataType & EnumToDataType to pick type
        std::for_each(outputsInfo.begin(), outputsInfo.end(),
            [response, &infer_request](
                    std::pair<const std::string,
                    std::shared_ptr<const InferenceEngine::Data> > output) {
                auto& tensor_proto = (*response->mutable_outputs())[output.first];
                tensor_proto.Clear();
                tensor_proto.set_dtype(DataType::DT_FLOAT);
                auto tensor_proto_shape = tensor_proto.mutable_tensor_shape();
                tensor_proto_shape->Clear();
                Blob::Ptr blob_output = infer_request.GetBlob(ov.m_output_name);
                tensor_proto_shape->add_dim()->set_size(blob_output->size());
                auto tensor_proto_content = tensor_proto.mutable_tensor_content();
                tensor_proto_content->assign((char*)blob_output->buffer(), blob_output->byteSize());
                /*cout.precision(17);
                for (int i = 0; i < OUTPUT_TENSOR_SIZE; i++)
                    if (buffer[i] > 0.01 || i < 5)
                        std::cout << "Index:" << i << " Value: " << std::fixed << (double) buffer[i] << endl;
                std::cout << std::endl;*/
                });
        return Status::OK;
    }
};

int main() {
    std::cout << "Initializing gRPC OVMS C++\n";

    PredictionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:9178", grpc::InsecureServerCredentials());
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
