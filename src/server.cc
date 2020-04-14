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
#include <condition_variable>
#include <mutex>
#include <tuple>

#include <inference_engine.hpp>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

#include "modelmanager.h"

#define DEBUG

#include "timer.h"

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

 InferRequest& performAsyncInfer(const std::shared_ptr<ovms::ModelInstance>& modelInstance, const std::string& inputName, const Blob::Ptr input) {
    std::condition_variable cv;
    std::mutex mx;
    std::unique_lock<std::mutex> lock(mx);

    auto& request = modelInstance->inferAsync(inputName, input, [&]() {
        cv.notify_one();
    });

    cv.wait(lock);
    return request;
}

class PredictionServiceImpl final : public PredictionService::Service {
    Status Predict(
                ServerContext*      context,
        const   PredictRequest*     request,
                PredictResponse*    response) {

        Timer timer;
        timer.start("total");
        ovms::ModelManager& manager = ovms::ModelManager::getInstance();

        const std::string& model_name = request->model_spec().name();
        const std::string& inputName = request->inputs().begin()->first;
        google::protobuf::int64 version = request->model_spec().version().value();

        // std::cout
        //     << timeStamp()
        //     << " Received Predict() request for model: "
        //     << model_name
        //     << "; version: "
        //     << version
        //     << std::endl;

        timer.start("model find");
        auto modelInstance = manager.findModelByName(model_name)->getModelInstanceByVersion(version);
        timer.stop("model find");

        // Deserialization
        timer.start("deserialization");
        std::unique_ptr<TensorBuffer> tensor_buffer = deserializePredict(request);
        const TensorDesc& tensorDesc = modelInstance->getInputsInfo().begin()->second->getTensorDesc();
        Blob::Ptr input = create_input_blob(tensorDesc, *tensor_buffer);
        timer.stop("deserialization");

        timer.start("async infer");
        auto& req = performAsyncInfer(modelInstance, inputName, input);
        timer.stop("async infer");

         timer.start("sync infer");
         //auto& req = modelInstance->infer(inputName, input);
         timer.stop("sync infer");

        // Serialization
        timer.start("serialization");
        std::string outputName = modelInstance->getOutputsInfo().begin()->first;
        auto output = req.GetBlob(outputName);
        auto& tensorProto = (*response->mutable_outputs())[outputName];
        tensorProto.Clear();
        tensorProto.set_dtype(modelInstance->getOutputsInfo().begin()->second->getPrecisionAsDataType());

        auto tensorProtoShape = tensorProto.mutable_tensor_shape();
        tensorProtoShape->Clear();
        tensorProtoShape->add_dim()->set_size(1);
        tensorProtoShape->add_dim()->set_size(output->size());

        auto tensorProtoContent = tensorProto.mutable_tensor_content();
        tensorProtoContent->assign((char*) output->buffer(), output->byteSize());
        timer.stop("serialization");
        timer.stop("total");

        // Statistics
        timer.print();

        return Status::OK;
    }
};

int main()
{
    const int PORT              = 9178;
    const int SERVER_COUNT      = 24;
    const std::string ADDR_URI  = std::string("0.0.0.0:") + std::to_string(PORT);

    ovms::ModelManager& manager = ovms::ModelManager::getInstance();

    ovms::Status status = manager.start("/models/config.json");

    if (status != ovms::Status::OK) {
        std::cout << "ovms::ModelManager::Start() Error: " << int(status) << std::endl;
        return 1;
    }

    PredictionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(ADDR_URI, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
    builder.RegisterService(&service);

    std::vector<std::unique_ptr<Server>> servers;
    for (int i = 0; i < SERVER_COUNT; i++) {
        servers.push_back(std::unique_ptr<Server>(builder.BuildAndStart()));
    }

    std::cout << "Server started on port " << PORT << std::endl;
    servers[0]->Wait();
    return 0;
}
