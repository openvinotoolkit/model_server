#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ctime>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ResourceQuota;

using namespace tensorflow::serving;

using std::cout;
using std::endl;

std::vector<int> get_shape(tensorflow::TensorShape shape) {
    std::vector<int> dims;
    for(const auto& dim : shape) {
        dims.push_back(dim.size);
    }
    return dims;
}

void printShape(const tensorflow::TensorShape& shape) {
    auto a=get_shape(shape);
    cout << "Tensor_shape" << endl;
    for(auto& i : a) {
            cout << i << endl;
    }
}

int getNumOfElements(const tensorflow::TensorProto& tensorProto) {
    tensorflow::TensorShape shape(tensorProto.tensor_shape());
    //printShape(shape);
    return shape.num_elements();
}

void printTensor(std::vector<float> t) {
    unsigned int i = 0;
    cout << "Vector: ";
    for(auto it=t.begin(); it!=t.end(); ++it, ++i) {
        if(i>6 && i < t.size() - 10)
             continue;
        cout << *it << " ";
    }
    cout << endl;
}

auto deserializePredict(const PredictRequest * const request) {
    for(auto& inputs : request->inputs()) {
        auto& tensor = inputs.second;
        //auto data_type = tensor.dtype();
        //cout << data_type << endl;
        auto num_of_elements = getNumOfElements(tensor);
        const float* proto_tensor_content = reinterpret_cast<const float*>(tensor.tensor_content().data());
        std::vector<float> t(proto_tensor_content, proto_tensor_content + num_of_elements);
        printTensor(t);
        //cout << "Ended processing" << endl;
        return t;
    }
}

std::string timeStamp()
{
    using std::chrono::system_clock;
    auto currentTime = std::chrono::system_clock::now();
    char buffer[80];

    auto transformed = currentTime.time_since_epoch().count() / 1000000;

    auto millis = transformed % 1000;

    std::time_t tt;
    tt = system_clock::to_time_t ( currentTime );
    auto timeinfo = localtime (&tt);
    strftime (buffer,80,"%F %H:%M:%S",timeinfo);
    sprintf(buffer, "%s:%03d",buffer,(int)millis);

    return std::string(buffer);
}

class PredictionServiceImpl final : public PredictionService::Service
{
    Status Predict(
                ServerContext*      context,
        const   PredictRequest*     request,
                PredictResponse*    response)
    {
        std::cout << timeStamp() << " Received Predict() request" << std::endl;
        auto tensor = deserializePredict(request);
        return Status::OK;
    }
};


int main()
{
    std::cout << "Initializing\n";

    PredictionServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:9178", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    const int SERVER_COUNT = 24;

    std::vector<std::unique_ptr<Server>> servers;
    for (int i = 0; i < SERVER_COUNT; i++)
    {
        servers.push_back(std::unique_ptr<Server>(builder.BuildAndStart()));
    }

    std::cout << "Servers started on port 9178" << std::endl;
    servers[0]->Wait();
    return 0;
}
