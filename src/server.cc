#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>


#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using namespace tensorflow::serving;


class PredictionServiceImpl final : public PredictionService::Service
{
    Status Predict(
                ServerContext*      context, 
        const   PredictRequest*     request, 
                PredictResponse*    response)
    {
        std::cout << "Received Predict() request" << std::endl;
        return Status::OK;
    }
};


int main()
{
    std::cout << "Initializing\n";
    PredictionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:9000", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server started on port 9000" << std::endl;
    server->Wait();
    return 0;
}