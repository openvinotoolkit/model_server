#pragma once

#include <grpcpp/server_context.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

namespace ovms {

class PredictionServiceImpl final : public tensorflow::serving::PredictionService::Service {
    grpc::Status Predict(
                grpc::ServerContext*                    context,
        const   tensorflow::serving::PredictRequest*    request,
                tensorflow::serving::PredictResponse*   response);
};

} // namespace ovms