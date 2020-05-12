//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#pragma once

#include <map>
#include <string>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#include "tensorflow_serving/apis/get_model_status.pb.h"


namespace ovms {

enum class GetModelStatusCode {
    OK,
    INTERNAL_ERROR, /*!< Internal server error, null pointer, etc. */
    INVALID_REQUEST,     /*!< Request lacks valid input data */
    NO_SUCH_MODEL_NAME,          /*!< Model with such name (in any version) does not exist */
    NO_SUCH_MODEL_VERSION,             /*!< Model with such name and version does not exist */
};

class GetModelStatus {
public:
    static const std::string& getError(const GetModelStatusCode code) {
        static const std::map<GetModelStatusCode, std::string> errors = {
            { GetModelStatusCode::OK,                               ""                              },
            { GetModelStatusCode::INTERNAL_ERROR,                   "Internal server error"         },
            { GetModelStatusCode::INVALID_REQUEST,                  "Request lacks valid input data"},
            { GetModelStatusCode::NO_SUCH_MODEL_NAME,               "Servable not found for request"},
            { GetModelStatusCode::NO_SUCH_MODEL_VERSION,            "Servable not found for request"},
        };

        return errors.find(code)->second;
    }
};

class ModelServiceImpl final : public tensorflow::serving::ModelService::Service {
 public:
  ::grpc::Status GetModelStatus(::grpc::ServerContext *context,
                                const tensorflow::serving::GetModelStatusRequest *request,
                                tensorflow::serving::GetModelStatusResponse *response) override;
  ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext *context,
                                           const tensorflow::serving::ReloadConfigRequest *request,
                                           tensorflow::serving::ReloadConfigResponse *response);
};

}  // namespace ovms
