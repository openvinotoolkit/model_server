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
#include <memory>
#include <string>

#include "tensorflow_serving/apis/get_model_metadata.pb.h"

#include "modelmanager.hpp"

namespace ovms {

using proto_signature_map_t = google::protobuf::Map<std::string, tensorflow::TensorInfo>;

enum class GetModelMetadataStatusCode {
    OK,
    REQUEST_MODEL_SPEC_MISSING,     /*!< Request lacks model_spec */
    INVALID_SIGNATURE_DEF,          /*!< Requested signature is not supported */
    MODEL_MISSING,             /*!< Model with such name and/or version does not exist */
};

class GetModelMetadataStatus {
public:
    static const std::string& getError(const GetModelMetadataStatusCode code) {
        static const std::map<GetModelMetadataStatusCode, std::string> errors = {
            { GetModelMetadataStatusCode::OK,                               ""                                  },
            { GetModelMetadataStatusCode::REQUEST_MODEL_SPEC_MISSING,       "model_spec missing in request"     },
            { GetModelMetadataStatusCode::INVALID_SIGNATURE_DEF,            "Invalid signature name"            },
            { GetModelMetadataStatusCode::MODEL_MISSING,                    "Servable not found for request"    },
        };

        return errors.find(code)->second;
    }
};

class GetModelMetadataImpl {
public:
    static GetModelMetadataStatusCode validate(
        const   tensorflow::serving::GetModelMetadataRequest*   request);

    static void convert(
        const   tensor_map_t&           from,
                proto_signature_map_t*  to);

    static void buildResponse(
        std::shared_ptr<ModelInstance>                  instance,
        tensorflow::serving::GetModelMetadataResponse*  response);

    static GetModelMetadataStatusCode getModelStatus(
        const   tensorflow::serving::GetModelMetadataRequest*   request,
                tensorflow::serving::GetModelMetadataResponse*  response);
};

}  // namespace ovms
