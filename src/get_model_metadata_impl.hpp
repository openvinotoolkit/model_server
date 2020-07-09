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
#include "status.hpp"

namespace ovms {

using proto_signature_map_t = google::protobuf::Map<std::string, tensorflow::TensorInfo>;

class GetModelMetadataImpl {
public:
    static Status validate(
        const   tensorflow::serving::GetModelMetadataRequest*   request);

    static void convert(
        const   tensor_map_t&           from,
                proto_signature_map_t*  to);

    static void buildResponse(
        std::shared_ptr<ModelInstance>                  instance,
        tensorflow::serving::GetModelMetadataResponse*  response);

    static Status getModelStatus(
        const   tensorflow::serving::GetModelMetadataRequest*   request,
                tensorflow::serving::GetModelMetadataResponse*  response);
    static Status createGrpcRequest(std::string model_name, std::optional<int64_t> model_version, tensorflow::serving::GetModelMetadataRequest * request);
    static Status serializeResponse2Json(const tensorflow::serving::GetModelMetadataResponse * response, std::string * output);
};

}  // namespace ovms
