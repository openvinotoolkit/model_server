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

#include <filesystem>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <tuple>

#include <inference_engine.hpp>

#include "../modelmanager.hpp"
#include "../tensorinfo.hpp"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using inputs_info_t = std::map<std::string, std::tuple<ovms::shape_t, tensorflow::DataType>>;

const std::string dummy_model_location = std::filesystem::current_path().u8string() + "/src/test/dummy";

const ovms::ModelConfig DUMMY_MODEL_CONFIG{
    "dummy",
    dummy_model_location,  // base path
    "CPU",  // backend
    "1",      // batchsize
    1,      // NIREQ
    0,       // model_version unuesed since version are read from path
    dummy_model_location,  // local path
};

constexpr const char* DUMMY_MODEL_INPUT_NAME = "b";
constexpr const char* DUMMY_MODEL_OUTPUT_NAME = "a";
constexpr const int DUMMY_MODEL_INPUT_SIZE = 10;
constexpr const int DUMMY_MODEL_OUTPUT_SIZE = 10;

static ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::shape_t>&& tensors,
    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32) {
    ovms::tensor_map_t result;
    for (const auto& kv : tensors) {
        result[kv.first] = std::make_shared<ovms::TensorInfo>(
            kv.first,
            precision,
            kv.second);
    }
    return result;
}

static tensorflow::serving::PredictRequest preparePredictRequest(inputs_info_t requestInputs){
    tensorflow::serving::PredictRequest request;
    for (auto const& it : requestInputs){
        auto& name = it.first;
        auto [ shape, dtype ] = it.second;

        auto& input = (*request.mutable_inputs())[name];
        input.set_dtype(dtype);
        size_t numberOfElements = 1;
        for (auto const& dim : shape){
            input.mutable_tensor_shape()->add_dim()->set_size(dim);
            numberOfElements *= dim;
        }
        *input.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(dtype), '1');
    }
    return request;
}

static std::vector<int> asVector(const tensorflow::TensorShapeProto& proto) {
    std::vector<int> shape;
    for (int i = 0; i < proto.dim_size(); i++) {
        shape.push_back(proto.dim(i).size());
    }
    return shape;
}

static std::vector<google::protobuf::int32> asVector(google::protobuf::RepeatedField<google::protobuf::int32>* container) {
    std::vector<google::protobuf::int32> result(container->size(), 0);
    std::memcpy(result.data(), container->mutable_data(), result.size() * sizeof(google::protobuf::int32));
    return result;
}

template <typename T>
static std::vector<T> asVector(const std::string& tensor_content) {
    std::vector<T> v(tensor_content.size() / sizeof(T) + 1);
    std::memcpy(
        reinterpret_cast<char*>(v.data()),
        reinterpret_cast<const char*>(tensor_content.data()),
        tensor_content.size());
    v.resize(tensor_content.size() / sizeof(T));
    return v;
}

// returns path to a file.
static std::string createConfigFileWithContent(const std::string& content, std::string filename = "/tmp/ovms_config_file.json") {
    std::ofstream configFile{filename};
    configFile << content << std::endl;
    configFile.close();
    return filename;
}

class ConstructorEnabledModelManager : public ovms::ModelManager {
public:
    ConstructorEnabledModelManager() :
        ovms::ModelManager() {}
    ~ConstructorEnabledModelManager() {
        SPDLOG_INFO("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
        models.clear();
        SPDLOG_INFO("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    }
};
