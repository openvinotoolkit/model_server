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
#include "rest_utils.hpp"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <spdlog/spdlog.h>

#include "absl/strings/escaping.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/util/json_tensor.h"
#pragma GCC diagnostic pop

#include "precision.hpp"
#include "tfs_frontend/tfs_utils.hpp"
#include "timer.hpp"

using tensorflow::DataType;
using tensorflow::DataTypeSize;
using tensorflow::serving::JsonPredictRequestFormat;
using tensorflow::serving::MakeJsonFromTensors;
using tensorflow::serving::PredictResponse;

namespace ovms {

Status checkValField(const size_t& fieldSize, const size_t& expectedElementsNumber) {
    if (fieldSize == 0)
        return StatusCode::REST_SERIALIZE_NO_DATA;
    if (fieldSize != expectedElementsNumber)
        return StatusCode::REST_SERIALIZE_VAL_FIELD_INVALID_SIZE;
    return StatusCode::OK;
}

Status makeJsonFromPredictResponse(
    PredictResponse& response_proto,
    std::string* response_json,
    Order order) {
    if (order == Order::UNKNOWN) {
        return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
    }

    Timer timer;
    using std::chrono::microseconds;

    timer.start("convert");

    for (auto& kv : *response_proto.mutable_outputs()) {
        auto& tensor = kv.second;

        size_t dataTypeSize = DataTypeSize(tensor.dtype());
        size_t expectedContentSize = dataTypeSize;
        for (int i = 0; i < tensor.tensor_shape().dim_size(); i++) {
            expectedContentSize *= tensor.tensor_shape().dim(i).size();
        }
        size_t expectedElementsNumber = dataTypeSize > 0 ? expectedContentSize / dataTypeSize : 0;
        bool seekDataInValField = false;

        if (tensor.tensor_content().size() == 0)
            seekDataInValField = true;
        else if (tensor.tensor_content().size() != expectedContentSize)
            return StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE;

        switch (tensor.dtype()) {
        case DataType::DT_FLOAT:
            if (seekDataInValField) {
                auto status = checkValField(tensor.float_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(float))
                    tensor.add_float_val(*reinterpret_cast<float*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT32:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int32_t))
                    tensor.add_int_val(*reinterpret_cast<int32_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT8:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int8_t))
                    tensor.add_int_val(*reinterpret_cast<int8_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT8:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint8_t))
                    tensor.add_int_val(*reinterpret_cast<uint8_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_DOUBLE:
            if (seekDataInValField) {
                auto status = checkValField(tensor.double_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(double))
                    tensor.add_double_val(*reinterpret_cast<double*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT16:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int16_t))
                    tensor.add_int_val(*reinterpret_cast<int16_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT64:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int64_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int64_t))
                    tensor.add_int64_val(*reinterpret_cast<int64_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT32:
            if (seekDataInValField) {
                auto status = checkValField(tensor.uint32_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint32_t))
                    tensor.add_uint32_val(*reinterpret_cast<uint32_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT64:
            if (seekDataInValField) {
                auto status = checkValField(tensor.uint64_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint64_t))
                    tensor.add_uint64_val(*reinterpret_cast<uint64_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        default:
            return StatusCode::REST_UNSUPPORTED_PRECISION;
        }
    }

    timer.stop("convert");
    timer.start("MakeJsonFromTensors");

    const auto& tf_status = MakeJsonFromTensors(
        response_proto.outputs(),
        order == Order::ROW ? JsonPredictRequestFormat::kRow : JsonPredictRequestFormat::kColumnar,
        response_json);

    timer.stop("MakeJsonFromTensors");
    SPDLOG_DEBUG("tensor_content to *_val container conversion: {:.3f} ms", timer.elapsed<microseconds>("convert") / 1000);
    SPDLOG_DEBUG("MakeJsonFromTensors call: {:.3f} ms", timer.elapsed<microseconds>("MakeJsonFromTensors") / 1000);

    if (!tf_status.ok()) {
        SPDLOG_ERROR("Creating json from tensors failed: {}", tf_status.error_message());
        return StatusCode::REST_PROTO_TO_STRING_ERROR;
    }

    return StatusCode::OK;
}

Status makeJsonFromPredictResponse(
    const ::inference::ModelInferResponse& response_proto,
    std::string* response_json) {
    Timer timer;
    using std::chrono::microseconds;
    timer.start("convert");

    rapidjson::Document response;
    response.SetObject();
    rapidjson::Value model_name, id;
    model_name = rapidjson::StringRef(response_proto.model_name().c_str());
    id = rapidjson::StringRef(response_proto.id().c_str());
    response.AddMember("model_name", model_name, response.GetAllocator());
    response.AddMember("id", id, response.GetAllocator());
    if (response_proto.model_version().length() > 0) {
        rapidjson::Value model_version;
        model_version = rapidjson::StringRef(response_proto.model_version().c_str());
        response.AddMember("model_version", model_version, response.GetAllocator());
    }
    if (response_proto.parameters_size() > 0) {
        rapidjson::Value parameters(rapidjson::kArrayType);

        for (const auto& parameter : response_proto.parameters()) {
            rapidjson::Value param_value, param_key;
            param_key = rapidjson::StringRef(parameter.first.c_str());
            switch (parameter.second.parameter_choice_case()) {
            case inference::InferParameter::ParameterChoiceCase::kBoolParam:
                param_value = parameter.second.bool_param();
                break;
            case inference::InferParameter::ParameterChoiceCase::kInt64Param:
                param_value = parameter.second.int64_param();
                break;
            case inference::InferParameter::ParameterChoiceCase::kStringParam:
                param_value = rapidjson::StringRef(parameter.second.string_param().c_str());
                break;
            default:
                break;  // return param error
            }
            parameters.AddMember(param_key, param_value, response.GetAllocator());
        }
        response.AddMember("parameters", parameters, response.GetAllocator());
    }
    rapidjson::Value outputs(rapidjson::kArrayType);

    bool seekDataInValField = false;
    if (response_proto.raw_output_contents_size() == 0)
        seekDataInValField = true;

    int tensor_it = 0;
    for (const auto& tensor : response_proto.outputs()) {
        size_t dataTypeSize = KFSDataTypeSize(tensor.datatype());
        size_t expectedContentSize = dataTypeSize;
        rapidjson::Value tensor_shape(rapidjson::kArrayType);
        for (int i = 0; i < tensor.shape().size(); i++) {
            expectedContentSize *= tensor.shape().at(i);
            tensor_shape.PushBack(tensor.shape().at(i), response.GetAllocator());
        }
        size_t expectedElementsNumber = dataTypeSize > 0 ? expectedContentSize / dataTypeSize : 0;
        rapidjson::Value output(rapidjson::kObjectType);
        rapidjson::Value tensor_name, tensor_datatype;
        tensor_name = rapidjson::StringRef(tensor.name().c_str());
        tensor_datatype = rapidjson::StringRef(tensor.datatype().c_str());
        rapidjson::Value tensor_data(rapidjson::kArrayType);
        output.AddMember("name", tensor_name, response.GetAllocator());
        output.AddMember("shape", tensor_shape, response.GetAllocator());
        output.AddMember("datatype", tensor_datatype, response.GetAllocator());

        if (tensor.parameters_size() > 0) {
            rapidjson::Value parameters(rapidjson::kArrayType);

            for (const auto& parameter : tensor.parameters()) {
                rapidjson::Value param_value, param_key;
                param_key = rapidjson::StringRef(parameter.first.c_str());
                switch (parameter.second.parameter_choice_case()) {
                case inference::InferParameter::ParameterChoiceCase::kBoolParam:
                    param_value = parameter.second.bool_param();
                    break;
                case inference::InferParameter::ParameterChoiceCase::kInt64Param:
                    param_value = parameter.second.int64_param();
                    break;
                case inference::InferParameter::ParameterChoiceCase::kStringParam:
                    param_value = rapidjson::StringRef(parameter.second.string_param().c_str());
                    break;
                default:
                    break;  // return param error
                }
                parameters.AddMember(param_key, param_value, response.GetAllocator());
            }
            output.AddMember("parameters", parameters, response.GetAllocator());
        }

        if (tensor.datatype() == "FP32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().fp32_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().fp32_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(float))
                    tensor_data.PushBack(*(reinterpret_cast<const float*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "INT64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int64_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(int64_t))
                    tensor_data.PushBack(*(reinterpret_cast<const int64_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "INT32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(int32_t))
                    tensor_data.PushBack(*(reinterpret_cast<const int32_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "INT16") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(int16_t))
                    tensor_data.PushBack(*(reinterpret_cast<const int16_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "INT8") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(int8_t))
                    tensor_data.PushBack(*(reinterpret_cast<const int8_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "UINT64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint64_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(uint64_t))
                    tensor_data.PushBack(*(reinterpret_cast<const uint64_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "UINT32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(uint32_t))
                    tensor_data.PushBack(*(reinterpret_cast<const uint32_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "UINT16") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(uint16_t))
                    tensor_data.PushBack(*(reinterpret_cast<const uint16_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "UINT8") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(uint8_t))
                    tensor_data.PushBack(*(reinterpret_cast<const uint8_t*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else if (tensor.datatype() == "FP64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().fp64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().fp64_contents()) {
                    tensor_data.PushBack(number, response.GetAllocator());
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(double))
                    tensor_data.PushBack(*(reinterpret_cast<const double*>(response_proto.raw_output_contents(tensor_it).data() + i)), response.GetAllocator());
            }
        } else {
            return StatusCode::REST_UNSUPPORTED_PRECISION;
        }

        output.AddMember("data", tensor_data, response.GetAllocator());
        outputs.PushBack(output, response.GetAllocator());
        tensor_it++;
    }

    response.AddMember("outputs", outputs, response.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    response.Accept(writer);
    *response_json = buffer.GetString();

    timer.stop("convert");
    SPDLOG_DEBUG("GRPC to HTTP response conversion: {:.3f} ms", timer.elapsed<microseconds>("convert") / 1000);

    return StatusCode::OK;
}

Status decodeBase64(std::string& bytes, std::string& decodedBytes) {
    auto status = Status(absl::Base64Unescape(bytes, &decodedBytes) ? StatusCode::OK : StatusCode::REST_BASE64_DECODE_ERROR);
    if (!status.ok()) {
        return status;
    }
    return StatusCode::OK;
}
}  // namespace ovms
