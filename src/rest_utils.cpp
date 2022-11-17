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
#include "kfs_frontend/kfs_utils.hpp"
#include "precision.hpp"
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "status.hpp"
#include "tfs_frontend/tfs_utils.hpp"
#include "timer.hpp"

using tensorflow::DataType;
using tensorflow::DataTypeSize;
using tensorflow::serving::JsonPredictRequestFormat;
using tensorflow::serving::MakeJsonFromTensors;
using tensorflow::serving::PredictResponse;

namespace {
enum : unsigned int {
    CONVERT,
    MAKE_JSON_FROM_TENSORS,
    TIMER_END
};
}

namespace ovms {

static Status checkValField(const size_t& fieldSize, const size_t& expectedElementsNumber) {
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

    Timer<TIMER_END> timer;
    using std::chrono::microseconds;

    timer.start(CONVERT);

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

    timer.stop(CONVERT);
    timer.start(MAKE_JSON_FROM_TENSORS);

    const auto& tf_status = MakeJsonFromTensors(
        response_proto.outputs(),
        order == Order::ROW ? JsonPredictRequestFormat::kRow : JsonPredictRequestFormat::kColumnar,
        response_json);

    timer.stop(MAKE_JSON_FROM_TENSORS);
    SPDLOG_DEBUG("tensor_content to *_val container conversion: {:.3f} ms", timer.elapsed<microseconds>(CONVERT) / 1000);
    SPDLOG_DEBUG("MakeJsonFromTensors call: {:.3f} ms", timer.elapsed<microseconds>(MAKE_JSON_FROM_TENSORS) / 1000);

    if (!tf_status.ok()) {
        SPDLOG_ERROR("Creating json from tensors failed: {}", tf_status.error_message());
        return StatusCode::REST_PROTO_TO_STRING_ERROR;
    }

    return StatusCode::OK;
}

static Status parseResponseParameters(const ::KFSResponse& response_proto, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer) {
    if (response_proto.parameters_size() > 0) {
        writer.Key("parameters");
        writer.StartObject();

        for (const auto& protoParameter : response_proto.parameters()) {
            writer.Key(protoParameter.first.c_str());
            switch (protoParameter.second.parameter_choice_case()) {
            case inference::InferParameter::ParameterChoiceCase::kBoolParam:
                writer.Bool(protoParameter.second.bool_param());
                break;
            case inference::InferParameter::ParameterChoiceCase::kInt64Param:
                writer.Int(protoParameter.second.int64_param());
                break;
            case inference::InferParameter::ParameterChoiceCase::kStringParam:
                writer.String(protoParameter.second.string_param().c_str());
                break;
            default:
                break;  // return param error
            }
        }
        writer.EndObject();
    }

    return StatusCode::OK;
}

static Status parseOutputParameters(const inference::ModelInferResponse_InferOutputTensor& output, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer, int bytesOutputSize) {
    if (output.parameters_size() > 0 || bytesOutputSize > 0) {
        writer.Key("parameters");
        writer.StartObject();

        for (const auto& protoParameter : output.parameters()) {
            writer.Key(protoParameter.first.c_str());
            switch (protoParameter.second.parameter_choice_case()) {
            case inference::InferParameter::ParameterChoiceCase::kBoolParam:
                writer.Bool(protoParameter.second.bool_param());
                break;
            case inference::InferParameter::ParameterChoiceCase::kInt64Param:
                writer.Int(protoParameter.second.int64_param());
                break;
            case inference::InferParameter::ParameterChoiceCase::kStringParam:
                writer.String(protoParameter.second.string_param().c_str());
                break;
            default:
                break;  // return param error
            }
        }
        if (bytesOutputSize > 0) {
            writer.Key("binary_data_size");
            writer.Int(bytesOutputSize);
        }
        writer.EndObject();
    }

    return StatusCode::OK;
}

template <typename ValueType>
static void fillTensorDataWithIntValuesFromRawContents(const ::KFSResponse& response_proto, int tensor_it, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer) {
    for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(ValueType))
        writer.Int(*(reinterpret_cast<const ValueType*>(response_proto.raw_output_contents(tensor_it).data() + i)));
}

template <typename ValueType>
static void fillTensorDataWithUintValuesFromRawContents(const ::KFSResponse& response_proto, int tensor_it, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer) {
    for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(ValueType))
        writer.Int(*(reinterpret_cast<const ValueType*>(response_proto.raw_output_contents(tensor_it).data() + i)));
}

template <typename ValueType>
static void fillTensorDataWithFloatValuesFromRawContents(const ::KFSResponse& response_proto, int tensor_it, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer) {
    for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(ValueType))
        writer.Double(*(reinterpret_cast<const ValueType*>(response_proto.raw_output_contents(tensor_it).data() + i)));
}

static Status parseOutputs(const ::KFSResponse& response_proto, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer, std::string& bytesOutputsBuffer) {
    writer.Key("outputs");
    writer.StartArray();

    bool seekDataInValField = false;
    if (response_proto.raw_output_contents_size() == 0)
        seekDataInValField = true;
    int tensor_it = 0;
    for (const auto& tensor : response_proto.outputs()) {
        size_t dataTypeSize = KFSDataTypeSize(tensor.datatype());
        size_t expectedContentSize = dataTypeSize;
        for (int i = 0; i < tensor.shape().size(); i++) {
            expectedContentSize *= tensor.shape().at(i);
        }
        size_t expectedElementsNumber = dataTypeSize > 0 ? expectedContentSize / dataTypeSize : 0;

        if (!seekDataInValField && (response_proto.raw_output_contents(tensor_it).size() != expectedContentSize))
            return StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE;
        writer.StartObject();
        writer.Key("name");
        writer.String(tensor.name().c_str());
        writer.Key("shape");
        writer.StartArray();
        for (int i = 0; i < tensor.shape().size(); i++) {
            writer.Int(tensor.shape().at(i));
        }
        writer.EndArray();
        writer.Key("datatype");
        writer.String(tensor.datatype().c_str());
        if (tensor.datatype() != "BYTES") {
            writer.Key("data");
            writer.StartArray();
        }
        if (tensor.datatype() == "FP32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().fp32_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().fp32_contents()) {
                    writer.Double(number);
                }
            } else {
                fillTensorDataWithFloatValuesFromRawContents<float>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "INT64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int64_contents()) {
                    writer.Int64(number);
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(int64_t))
                    writer.Int64(*(reinterpret_cast<const int64_t*>(response_proto.raw_output_contents(tensor_it).data() + i)));
            }
        } else if (tensor.datatype() == "INT32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    writer.Int(number);
                }
            } else {
                fillTensorDataWithIntValuesFromRawContents<int32_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "INT16") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    writer.Int(number);
                }
            } else {
                fillTensorDataWithIntValuesFromRawContents<int16_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "INT8") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().int_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().int_contents()) {
                    writer.Int(number);
                }
            } else {
                fillTensorDataWithIntValuesFromRawContents<int8_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "UINT64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint64_contents()) {
                    writer.Uint64(number);
                }
            } else {
                for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(uint64_t))
                    writer.Uint64(*(reinterpret_cast<const uint64_t*>(response_proto.raw_output_contents(tensor_it).data() + i)));
            }
        } else if (tensor.datatype() == "UINT32") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    writer.Uint(number);
                }
            } else {
                fillTensorDataWithUintValuesFromRawContents<uint32_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "UINT16") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    writer.Uint(number);
                }
            } else {
                fillTensorDataWithUintValuesFromRawContents<uint16_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "UINT8") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().uint_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().uint_contents()) {
                    writer.Uint(number);
                }
            } else {
                fillTensorDataWithUintValuesFromRawContents<uint8_t>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "FP64") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().fp64_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& number : tensor.contents().fp64_contents()) {
                    writer.Double(number);
                }
            } else {
                fillTensorDataWithFloatValuesFromRawContents<double>(response_proto, tensor_it, writer);
            }
        } else if (tensor.datatype() == "BYTES") {
            if (seekDataInValField) {
                auto status = checkValField(tensor.contents().bytes_contents_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
                for (auto& bytes : tensor.contents().bytes_contents()) {
                    bytesOutputsBuffer.append(bytes);
                }
            } else {
                bytesOutputsBuffer.append((char*)response_proto.raw_output_contents(tensor_it).data(), response_proto.raw_output_contents(tensor_it).size());
            }
        } else {
            return StatusCode::REST_UNSUPPORTED_PRECISION;
        }
        if (tensor.datatype() != "BYTES") {
            writer.EndArray();
        }
        auto status = parseOutputParameters(tensor, writer, bytesOutputsBuffer.size());
        if (!status.ok()) {
            return status;
        }
        writer.EndObject();
        tensor_it++;
    }
    writer.EndArray();
    return StatusCode::OK;
}

Status makeJsonFromPredictResponse(
    const ::KFSResponse& response_proto,
    std::string* response_json,
    std::optional<int>& inferenceHeaderContentLength) {
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;
    timer.start(CONVERT);

    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
    writer.StartObject();
    writer.Key("model_name");
    writer.String(response_proto.model_name().c_str());
    if (response_proto.id().length() > 0) {
        writer.Key("id");
        writer.String(response_proto.id().c_str());
    }
    if (response_proto.model_version().length() > 0) {
        writer.Key("model_version");
        writer.String(response_proto.model_version().c_str());
    }

    auto status = parseResponseParameters(response_proto, writer);
    if (!status.ok()) {
        return status;
    }

    if (response_proto.outputs_size() == 0) {
        SPDLOG_ERROR("Creating json from tensors failed: No outputs found.");
        return StatusCode::REST_PROTO_TO_STRING_ERROR;
    }

    std::string binaryOutputsBuffer;
    status = parseOutputs(response_proto, writer, binaryOutputsBuffer);
    if (!status.ok()) {
        return status;
    }

    writer.EndObject();
    response_json->assign(buffer.GetString());
    if (binaryOutputsBuffer.size() > 0) {
        inferenceHeaderContentLength = response_json->length();
    }
    response_json->append(binaryOutputsBuffer);

    timer.stop(CONVERT);
    SPDLOG_DEBUG("GRPC to HTTP response conversion: {:.3f} ms", timer.elapsed<microseconds>(CONVERT) / 1000);

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
