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

#include <optional>
#include <set>
#include <sstream>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)
#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386 6011 4457 6308 6387 6246)
#include "absl/strings/escaping.h"
#pragma warning(pop)
#include "kfs_frontend/kfs_utils.hpp"
#include "precision.hpp"
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "status.hpp"
#include "timer.hpp"

namespace {
enum : unsigned int {
    CONVERT,
    TIMER_END
};
}

namespace ovms {

static Status checkValField(size_t valFieldSize, size_t expectedElementsNumber) {
    if (valFieldSize != expectedElementsNumber) {
        std::stringstream ss;
        ss << "Expected val field elements number: " << expectedElementsNumber << "; actual: " << valFieldSize;
        return Status(StatusCode::REST_SERIALIZE_VAL_FIELD_INVALID_SIZE, ss.str());
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

static Status parseOutputParameters(const inference::ModelInferResponse_InferOutputTensor& output, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer, int binaryOutputSize) {
    if (output.parameters_size() > 0 || binaryOutputSize > 0) {
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
        if (binaryOutputSize > 0) {
            writer.Key("binary_data_size");
            writer.Int(binaryOutputSize);
        }
        writer.EndObject();
    }

    return StatusCode::OK;
}

static void appendBinaryOutput(std::string& bytesOutputsBuffer, char* output, size_t outputSize) {
    bytesOutputsBuffer.append(output, outputSize);
}

#define PARSE_OUTPUT_DATA(CONTENTS_FIELD, DATATYPE, WRITER_TYPE)                                                                      \
    if (seekDataInValField) {                                                                                                         \
        auto status = checkValField(tensor.contents().CONTENTS_FIELD##_size(), expectedElementsNumber);                               \
        if (!status.ok())                                                                                                             \
            return status;                                                                                                            \
        if (binaryOutput) {                                                                                                           \
            appendBinaryOutput(bytesOutputsBuffer, (char*)tensor.contents().CONTENTS_FIELD().data(), expectedContentSize);            \
        } else {                                                                                                                      \
            for (auto& number : tensor.contents().CONTENTS_FIELD()) {                                                                 \
                writer.WRITER_TYPE(number);                                                                                           \
            }                                                                                                                         \
        }                                                                                                                             \
    } else {                                                                                                                          \
        if (binaryOutput) {                                                                                                           \
            appendBinaryOutput(bytesOutputsBuffer, (char*)response_proto.raw_output_contents(tensor_it).data(), expectedContentSize); \
        } else {                                                                                                                      \
            for (size_t i = 0; i < response_proto.raw_output_contents(tensor_it).size(); i += sizeof(DATATYPE))                       \
                writer.WRITER_TYPE(*(reinterpret_cast<const DATATYPE*>(response_proto.raw_output_contents(tensor_it).data() + i)));   \
        }                                                                                                                             \
    }

#define PARSE_OUTPUT_DATA_STRING(CONTENTS_FIELD, WRITER_TYPE)                                                                                                          \
    expectedContentSize = 0;                                                                                                                                           \
    if (seekDataInValField) {                                                                                                                                          \
        if (binaryOutput) {                                                                                                                                            \
            for (auto& sentence : tensor.contents().CONTENTS_FIELD()) {                                                                                                \
                uint32_t length = static_cast<uint32_t>(                                                                                                               \
                    sentence.size());                                                                                                                                  \
                expectedContentSize += length + 4;                                                                                                                     \
                appendBinaryOutput(bytesOutputsBuffer,                                                                                                                 \
                    (char*)&length, sizeof(length));                                                                                                                   \
                appendBinaryOutput(                                                                                                                                    \
                    bytesOutputsBuffer,                                                                                                                                \
                    (char*)sentence.data(),                                                                                                                            \
                    length);                                                                                                                                           \
            }                                                                                                                                                          \
        } else {                                                                                                                                                       \
            for (auto& sentence : tensor.contents().CONTENTS_FIELD()) {                                                                                                \
                writer.WRITER_TYPE(sentence.data(), sentence.size());                                                                                                  \
            }                                                                                                                                                          \
        }                                                                                                                                                              \
    } else {                                                                                                                                                           \
        if (binaryOutput) {                                                                                                                                            \
            expectedContentSize += response_proto.raw_output_contents(tensor_it).size();                                                                               \
            appendBinaryOutput(bytesOutputsBuffer, (char*)response_proto.raw_output_contents(tensor_it).data(), response_proto.raw_output_contents(tensor_it).size()); \
        } else {                                                                                                                                                       \
            size_t i = 0;                                                                                                                                              \
            while (i < response_proto.raw_output_contents(tensor_it).size()) {                                                                                         \
                uint32_t length = *(reinterpret_cast<const uint32_t*>(response_proto.raw_output_contents(tensor_it).data() + i));                                      \
                i += sizeof(length);                                                                                                                                   \
                if (i + length > response_proto.raw_output_contents(tensor_it).size())                                                                                 \
                    return StatusCode::INTERNAL_ERROR;                                                                                                                 \
                writer.WRITER_TYPE(response_proto.raw_output_contents(tensor_it).data() + i, length);                                                                  \
                i += length;                                                                                                                                           \
            }                                                                                                                                                          \
        }                                                                                                                                                              \
    }

static Status parseOutputs(const ::KFSResponse& response_proto, rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer, std::string& bytesOutputsBuffer, const std::set<std::string>& binaryOutputsNames) {
    writer.Key("outputs");
    writer.StartArray();

    bool seekDataInValField = false;
    if (response_proto.raw_output_contents_size() == 0)
        seekDataInValField = true;
    int tensor_it = 0;
    for (const auto& tensor : response_proto.outputs()) {
        size_t dataTypeSize = KFSDataTypeSize(tensor.datatype());
        // expected size calculated for static types
        // for BYTES the calculation is dynamic inside PARSE_OUTPUT_DATA_EX since all strings can be of different length
        size_t expectedContentSize = dataTypeSize;
        for (int i = 0; i < tensor.shape().size(); i++) {
            expectedContentSize *= tensor.shape().at(i);
        }
        size_t expectedElementsNumber = dataTypeSize > 0 ? expectedContentSize / dataTypeSize : 0;

        if (!seekDataInValField && (tensor.datatype() != "BYTES" && response_proto.raw_output_contents(tensor_it).size() != expectedContentSize)) {
            std::stringstream ss;
            ss << "Expected raw output content size: " << expectedContentSize << "; actual: " << response_proto.raw_output_contents(tensor_it).size();
            return Status(StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, ss.str());
        }
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
        bool binaryOutput = ((binaryOutputsNames.find(tensor.name().c_str()) != binaryOutputsNames.end()));
        if (!binaryOutput) {
            writer.Key("data");
            writer.StartArray();
        }
        if (tensor.datatype() == "FP32") {
            PARSE_OUTPUT_DATA(fp32_contents, float, Double)
        } else if (tensor.datatype() == "INT32") {
            PARSE_OUTPUT_DATA(int_contents, int32_t, Int)
        } else if (tensor.datatype() == "INT16") {
            PARSE_OUTPUT_DATA(int_contents, int16_t, Int)
        } else if (tensor.datatype() == "INT8") {
            PARSE_OUTPUT_DATA(int_contents, int8_t, Int)
        } else if (tensor.datatype() == "UINT32") {
            PARSE_OUTPUT_DATA(uint_contents, uint32_t, Uint)
        } else if (tensor.datatype() == "UINT16") {
            PARSE_OUTPUT_DATA(uint_contents, uint16_t, Uint)
        } else if (tensor.datatype() == "UINT8") {
            PARSE_OUTPUT_DATA(uint_contents, uint8_t, Uint)
        } else if (tensor.datatype() == "FP64") {
            PARSE_OUTPUT_DATA(fp64_contents, double, Double)
        } else if (tensor.datatype() == "INT64") {
            PARSE_OUTPUT_DATA(int64_contents, int64_t, Int64)
        } else if (tensor.datatype() == "UINT64") {
            PARSE_OUTPUT_DATA(uint64_contents, uint64_t, Uint64)
        } else if (tensor.datatype() == "BOOL") {
            PARSE_OUTPUT_DATA(bool_contents, bool, Bool)
        } else if (tensor.datatype() == "BYTES") {
            PARSE_OUTPUT_DATA_STRING(bytes_contents, String)
        } else {
            std::stringstream ss;
            ss << "Unsupported precision" << tensor.datatype();
            return Status(StatusCode::REST_UNSUPPORTED_PRECISION, ss.str());
        }
        if (!binaryOutput) {
            writer.EndArray();
        }
        auto status = parseOutputParameters(tensor, writer, binaryOutput ? expectedContentSize : 0);
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
    std::optional<int>& inferenceHeaderContentLength,
    const std::set<std::string>& requestedBinaryOutputsNames) {
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
    status = parseOutputs(response_proto, writer, binaryOutputsBuffer, requestedBinaryOutputsNames);
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
    return absl::Base64Unescape(bytes, &decodedBytes) ? StatusCode::OK : StatusCode::REST_BASE64_DECODE_ERROR;
}
}  // namespace ovms
