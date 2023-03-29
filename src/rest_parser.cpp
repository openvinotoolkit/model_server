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
#include "rest_parser.hpp"

#include <functional>
#include <string>

#include "precision.hpp"
#include "rest_utils.hpp"
#include "status.hpp"
#include "tfs_frontend/tfs_utils.hpp"

namespace ovms {

TFSRestParser::TFSRestParser(const tensor_map_t& tensors) {
    for (const auto& kv : tensors) {
        const auto& name = kv.first;
        const auto& tensor = kv.second;
        tensorPrecisionMap[name] = tensor->getPrecision();
        auto& input = (*requestProto.mutable_inputs())[name];
        input.set_dtype(getPrecisionAsDataType(tensor->getPrecision()));

        auto fold = [](size_t a, const Dimension& b) {
            if (b.isDynamic()) {
                if (b.isAny()) {
                    return static_cast<size_t>(0);
                } else {
                    return static_cast<size_t>(b.getMaxValue());
                }
            } else {
                return a * static_cast<size_t>(b.getStaticValue());
            }
        };
        input.mutable_tensor_content()->reserve(std::accumulate(
                                                    tensor->getShape().cbegin(),
                                                    tensor->getShape().cend(),
                                                    1,
                                                    fold) *
                                                DataTypeSize(getPrecisionAsDataType(tensor->getPrecision())));
    }
}

void TFSRestParser::removeUnusedInputs() {
    auto& inputs = (*requestProto.mutable_inputs());
    auto it = inputs.begin();
    while (it != inputs.end()) {
        if (!it->second.tensor_shape().dim_size()) {
            SPDLOG_DEBUG("Removing {} input from proto since it's not included in the request", it->first);
            it = inputs.erase(it);
        } else {
            it++;
        }
    }
}

bool TFSRestParser::parseSequenceIdInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName) {
    proto.set_dtype(tensorflow::DataType::DT_UINT64);
    for (auto& value : doc.GetArray()) {
        if (value.IsUint64())
            proto.add_uint64_val(value.GetUint64());
        else
            return false;
    }
    return true;
}

bool TFSRestParser::parseSequenceControlInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName) {
    proto.set_dtype(tensorflow::DataType::DT_UINT32);
    for (auto& value : doc.GetArray()) {
        if (value.IsUint())
            proto.add_uint32_val(value.GetUint());
        else
            return false;
    }
    return true;
}

bool TFSRestParser::parseSpecialInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName) {
    // Special tensors are given in 1 dimentional array
    if (doc.GetArray()[0].IsArray())
        return false;

    if (tensorName == "sequence_id")
        return parseSequenceIdInput(doc, proto, tensorName);
    else if (tensorName == "sequence_control_input")
        return parseSequenceControlInput(doc, proto, tensorName);

    return false;
}

static bool isBinary(const rapidjson::Value& value) {
    if (!value.IsObject()) {
        return false;
    }

    if (!(value.HasMember("b64") && ((value.MemberEnd() - value.MemberBegin()) == 1))) {
        return false;
    }

    if (!value["b64"].IsString()) {
        return false;
    }

    return true;
}

bool TFSRestParser::parseArray(rapidjson::Value& doc, int dim, tensorflow::TensorProto& proto, const std::string& tensorName) {
    if (isBinary(doc)) {
        if (!addValue(proto, doc)) {
            return false;
        }
        return true;
    }
    if (doc.IsString() && tensorPrecisionMap[tensorName] == ovms::Precision::U8 && (proto.dtype() == tensorflow::DataType::DT_UINT8 || proto.dtype() == tensorflow::DataType::DT_STRING)) {
        if (!addValue(proto, doc)) {
            return false;
        }
        return true;
    }
    if (!doc.IsArray()) {
        return false;
    }
    if (doc.GetArray().Size() == 0) {
        return false;
    }
    if (!setDimOrValidate(proto, dim, doc.GetArray().Size())) {
        return false;
    }
    if (tensorName == "sequence_id" || tensorName == "sequence_control_input") {
        if (!parseSpecialInput(doc, proto, tensorName)) {
            return false;
        }
        return true;
    }
    if (doc.GetArray()[0].IsArray()) {
        for (auto& itr : doc.GetArray()) {
            if (!parseArray(itr, dim + 1, proto, tensorName)) {
                return false;
            }
        }
        return true;
    } else {
        if (!setDTypeIfNotSet(doc.GetArray()[0], proto, tensorName)) {
            return false;
        }
        for (auto& value : doc.GetArray()) {
            if (!addValue(proto, value)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool TFSRestParser::parseInstance(rapidjson::Value& doc) {
    if (doc.GetObject().MemberCount() == 0) {
        return false;
    }
    for (auto& itr : doc.GetObject()) {
        std::string tensorName = itr.name.GetString();
        auto& proto = (*requestProto.mutable_inputs())[tensorName];
        increaseBatchSize(proto);
        if (!parseArray(itr.value, 1, proto, tensorName)) {
            return false;
        }
    }
    return true;
}

bool TFSRestParser::isBatchSizeEqualForAllInputs() const {
    int64_t size = 0;
    for (const auto& kv : requestProto.inputs()) {
        if (size == 0) {
            size = kv.second.tensor_shape().dim(0).size();
        } else if (kv.second.tensor_shape().dim(0).size() != size) {
            return false;
        }
    }
    return true;
}

Status TFSRestParser::parseRowFormat(rapidjson::Value& node) {
    order = Order::ROW;
    if (!node.IsArray()) {
        return StatusCode::REST_INSTANCES_NOT_AN_ARRAY;
    }
    if (node.GetArray().Size() == 0) {
        return StatusCode::REST_NO_INSTANCES_FOUND;
    }
    if (node.GetArray()[0].IsObject() && !isBinary(node.GetArray()[0])) {
        // named format
        for (auto& instance : node.GetArray()) {
            if (!instance.IsObject()) {
                return StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT;
            }

            if (!this->parseInstance(instance)) {
                return StatusCode::REST_COULD_NOT_PARSE_INSTANCE;
            }
        }
    } else if (node.GetArray()[0].IsArray() || node.GetArray()[0].IsNumber() || isBinary(node.GetArray()[0]) || node.GetArray()[0].IsString()) {
        // no named format
        if (requestProto.inputs_size() != 1) {
            return StatusCode::REST_INPUT_NOT_PREALLOCATED;
        }
        auto inputsIterator = requestProto.mutable_inputs()->begin();
        if (inputsIterator == requestProto.mutable_inputs()->end()) {
            const std::string details = "Failed to parse row formatted request.";
            SPDLOG_ERROR("Internal error occured: {}", details);
            return Status(StatusCode::INTERNAL_ERROR, details);
        }
        if (!parseArray(node, 0, inputsIterator->second, inputsIterator->first)) {
            return StatusCode::REST_COULD_NOT_PARSE_INSTANCE;
        } else {
            format = Format::NONAMED;
            return StatusCode::OK;
        }
    } else {
        return StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED;
    }
    removeUnusedInputs();
    if (!isBatchSizeEqualForAllInputs()) {
        return StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER;
    }
    format = Format::NAMED;
    return StatusCode::OK;
}

Status TFSRestParser::parseColumnFormat(rapidjson::Value& node) {
    order = Order::COLUMN;
    // no named format
    if (node.IsArray()) {
        if (requestProto.inputs_size() != 1) {
            return StatusCode::REST_INPUT_NOT_PREALLOCATED;
        }
        auto inputsIterator = requestProto.mutable_inputs()->begin();
        if (inputsIterator == requestProto.mutable_inputs()->end()) {
            const std::string details = "Failed to parse column formatted request.";
            SPDLOG_ERROR("Internal error occured: {}", details);
            return Status(StatusCode::INTERNAL_ERROR, details);
        }
        if (!parseArray(node, 0, inputsIterator->second, inputsIterator->first)) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
        format = Format::NONAMED;
        return StatusCode::OK;
    }
    // named format
    if (!node.IsObject()) {
        return StatusCode::REST_INPUTS_NOT_AN_OBJECT;
    }
    if (node.GetObject().MemberCount() == 0) {
        return StatusCode::REST_NO_INPUTS_FOUND;
    }
    for (auto& kv : node.GetObject()) {
        std::string tensorName = kv.name.GetString();
        auto& proto = (*requestProto.mutable_inputs())[tensorName];
        if (!parseArray(kv.value, 0, proto, tensorName)) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
    }
    removeUnusedInputs();
    format = Format::NAMED;
    return StatusCode::OK;
}

Status TFSRestParser::parse(const char* json) {
    rapidjson::Document doc;
    if (doc.Parse(json).HasParseError()) {
        return StatusCode::JSON_INVALID;
    }
    if (!doc.IsObject()) {
        return StatusCode::REST_BODY_IS_NOT_AN_OBJECT;
    }
    auto instancesItr = doc.FindMember("instances");
    auto inputsItr = doc.FindMember("inputs");
    if (instancesItr != doc.MemberEnd() && inputsItr != doc.MemberEnd()) {
        return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
    }
    if (instancesItr != doc.MemberEnd()) {
        return parseRowFormat(instancesItr->value);
    }
    if (inputsItr != doc.MemberEnd()) {
        return parseColumnFormat(inputsItr->value);
    }
    return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
}

void TFSRestParser::increaseBatchSize(tensorflow::TensorProto& proto) {
    if (proto.tensor_shape().dim_size() < 1) {
        proto.mutable_tensor_shape()->add_dim()->set_size(0);
    }
    proto.mutable_tensor_shape()->mutable_dim(0)->set_size(proto.tensor_shape().dim(0).size() + 1);
}

bool TFSRestParser::setDimOrValidate(tensorflow::TensorProto& proto, int dim, int size) {
    if (proto.tensor_shape().dim_size() > dim) {
        return proto.tensor_shape().dim(dim).size() == size;
    } else {
        while (proto.tensor_shape().dim_size() <= dim) {
            proto.mutable_tensor_shape()->add_dim()->set_size(0);
        }
        proto.mutable_tensor_shape()->mutable_dim(dim)->set_size(size);
        return true;
    }
}

static bool getB64FromValue(const rapidjson::Value& value, std::string& b64Val) {
    if (!isBinary(value)) {
        return false;
    }

    b64Val = value["b64"].GetString();
    return true;
}

template <typename T>
static bool addToTensorContent(tensorflow::TensorProto& proto, T value) {
    if (sizeof(T) != DataTypeSize(proto.dtype())) {
        return false;
    }
    proto.mutable_tensor_content()->append(reinterpret_cast<const char*>(&value), sizeof(T));
    return true;
}

template <typename T>
static bool addToTensorContent(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetDouble()));
    }
    if (value.IsInt64()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetInt64()));
    }
    if (value.IsUint64()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetUint64()));
    }
    if (value.IsInt()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetInt()));
    }
    if (value.IsUint()) {
        return addToTensorContent<T>(proto, static_cast<T>(value.GetUint()));
    }

    return false;
}

static bool addToHalfVal(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        proto.add_half_val(value.GetDouble());
        return true;
    }
    if (value.IsInt64()) {
        proto.add_half_val(value.GetInt64());
        return true;
    }
    if (value.IsUint64()) {
        proto.add_half_val(value.GetUint64());
        return true;
    }
    if (value.IsInt()) {
        proto.add_half_val(value.GetInt());
        return true;
    }
    if (value.IsUint()) {
        proto.add_half_val(value.GetUint());
        return true;
    }

    return false;
}

static bool addToIntVal(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (value.IsDouble()) {
        proto.add_int_val(value.GetDouble());
        return true;
    }
    if (value.IsInt64()) {
        proto.add_int_val(value.GetInt64());
        return true;
    }
    if (value.IsUint64()) {
        proto.add_int_val(value.GetUint64());
        return true;
    }
    if (value.IsInt()) {
        proto.add_int_val(value.GetInt());
        return true;
    }
    if (value.IsUint()) {
        proto.add_int_val(value.GetUint());
        return true;
    }

    return false;
}

bool TFSRestParser::addValue(tensorflow::TensorProto& proto, const rapidjson::Value& value) {
    if (isBinary(value)) {
        std::string b64Val;
        if (!getB64FromValue(value, b64Val))
            return false;
        std::string decodedBytes;
        if (decodeBase64(b64Val, decodedBytes) == StatusCode::OK) {
            proto.add_string_val(decodedBytes.c_str(), decodedBytes.length());
            proto.set_dtype(tensorflow::DataType::DT_STRING);
            return true;
        } else {
            return false;
        }
    }
    if (value.IsString() && (proto.dtype() == tensorflow::DataType::DT_UINT8 || proto.dtype() == tensorflow::DataType::DT_STRING)) {
        proto.add_string_val(value.GetString(), strlen(value.GetString()));
        proto.set_dtype(tensorflow::DataType::DT_STRING);
        return true;
    }

    if (!value.IsNumber()) {
        return false;
    }

    switch (proto.dtype()) {
    case tensorflow::DataType::DT_FLOAT:
        return addToTensorContent<float>(proto, value);
    case tensorflow::DataType::DT_INT32:
        return addToTensorContent<int32_t>(proto, value);
    case tensorflow::DataType::DT_INT8:
        return addToTensorContent<int8_t>(proto, value);
    case tensorflow::DataType::DT_UINT8:
        return addToTensorContent<uint8_t>(proto, value);
    case tensorflow::DataType::DT_DOUBLE:
        return addToTensorContent<double>(proto, value);
    case tensorflow::DataType::DT_HALF:
        return addToHalfVal(proto, value);
    case tensorflow::DataType::DT_INT16:
        return addToTensorContent<int16_t>(proto, value);
    case tensorflow::DataType::DT_UINT16:
        return addToIntVal(proto, value);
    case tensorflow::DataType::DT_INT64:
        return addToTensorContent<int64_t>(proto, value);
    case tensorflow::DataType::DT_UINT32:
        return addToTensorContent<uint32_t>(proto, value);
    case tensorflow::DataType::DT_UINT64:
        return addToTensorContent<uint64_t>(proto, value);
    default:
        return false;
    }
    return false;
}

// This is still required for parsing inputs which are not present in model/DAG.
// Such inputs are then removed from proto at the end of parsing phase.
bool TFSRestParser::setDTypeIfNotSet(const rapidjson::Value& value, tensorflow::TensorProto& proto, const std::string& tensorName) {
    if (tensorPrecisionMap.count(tensorName))
        return true;

    if (value.IsInt())
        tensorPrecisionMap[tensorName] = ovms::Precision::I32;
    else if (value.IsDouble())
        tensorPrecisionMap[tensorName] = ovms::Precision::FP32;
    else
        return false;

    proto.set_dtype(getPrecisionAsDataType(tensorPrecisionMap[tensorName]));
    return true;
}

Status KFSRestParser::parseId(rapidjson::Value& node) {
    if (!node.IsString()) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    requestProto.set_id(node.GetString());
    return StatusCode::OK;
}

#define PARSE_PARAMETER(PROTO)                                                                                \
    if (!node.IsObject()) {                                                                                   \
        return StatusCode::REST_COULD_NOT_PARSE_PARAMETERS;                                                   \
    }                                                                                                         \
                                                                                                              \
    for (auto& parameter : node.GetObject()) {                                                                \
        if (!parameter.name.IsString()) {                                                                     \
            return StatusCode::REST_COULD_NOT_PARSE_PARAMETERS;                                               \
        }                                                                                                     \
                                                                                                              \
        if (parameter.value.IsString()) {                                                                     \
            auto requestParameters = PROTO.mutable_parameters();                                              \
            ((*requestParameters)[parameter.name.GetString()]).set_string_param(parameter.value.GetString()); \
        } else if (parameter.value.IsBool()) {                                                                \
            auto requestParameters = PROTO.mutable_parameters();                                              \
            ((*requestParameters)[parameter.name.GetString()]).set_bool_param(parameter.value.GetBool());     \
        } else if (parameter.value.IsInt()) {                                                                 \
            auto requestParameters = PROTO.mutable_parameters();                                              \
            ((*requestParameters)[parameter.name.GetString()]).set_int64_param(parameter.value.GetInt());     \
        } else {                                                                                              \
            return StatusCode::REST_COULD_NOT_PARSE_PARAMETERS;                                               \
        }                                                                                                     \
    }                                                                                                         \
    return StatusCode::OK;

Status KFSRestParser::parseRequestParameters(rapidjson::Value& node){
    PARSE_PARAMETER(requestProto)}

Status KFSRestParser::parseInputParameters(rapidjson::Value& node, ::KFSRequest::InferInputTensor& input){
    PARSE_PARAMETER(input)}

Status KFSRestParser::parseOutputParameters(rapidjson::Value& node, ::KFSRequest::InferRequestedOutputTensor& output){
    PARSE_PARAMETER(output)}

Status KFSRestParser::parseOutput(rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::REST_COULD_NOT_PARSE_OUTPUT;
    }
    requestProto.mutable_outputs()->Clear();
    auto output = requestProto.add_outputs();
    auto nameItr = node.FindMember("name");
    if ((nameItr == node.MemberEnd()) || !(nameItr->value.IsString())) {
        return StatusCode::REST_COULD_NOT_PARSE_OUTPUT;
    }
    output->set_name(nameItr->value.GetString());

    auto parametersItr = node.FindMember("parameters");
    if (parametersItr != node.MemberEnd()) {
        auto status = parseOutputParameters(parametersItr->value, *output);
        if (!status.ok()) {
            return status;
        }
    }
    return StatusCode::OK;
}

Status KFSRestParser::parseOutputs(rapidjson::Value& node) {
    if (!node.IsArray()) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    for (auto& output : node.GetArray()) {
        auto status = parseOutput(output);
        if (!status.ok()) {
            return status;
        }
    }
    return StatusCode::OK;
}

#define HANDLE_VALUE(CONTENTS, TYPE_GETTER, TYPE_CHECK)                 \
    for (auto& value : node.GetArray()) {                               \
        if (value.IsArray()) {                                          \
            auto status = parseData(value, input);                      \
            if (!status.ok()) {                                         \
                return status;                                          \
            }                                                           \
            continue;                                                   \
        }                                                               \
        if (!value.TYPE_CHECK()) {                                      \
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;              \
        }                                                               \
        input.mutable_contents()->CONTENTS()->Add(value.TYPE_GETTER()); \
    }

Status KFSRestParser::parseData(rapidjson::Value& node, ::KFSRequest::InferInputTensor& input) {
    if (input.datatype() == "FP32") {
        HANDLE_VALUE(mutable_fp32_contents, GetFloat, IsNumber)
    } else if (input.datatype() == "INT64") {
        HANDLE_VALUE(mutable_int64_contents, GetInt64, IsInt)
    } else if (input.datatype() == "INT32") {
        HANDLE_VALUE(mutable_int_contents, GetInt, IsInt)
    } else if (input.datatype() == "INT16") {
        HANDLE_VALUE(mutable_int_contents, GetInt, IsInt)
    } else if (input.datatype() == "INT8") {
        HANDLE_VALUE(mutable_int_contents, GetInt, IsInt)
    } else if (input.datatype() == "UINT64") {
        HANDLE_VALUE(mutable_uint64_contents, GetUint64, IsUint)
    } else if (input.datatype() == "UINT32") {
        HANDLE_VALUE(mutable_uint_contents, GetUint, IsUint)
    } else if (input.datatype() == "UINT16") {
        HANDLE_VALUE(mutable_uint_contents, GetUint, IsUint)
    } else if (input.datatype() == "UINT8") {
        HANDLE_VALUE(mutable_uint_contents, GetUint, IsUint)
    } else if (input.datatype() == "FP64") {
        HANDLE_VALUE(mutable_fp64_contents, GetFloat, IsNumber)
    } else if (input.datatype() == "BOOL") {
        HANDLE_VALUE(mutable_bool_contents, GetBool, IsBool)
    } else if (input.datatype() == "BYTES") {
        for (auto& value : node.GetArray()) {
            if (value.IsArray()) {
                auto status = parseData(value, input);
                if (!status.ok()) {
                    return status;
                }
                continue;
            }
            if (value.IsString()) {
                input.mutable_contents()->add_bytes_contents(value.GetString());
            } else {
                SPDLOG_DEBUG("BYTES datatype used in REST request, but data contains non string JSON values");
                return StatusCode::REST_COULD_NOT_PARSE_INPUT;
            }
        }
    } else {
        return StatusCode::REST_UNSUPPORTED_PRECISION;
    }
    return StatusCode::OK;
}

static Status binaryDataSizeCanBeCalculated(::KFSRequest::InferInputTensor& input, bool onlyOneInput) {
    if (input.datatype() == "BYTES" && (!onlyOneInput || input.shape_size() != 1 || input.shape()[0] != 1)) {
        SPDLOG_DEBUG("Tensor: {} with datatype BYTES has no binary_data_size parameter and the size of the data cannot be calculated from shape.", input.name());
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    return StatusCode::OK;
}

Status KFSRestParser::parseInput(rapidjson::Value& node, bool onlyOneInput) {
    if (!node.IsObject()) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }

    auto input = requestProto.add_inputs();
    auto nameItr = node.FindMember("name");
    if ((nameItr == node.MemberEnd()) || !(nameItr->value.IsString())) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    input->set_name(nameItr->value.GetString());

    auto shapeItr = node.FindMember("shape");
    if ((shapeItr == node.MemberEnd()) || !(shapeItr->value.IsArray())) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    for (auto& dim : shapeItr->value.GetArray()) {
        if (!dim.IsInt()) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
        if (dim.GetInt() <= 0) {
            SPDLOG_DEBUG("Shape dimension is invalid: {}", dim.GetInt());
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
        input->mutable_shape()->Add(dim.GetInt());
    }

    auto datatypeItr = node.FindMember("datatype");
    if ((datatypeItr == node.MemberEnd()) || !(datatypeItr->value.IsString())) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    input->set_datatype(datatypeItr->value.GetString());

    auto parametersItr = node.FindMember("parameters");
    if (parametersItr != node.MemberEnd()) {
        auto status = parseInputParameters(parametersItr->value, *input);
        if (!status.ok()) {
            return status;
        }
    }

    auto dataItr = node.FindMember("data");
    if ((dataItr != node.MemberEnd())) {
        if (!(dataItr->value.IsArray())) {
            return StatusCode::REST_COULD_NOT_PARSE_INPUT;
        }
        return parseData(dataItr->value, *input);
    } else {
        auto binary_data_size_parameter = input->parameters().find("binary_data_size");
        if (binary_data_size_parameter != input->parameters().end()) {
            return StatusCode::OK;
        }
        return binaryDataSizeCanBeCalculated(*input, onlyOneInput);
    }
}

Status KFSRestParser::parseInputs(rapidjson::Value& node) {
    if (!node.IsArray()) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
    if (node.GetArray().Size() == 0) {
        return StatusCode::REST_NO_INPUTS_FOUND;
    }
    requestProto.mutable_inputs()->Clear();
    for (auto& input : node.GetArray()) {
        auto status = parseInput(input, (node.GetArray().Size() == 1));
        if (!status.ok()) {
            return status;
        }
    }
    return StatusCode::OK;
}

Status KFSRestParser::parse(const char* json) {
    rapidjson::Document doc;
    if (doc.Parse(json).HasParseError()) {
        SPDLOG_DEBUG("Request parsing is not a valid JSON");
        return StatusCode::JSON_INVALID;
    }
    if (!doc.IsObject()) {
        SPDLOG_DEBUG("Request body is not an object");
        return StatusCode::REST_BODY_IS_NOT_AN_OBJECT;
    }
    auto idItr = doc.FindMember("id");
    if (idItr != doc.MemberEnd()) {
        auto status = parseId(idItr->value);
        if (!status.ok()) {
            SPDLOG_DEBUG("Parsing request ID failed");
            return status;
        }
    }

    auto parametersItr = doc.FindMember("parameters");
    if (parametersItr != doc.MemberEnd()) {
        auto status = parseRequestParameters(parametersItr->value);
        if (!status.ok()) {
            SPDLOG_DEBUG("Parsing request parameters failed");
            return status;
        }
    }

    auto outputsItr = doc.FindMember("outputs");
    if (outputsItr != doc.MemberEnd()) {
        auto status = parseOutputs(outputsItr->value);
        if (!status.ok()) {
            SPDLOG_DEBUG("Parsing request outputs failed");
            return status;
        }
    }

    auto inputsItr = doc.FindMember("inputs");
    if (inputsItr == doc.MemberEnd()) {
        SPDLOG_DEBUG("No inputs found in request");
        return StatusCode::REST_NO_INPUTS_FOUND;
    }
    auto status = parseInputs(inputsItr->value);
    if (!status.ok()) {
        SPDLOG_DEBUG("Parsing request inputs failed");
        return status;
    }

    return StatusCode::OK;
}

}  // namespace ovms
