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

#include <string>

#include "logging.hpp"
#include "status.hpp"
#include "utils/rapidjson_utils.hpp"

namespace ovms {

static constexpr int MAX_NESTING_DEPTH = 100;

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
    if (!node.IsArray()) {
        return StatusCode::REST_COULD_NOT_PARSE_INPUT;
    }
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
        if (dim.GetInt() < 0) {
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
        if (std::strcmp(datatypeItr->value.GetString(), "FP16") == 0 || std::strcmp(datatypeItr->value.GetString(), "BF16") == 0) {
            SPDLOG_DEBUG("{} datatype is supported only when data is located in raw_input_contents", datatypeItr->value.GetString());
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
    auto status = parseJsonWithDepthLimit(doc, json, MAX_NESTING_DEPTH);
    if (!status.ok()) {
        SPDLOG_DEBUG("Request is not a valid JSON: {}", status.string());
        return status;
    }
    if (!doc.IsObject()) {
        SPDLOG_DEBUG("Request body is not an object");
        return StatusCode::REST_BODY_IS_NOT_AN_OBJECT;
    }
    auto idItr = doc.FindMember("id");
    if (idItr != doc.MemberEnd()) {
        status = parseId(idItr->value);
        if (!status.ok()) {
            SPDLOG_DEBUG("Parsing request ID failed");
            return status;
        }
    }

    auto parametersItr = doc.FindMember("parameters");
    if (parametersItr != doc.MemberEnd()) {
        status = parseRequestParameters(parametersItr->value);
        if (!status.ok()) {
            SPDLOG_DEBUG("Parsing request parameters failed");
            return status;
        }
    }

    auto outputsItr = doc.FindMember("outputs");
    if (outputsItr != doc.MemberEnd()) {
        status = parseOutputs(outputsItr->value);
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
    status = parseInputs(inputsItr->value);
    if (!status.ok()) {
        SPDLOG_DEBUG("Parsing request inputs failed");
        return status;
    }

    return StatusCode::OK;
}

}  // namespace ovms
