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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rest_parser.hpp"

using namespace ovms;

using namespace testing;

class KFSRestParserTest : public Test {
public:
    KFSRestParser parser;
};

TEST_F(KFSRestParserTest, parseValidRequestTwoInputs) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        },
        {
        "name" : "input1",
        "shape" : [ 3 ],
        "datatype" : "BOOL",
        "data" : [ true ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 2);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "UINT32");
    ASSERT_EQ(proto.inputs()[0].contents().uint_contents_size(), 4);
    ASSERT_THAT(proto.inputs()[0].contents().uint_contents(), ElementsAre(1, 2, 3, 4));

    ASSERT_EQ(proto.inputs()[1].name(), "input1");
    ASSERT_THAT(proto.inputs()[1].shape(), ElementsAre(3));
    ASSERT_EQ(proto.inputs()[1].datatype(), "BOOL");
    ASSERT_EQ(proto.inputs()[1].contents().bool_contents_size(), 1);
    ASSERT_THAT(proto.inputs()[1].contents().bool_contents(), ElementsAre(true));
}

#define VALIDATE_INPUT(DATATYPE, CONTENTS_SIZE, CONTENTS)       \
    auto proto = parser.getProto();                             \
    ASSERT_EQ(proto.inputs_size(), 1);                          \
    ASSERT_EQ(proto.inputs()[0].name(), "input0");              \
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));  \
    ASSERT_EQ(proto.inputs()[0].datatype(), DATATYPE);          \
    ASSERT_EQ(proto.inputs()[0].contents().CONTENTS_SIZE(), 4); \
    ASSERT_THAT(proto.inputs()[0].contents().CONTENTS(), ElementsAre(1, 2, 3, 4));

TEST_F(KFSRestParserTest, parseValidRequestUINT64) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT64",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT64", uint64_contents_size, uint64_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestUINT32) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestUINT16) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT16",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT16", uint_contents_size, uint_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestUINT8) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT8",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT8", uint_contents_size, uint_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestINT64) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "INT64",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("INT64", int64_contents_size, int64_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestINT32) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "INT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("INT32", int_contents_size, int_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestINT16) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "INT16",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("INT16", int_contents_size, int_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestINT8) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "INT8",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("INT8", int_contents_size, int_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestFP64) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "FP64",
        "data" : [ 1.1, 2.2, 3.3, 4.4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 1);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(),ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "FP64");
    ASSERT_EQ(proto.inputs()[0].contents().fp64_contents_size(), 4);
    ASSERT_THAT(proto.inputs()[0].contents().fp64_contents(), ElementsAre((float)1.1, (float)2.2, (float)3.3, (float)4.4));
}

TEST_F(KFSRestParserTest, parseValidRequestFP32) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "FP32",
        "data" : [ 1.5, 2.9, 3.0, 4.1 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 1);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "FP32");
    ASSERT_EQ(proto.inputs()[0].contents().fp32_contents_size(), 4);
    ASSERT_THAT(proto.inputs()[0].contents().fp32_contents(), ElementsAre(1.5, 2.9, 3.0, 4.1));
}

TEST_F(KFSRestParserTest, parseValidRequestFP32WithIntegers) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "FP32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("FP32", fp32_contents_size, fp32_contents);
}

TEST_F(KFSRestParserTest, parseValidRequestBOOL) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "BOOL",
        "data" : [ true, true, false, false]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 1);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "BOOL");
    ASSERT_EQ(proto.inputs()[0].contents().bool_contents_size(), 4);
    ASSERT_THAT(proto.inputs()[0].contents().bool_contents(), ElementsAre(true, true, false, false));
}

TEST_F(KFSRestParserTest, parseValidRequestBYTES) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "BYTES",
        "parameters" : {"binary_data_size" : 4}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 1);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "BYTES");
    auto binary_data_size_parameter = proto.inputs()[0].parameters().find("binary_data_size");
    ASSERT_NE(binary_data_size_parameter, proto.inputs()[0].parameters().end());
    ASSERT_EQ(binary_data_size_parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kInt64Param);
    ASSERT_EQ(binary_data_size_parameter->second.int64_param(), 4);
}

TEST_F(KFSRestParserTest, parseValidRequestWithStringRequestParameter) {
    std::string request = R"({
    "parameters" : {"param" : "value"},
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.parameters().find("param");
    ASSERT_NE(parameter, proto.parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kStringParam);
    ASSERT_EQ(parameter->second.string_param(), "value");
}

TEST_F(KFSRestParserTest, parseValidRequestWithIntRequestParameter) {
    std::string request = R"({
    "parameters" : {"param" : 5},
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.parameters().find("param");
    ASSERT_NE(parameter, proto.parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kInt64Param);
    ASSERT_EQ(parameter->second.int64_param(), 5);
}

TEST_F(KFSRestParserTest, parseValidRequestWithBoolRequestParameter) {
    std::string request = R"({
    "parameters" : {"param" : true},
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.parameters().find("param");
    ASSERT_NE(parameter, proto.parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kBoolParam);
    ASSERT_EQ(parameter->second.bool_param(), true);
}

TEST_F(KFSRestParserTest, parseValidRequestWithId) {
    std::string request = R"({
    "id" : "50",
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
    ASSERT_EQ(proto.id(), "50");
}

TEST_F(KFSRestParserTest, parseValidRequestWithOutput) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ],
    "outputs" : [
        {
        "name" : "output0"
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
    ASSERT_EQ(proto.outputs_size(), 1);
    ASSERT_EQ(proto.outputs()[0].name(), "output0");
}

TEST_F(KFSRestParserTest, parseValidRequestWithStringOutputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ],
    "outputs" : [
        {
        "name" : "output0",
        "parameters" : {"param" : "value"}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
    ASSERT_EQ(proto.outputs_size(), 1);
    ASSERT_EQ(proto.outputs()[0].name(), "output0");

    auto parameter = proto.outputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.outputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kStringParam);
    ASSERT_EQ(parameter->second.string_param(), "value");
}

TEST_F(KFSRestParserTest, parseValidRequestWithIntOutputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ],
    "outputs" : [
        {
        "name" : "output0",
        "parameters" : {"param" : 5}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
    ASSERT_EQ(proto.outputs_size(), 1);
    ASSERT_EQ(proto.outputs()[0].name(), "output0");

    auto parameter = proto.outputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.outputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kInt64Param);
    ASSERT_EQ(parameter->second.int64_param(), 5);
}

TEST_F(KFSRestParserTest, parseValidRequestWithBoolOutputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ],
    "outputs" : [
        {
        "name" : "output0",
        "parameters" : {"param" : true}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);
    ASSERT_EQ(proto.outputs_size(), 1);
    ASSERT_EQ(proto.outputs()[0].name(), "output0");

    auto parameter = proto.outputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.outputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kBoolParam);
    ASSERT_EQ(parameter->second.bool_param(), true);
}

TEST_F(KFSRestParserTest, parseValidRequestWithStringInputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ],
        "parameters" : {"param" : "value"}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.inputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.inputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kStringParam);
    ASSERT_EQ(parameter->second.string_param(), "value");
}

TEST_F(KFSRestParserTest, parseValidRequestWithIntInputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ],
        "parameters" : {"param" : 5}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.inputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.inputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kInt64Param);
    ASSERT_EQ(parameter->second.int64_param(), 5);
}

TEST_F(KFSRestParserTest, parseValidRequestWithBoolInputParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ],
        "parameters" : {"param" : true}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    VALIDATE_INPUT("UINT32", uint_contents_size, uint_contents);

    auto parameter = proto.inputs()[0].parameters().find("param");
    ASSERT_NE(parameter, proto.inputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kBoolParam);
    ASSERT_EQ(parameter->second.bool_param(), true);
}

TEST_F(KFSRestParserTest, parseValidRequestWithNoDataButBinaryInputsParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "BYTES",
        "parameters" : {"binary_data_size" : 16}
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::OK);

    auto proto = parser.getProto();
    ASSERT_EQ(proto.inputs_size(), 1);
    ASSERT_EQ(proto.inputs()[0].name(), "input0");
    ASSERT_THAT(proto.inputs()[0].shape(), ElementsAre(2, 2));
    ASSERT_EQ(proto.inputs()[0].datatype(), "BYTES");
    ASSERT_EQ(proto.inputs()[0].contents().bytes_contents_size(), 0);

    auto parameter = proto.inputs()[0].parameters().find("binary_data_size");
    ASSERT_NE(parameter, proto.inputs()[0].parameters().end());
    ASSERT_EQ(parameter->second.parameter_choice_case(), inference::InferParameter::ParameterChoiceCase::kInt64Param);
    ASSERT_EQ(parameter->second.int64_param(), 16);
}

TEST_F(KFSRestParserTest, parseInValidRequestUINT32WithFloatingPointValues) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1.0, 2.0, 3.0, 4.0 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInValidRequestUINT32WithNegativeValues) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, -4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestBoolWithIntData) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "BOOL",
        "data" : [ 0, 1, 0, 1]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithNoDataAndNoBinaryInputsParameter) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32"
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithDataAndBYTESdatatype) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "BYTES",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithNoName) {
    std::string request = R"({
    "inputs" : [
        {
        "shape" : [ 2, 2 ],
        "datatype" : "FP32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithNoShape) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "datatype" : "FP32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithNoDatatype) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestUINT32WithStringData) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ "a", "bc", "d", "ef" ]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestUINT32WithBoolData) {
    std::string request = R"({
    "inputs" : [
        {
        "name" : "input0",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ false, true, false, true]
        }
    ]
    })";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_COULD_NOT_PARSE_INPUT);
}

TEST_F(KFSRestParserTest, parseInvalidRequestWithInputsMissing) {
    std::string request = R"({})";
    auto status = parser.parse(request.c_str());
    ASSERT_EQ(status, StatusCode::REST_NO_INPUTS_FOUND);
}
