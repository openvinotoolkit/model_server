//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "tokenize_parser.hpp"

#include <utility>

#include "src/port/rapidjson_writer.hpp"

namespace ovms {
absl::Status TokenizeParser::parseTokenizeResponse(rapidjson::StringBuffer& buffer, const ov::genai::TokenizedInputs& tokens, const ov::AnyMap& parameters) {
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    writer.String("tokens");
    ov::Shape outputShape = tokens.input_ids.get_shape();
    auto inputIdsTensor = tokens.input_ids;
    auto attentionMaskTensor = tokens.attention_mask;
    auto pad_to_max_length = parameters.find("pad_to_max_length") != parameters.end() ? parameters.at("pad_to_max_length").as<bool>() : false;
    if (outputShape.size() != 2) {
        return absl::InvalidArgumentError("Invalid input ids tensor shape");
    }

    const bool isBatched = outputShape[0] > 1;
    if (isBatched) {
        writer.StartArray();
    }
    for (size_t batchIterator = 0; batchIterator < outputShape[0]; batchIterator++) {
        writer.StartArray();
        size_t size = outputShape[1];
        int64_t* dataPtr = reinterpret_cast<int64_t*>(inputIdsTensor.data()) + batchIterator * size;
        int64_t* attentionMaskPtr = reinterpret_cast<int64_t*>(attentionMaskTensor.data()) + batchIterator * size;
        for (size_t i = 0; i < size; ++i) {
            if (attentionMaskPtr[i] == 0 && !pad_to_max_length) {
                break;
            }
            writer.Int64(dataPtr[i]);
        }
        writer.EndArray();
    }
    if (isBatched) {
        writer.EndArray();
    }
    writer.EndObject();
    return absl::OkStatus();
}

std::variant<TokenizeRequest, std::string> TokenizeParser::validateTokenizeRequest(rapidjson::Document& parsedJson) {
    TokenizeRequest request;
    if (parsedJson.HasParseError()) {
        return "Failed to parse JSON";
    }
    if (!parsedJson.IsObject())
        return "Received json is not an object";

    auto parsedInput = parseInput(parsedJson, "text");

    if (std::holds_alternative<std::string>(parsedInput)) {
        return std::get<std::string>(parsedInput);
    } else {
        auto inputVariant = std::get<TokenizeRequest::InputDataType>(parsedInput);
        if (std::holds_alternative<std::vector<std::string>>(inputVariant)) {
            request.input = std::get<std::vector<std::string>>(inputVariant);
        } else {
            request.input = std::get<std::vector<std::vector<int64_t>>>(inputVariant);
        }
    }

    auto it = parsedJson.FindMember("max_length");
    if (it != parsedJson.MemberEnd()) {
        if (it->value.IsInt()) {
            size_t max_length = it->value.GetInt();
            request.parameters["max_length"] = max_length;
        } else {
            return "max_length should be integer";
        }
    }
    it = parsedJson.FindMember("pad_to_max_length");
    if (it != parsedJson.MemberEnd()) {
        if (it->value.IsBool()) {
            bool pad_to_max_length = it->value.GetBool();
            request.parameters["pad_to_max_length"] = pad_to_max_length;
        } else {
            return "pad_to_max_length should be boolean";
        }
    }
    it = parsedJson.FindMember("add_special_tokens");
    if (it != parsedJson.MemberEnd()) {
        if (it->value.IsBool()) {
            bool add_special_tokens = it->value.GetBool();
            request.parameters["add_special_tokens"] = add_special_tokens;
        } else {
            return "add_special_tokens should be boolean";
        }
    }

    it = parsedJson.FindMember("padding_side");
    if (it != parsedJson.MemberEnd()) {
        if (it->value.IsString()) {
            std::string padding_side = it->value.GetString();
            if (padding_side != "left" && padding_side != "right") {
                return "padding_side should be either left or right";
            }
            request.parameters["padding_side"] = padding_side;
        } else {
            return "padding_side should be string, either left or right";
        }
    }
    return request;
}

std::variant<TokenizeRequest::InputDataType, std::string> TokenizeParser::parseInput(rapidjson::Document& parsedJson, const std::string& field_name) {
    enum class InputType {
        NONE,
        STRING,
        STRING_VEC,
        INT,
        INT_VEC
    };

    std::vector<std::string> input_strings;
    std::vector<std::vector<std::string>> input_string_vectors;
    std::vector<std::vector<int64_t>> input_tokens;

    auto it = parsedJson.FindMember(field_name.c_str());

    if (it != parsedJson.MemberEnd()) {
        if (it->value.IsString()) {
            input_strings.push_back(it->value.GetString());
        } else if (it->value.IsArray()) {
            if (it->value.GetArray().Size() == 0) {
                return field_name + " array should not be empty";
            }
            InputType input_type = InputType::NONE;
            for (auto& input : it->value.GetArray()) {
                if (input.IsArray()) {
                    if (input_type != InputType::NONE && input_type != InputType::INT_VEC && input_type != InputType::STRING_VEC)
                        return field_name + " must be homogeneous";
                    if (input.GetArray()[0].IsInt()) {
                        if (input_type == InputType::STRING_VEC)
                            return field_name + " must be homogeneous";
                        input_type = InputType::INT_VEC;
                        std::vector<int64_t> ints;
                        ints.reserve(input.GetArray().Size());
                        for (auto& val : input.GetArray()) {
                            if (val.IsInt())
                                ints.push_back(val.GetInt());
                            else
                                return field_name + " must be homogeneous";
                        }
                        input_tokens.emplace_back(std::move(ints));
                    } else if (input.GetArray()[0].IsString()) {
                        if (input_type == InputType::INT_VEC)
                            return field_name + " must be homogeneous";
                        input_type = InputType::STRING_VEC;
                        std::vector<std::string> strings;
                        strings.reserve(input.GetArray().Size());
                        for (auto& val : input.GetArray()) {
                            if (val.IsString())
                                strings.push_back(val.GetString());
                            else
                                return field_name + " must be homogeneous";
                        }
                        input_string_vectors.emplace_back(std::move(strings));
                    } else {
                        return "every element in " + field_name + " array should be either string or int";
                    }
                } else if (input.IsString()) {
                    if (input_type != InputType::NONE && input_type != InputType::STRING)
                        return field_name + " must be homogeneous";
                    input_type = InputType::STRING;
                    input_strings.push_back(input.GetString());
                } else if (input.IsInt()) {
                    if (input_type != InputType::NONE && input_type != InputType::INT)
                        return field_name + " must be homogeneous";
                    input_type = InputType::INT;
                    if (input_tokens.size() == 0) {
                        input_tokens.push_back(std::vector<int64_t>());
                    }
                    input_tokens[0].push_back(input.GetInt());
                } else {
                    return "every element in " + field_name + " array should be either string or int";
                }
            }
        } else {
            return field_name + " should be string, array of strings or array of integers";
        }
    } else {
        return field_name + " field is required";
    }

    if (input_strings.size() > 0) {
        return input_strings;
    } else if (input_tokens.size() > 0) {
        return input_tokens;
    } else if (input_string_vectors.size() > 0) {
        return input_string_vectors;
    } else {
        return field_name + " field is required";
    }
}

absl::Status TokenizeParser::parseTokenizeRequest(rapidjson::Document& parsedJson, TokenizeRequest& request) {
    auto validated = TokenizeParser::validateTokenizeRequest(parsedJson);
    if (auto error = std::get_if<std::string>(&validated)) {
        return absl::InvalidArgumentError(*error);
    }
    request = std::get<TokenizeRequest>(validated);
    return absl::OkStatus();
}
}  // namespace ovms
