#include "tokenize_parser.hpp"

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/writer.h>
#pragma warning(pop)

namespace ovms {
absl::Status TokenizeParser::parseTokenizeResponse(rapidjson::StringBuffer& buffer, const ov::Tensor& inputIdsTensor) {
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    writer.String("tokens");
    ov::Shape outputShape = inputIdsTensor.get_shape();
    if (outputShape.size() != 2) {
        return absl::InvalidArgumentError("Invalid input ids tensor shape");
    }
    writer.StartArray();
    for (size_t batchIterator = 0; batchIterator < outputShape[0]; batchIterator++) {
        size_t size = outputShape[1];
        int64_t* dataPtr = reinterpret_cast<int64_t*>(inputIdsTensor.data()) + batchIterator * size;
        for (size_t i = 0; i < size; ++i) {
            writer.Int64(dataPtr[i]);
        }
    }
    writer.EndArray();
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
        INT,
        INT_VEC
    };

    std::vector<std::string> input_strings;
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
                    if (input_type != InputType::NONE && input_type != InputType::INT_VEC)
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
