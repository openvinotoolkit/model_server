#pragma once

#include <string>
#include <variant>
#include <vector>

#include <openvino/runtime/tensor.hpp>

#pragma warning(push)
#pragma warning(disable : 6001 6385 6386 6011 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6001 6385 6386)
#include "absl/strings/escaping.h"
#pragma warning(pop)

namespace ovms {

struct TokenizeRequest {
    using InputDataType = std::variant<std::vector<std::string>, std::vector<std::vector<int64_t>>>;
    InputDataType input;
    ov::AnyMap parameters = {};
};

class TokenizeParser {
public:
    static std::variant<TokenizeRequest::InputDataType, std::string> parseInput(rapidjson::Document& parsedJson, const std::string& field_name);
    static absl::Status parseTokenizeResponse(rapidjson::StringBuffer& buffer, const ov::Tensor& inputIdsTensor);
    static absl::Status parseTokenizeRequest(rapidjson::Document& parsedJson, TokenizeRequest& request);
    static std::variant<TokenizeRequest, std::string> validateTokenizeRequest(rapidjson::Document& parsedJson);
};
}  // namespace ovms
