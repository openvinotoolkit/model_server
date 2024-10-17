#include <variant>
#include <vector>

#include "rapidjson/document.h"

enum class EncodingFormat {
    FLOAT,
    BASE64
};

struct EmbeddingsRequest {
    std::variant<std::vector<std::string>, std::vector<std::vector<int>>> input;
    EncodingFormat encoding_format;

    static std::variant<EmbeddingsRequest, std::string> from_json(rapidjson::Document* request);
};

struct EmbeddingsResponse {

    static std::string to_json(EmbeddingsResponse response);
};
