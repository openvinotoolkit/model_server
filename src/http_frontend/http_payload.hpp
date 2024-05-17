#include <map>
#include <string>
#include <vector>

#include <rapidjson/document.h>

namespace ovms {
struct HttpPayload {
    HttpPayload(const HttpPayload& src) {
        headers = src.headers;
        body = src.body;
        doc.CopyFrom(src.doc, doc.GetAllocator());
    }
    HttpPayload() = default;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;         // always
    rapidjson::Document doc;  // pre-parsed body             = null
};

struct LLMdata {
    std::string prompt;
    std::map<std::string, std::string> params;
};
}  // namespace ovms