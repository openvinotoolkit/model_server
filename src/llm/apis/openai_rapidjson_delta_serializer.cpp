// Copyright 2026 Intel Corporation
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

#include <string>

#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "src/llm/apis/openai_rapidjson_delta_serializer.hpp"

namespace ovms {

std::string RapidJsonDeltaSerializer::serialize(const ContentDelta& d) const {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    w.StartObject();
    w.Key("delta");
    w.StartObject();
    w.Key("content");
    w.String(d.text.c_str(), static_cast<rapidjson::SizeType>(d.text.size()));
    w.EndObject();
    w.EndObject();
    return buf.GetString();
}

std::string RapidJsonDeltaSerializer::serialize(const ReasoningDelta& d) const {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    w.StartObject();
    w.Key("delta");
    w.StartObject();
    w.Key("reasoning_content");
    w.String(d.text.c_str(), static_cast<rapidjson::SizeType>(d.text.size()));
    w.EndObject();
    w.EndObject();
    return buf.GetString();
}

std::string RapidJsonDeltaSerializer::serialize(const ToolCallDelta& d) const {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    w.StartObject();
    w.Key("delta");
    w.StartObject();
    w.Key("tool_calls");
    w.StartArray();
    w.StartObject();
    if (d.id) {
        w.Key("id");
        w.String(d.id->c_str(), static_cast<rapidjson::SizeType>(d.id->size()));
        w.Key("type");
        w.String("function");
    }
    w.Key("index");
    w.Int(d.index);
    w.Key("function");
    w.StartObject();
    if (d.name) {
        w.Key("name");
        w.String(d.name->c_str(), static_cast<rapidjson::SizeType>(d.name->size()));
    }
    if (!d.arguments.empty()) {
        w.Key("arguments");
        w.String(d.arguments.c_str(), static_cast<rapidjson::SizeType>(d.arguments.size()));
    }
    w.EndObject();
    w.EndObject();
    w.EndArray();
    w.EndObject();
    w.EndObject();
    return buf.GetString();
}

std::string RapidJsonDeltaSerializer::serialize(const FinishDelta&) const {
    return "{}";
}

std::string RapidJsonDeltaSerializer::serialize(const AudioDelta&) const {
    return "{}";
}

}  // namespace ovms
