// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "../../apis/tool_schema_wrapper.hpp"
#include "../qwen3coder/tool_schema_profile.hpp"

namespace ovms::lfm2 {
// Native Python-call generation uses JSON string/value literals, which are an
// accepted subset of our native literal reader. It never evaluates Python.
class ToolGrammar {
    std::vector<std::string> rules;
    unsigned serial = 0;
    unsigned nodeCount = 0;

    [[noreturn]] static void fail(const std::string& why) {
        throw std::invalid_argument("LFM hard tool grammar: " + why);
    }
    std::string fresh() {
        if (++serial > 8192) fail("schema produces too many grammar rules");
        return "lfmv" + std::to_string(serial);
    }
    void define(const std::string& name, const std::string& body) {
        rules.push_back(name + " ::= " + body);
    }
    static std::string quote(const std::string& value) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
        return {buffer.GetString(), buffer.GetSize()};
    }
    static std::string jsonValue(const rapidjson::Value& value) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        value.Accept(writer);
        return {buffer.GetString(), buffer.GetSize()};
    }
    static bool identifier(const std::string& text) {
        if (text.empty() || text.size() > 256) return false;
        auto alpha = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
        if (!alpha(text[0])) return false;
        for (unsigned char c : text)
            if (!alpha(c) && !(c >= '0' && c <= '9')) return false;
        return true;
    }
    static bool toolName(const std::string& text) {
        return identifier(text);
    }
    static void checkKeys(const rapidjson::Value& schema) {
        static const std::set<std::string> allowed{
            "type", "properties", "required", "additionalProperties", "items", "enum", "const",
            "description", "title", "default", "examples", "$comment"};
        std::set<std::string> seen;
        for (auto it = schema.MemberBegin(); it != schema.MemberEnd(); ++it) {
            const std::string key(it->name.GetString(), it->name.GetStringLength());
            if (!seen.insert(key).second) fail("duplicate schema keyword: " + key);
            if (!allowed.count(key)) fail("unsupported schema keyword: " + key);
        }
    }
    static bool scalarMatches(const rapidjson::Value& value, const std::string& type) {
        return (type == "string" && value.IsString()) || (type == "boolean" && value.IsBool()) ||
               (type == "null" && value.IsNull()) || (type == "number" && value.IsNumber()) ||
               (type == "integer" && (value.IsInt64() || value.IsUint64()));
    }
    std::string object(const rapidjson::Value& schema, unsigned depth, bool arguments) {
        if (schema.HasMember("items")) fail("items is only supported on arrays");
        if (schema.HasMember("additionalProperties") && !schema["additionalProperties"].IsBool())
            fail("additionalProperties must be boolean");
        if (schema.HasMember("properties") && !schema["properties"].IsObject()) fail("properties must be an object");
        if (schema.HasMember("required") && !schema["required"].IsArray()) fail("required must be an array");
        std::set<std::string> required;
        if (schema.HasMember("required")) {
            for (const auto& value : schema["required"].GetArray()) {
                if (!value.IsString()) fail("required entries must be strings");
                if (!required.insert({value.GetString(), value.GetStringLength()}).second) fail("duplicate required entry");
            }
        }
        struct Field { std::string name; std::string grammar; bool required; };
        std::vector<Field> fields;
        std::set<std::string> names;
        if (schema.HasMember("properties")) {
            for (auto it = schema["properties"].MemberBegin(); it != schema["properties"].MemberEnd(); ++it) {
                const std::string name(it->name.GetString(), it->name.GetStringLength());
                if (!names.insert(name).second) fail("duplicate property name");
                if (arguments && !identifier(name)) fail("argument name is not a native keyword identifier: " + name);
                fields.push_back({name, value(it->value, depth + 1), required.count(name) != 0});
                if (fields.size() > 128) fail("more than 128 object fields");
            }
        }
        for (const auto& name : required)
            if (!names.count(name)) fail("required property lacks a schema: " + name);
        // Emit a canonical-order subset. Optional fields have explicit skip
        // branches, so commas are never required for an omitted predecessor.
        std::map<std::pair<size_t, bool>, std::string> memo;
        const auto tail = [&](auto&& self, size_t i, bool comma) -> std::string {
            const auto key = std::make_pair(i, comma);
            const auto prior = memo.find(key);
            if (prior != memo.end()) return prior->second;
            const auto name = fresh();
            memo.emplace(key, name);
            if (i == fields.size()) {
                define(name, quote(arguments ? ")" : "}"));
                return name;
            }
            const auto& field = fields[i];
            std::string body = comma ? quote(",") + " lfmws " : "";
            body += quote(arguments ? field.name : quote(field.name)) + " lfmws " +
                    quote(arguments ? "=" : ":") + " lfmws " + field.grammar + " lfmws " + self(self, i + 1, true);
            if (!field.required) body += " | " + self(self, i + 1, comma);
            define(name, body);
            return name;
        };
        return quote(arguments ? "(" : "{") + " lfmws " + tail(tail, 0, false);
    }
    std::string value(const rapidjson::Value& schema, unsigned depth) {
        if (depth > 32 || ++nodeCount > 2048) fail("schema complexity limit exceeded");
        if (!schema.IsObject()) fail("schema node must be an object");
        checkKeys(schema);
        std::vector<std::string> types;
        if (schema.HasMember("type")) {
            const auto& type = schema["type"];
            if (type.IsString()) types.emplace_back(type.GetString(), type.GetStringLength());
            else if (type.IsArray() && !type.Empty()) {
                for (const auto& entry : type.GetArray()) {
                    if (!entry.IsString()) fail("type union entry must be a string");
                    types.emplace_back(entry.GetString(), entry.GetStringLength());
                }
            } else fail("type must be a string or nonempty string array");
        } else if (schema.HasMember("properties") || schema.HasMember("required")) types.emplace_back("object");
        if (schema.HasMember("enum") || schema.HasMember("const")) {
            if (schema.HasMember("enum") && schema.HasMember("const")) fail("combined enum and const are unsupported");
            const auto literal = [&](const rapidjson::Value& candidate) {
                if (candidate.IsArray() || candidate.IsObject()) fail("enum and const must be scalar");
                if (!types.empty() && std::none_of(types.begin(), types.end(), [&](const auto& type) { return scalarMatches(candidate, type); }))
                    fail("enum or const value contradicts type");
                return quote(jsonValue(candidate));
            };
            std::string body;
            if (schema.HasMember("const")) body = literal(schema["const"]);
            else {
                if (!schema["enum"].IsArray() || schema["enum"].Empty() || schema["enum"].Size() > 256) fail("enum must have 1..256 values");
                for (const auto& candidate : schema["enum"].GetArray()) {
                    if (!body.empty()) body += " | ";
                    body += literal(candidate);
                }
            }
            const auto name = fresh();
            define(name, body);
            return name;
        }
        if (types.empty()) fail("value schema needs an explicit supported type");
        std::string branches;
        for (const auto& type : types) {
            if (!branches.empty()) branches += " | ";
            if (type != "object" && (schema.HasMember("properties") || schema.HasMember("required") || schema.HasMember("additionalProperties")))
                fail("object keywords on non-object types are unsupported");
            if (type != "array" && schema.HasMember("items")) fail("items on non-array type is unsupported");
            if (type == "string") branches += "lfmstr";
            else if (type == "integer") branches += "lfmint";
            else if (type == "number") branches += "lfmnum";
            else if (type == "boolean") branches += "lfmbool";
            else if (type == "null") branches += quote("null");
            else if (type == "object") branches += object(schema, depth, false);
            else if (type == "array") {
                if (!schema.HasMember("items") || !schema["items"].IsObject()) fail("array items needs a single schema object");
                const auto item = value(schema["items"], depth + 1);
                branches += quote("[") + " lfmws (" + item + " (lfmws " + quote(",") + " lfmws " + item + ")*)? lfmws " + quote("]");
            } else fail("unsupported type: " + type);
        }
        const auto name = fresh();
        define(name, branches);
        return name;
    }
public:
    std::string build(const ToolsSchemas_t& tools, bool parallel) {
        if (tools.size() > 128) fail("more than 128 tools");
        std::string calls;
        for (const auto& [name, wrapper] : tools) {
            if (!toolName(name)) fail("unsafe tool name: " + name);
            if (wrapper.stringRepr.size() > 1024 * 1024) fail("schema exceeds 1 MiB");
            // Use the retainer's existing read-only Draft 4 profile before the
            // narrower native grammar compiler. A grammar-only accepted schema
            // must never be rejected at publication for profile drift.
            if (!isSupportedQwenXmlToolSchema(wrapper.stringRepr))
                fail("schema is outside the executable retainer profile: " + name);
            rapidjson::Document schema;
            if (schema.Parse(wrapper.stringRepr.c_str()).HasParseError() || !schema.IsObject()) fail("tool schema is not a JSON object");
            checkKeys(schema);
            if (schema.HasMember("type") && (!schema["type"].IsString() || std::string(schema["type"].GetString()) != "object")) fail("tool arguments must be an object");
            if (schema.HasMember("enum") || schema.HasMember("const")) fail("enum/const on root argument object is unsupported");
            if (!calls.empty()) calls += " | ";
            calls += quote(name) + " " + object(schema, 0, true);
        }
        if (calls.empty()) fail("at least one tool is required");
        std::string result = "root ::= " + quote("<|tool_call_start|>[") + " lfmws lfmcall";
        if (parallel) result += " (lfmws " + quote(",") + " lfmws lfmcall)*";
        result += " lfmws " + quote("]<|tool_call_end|>") + "\n";
        result += "lfmcall ::= " + calls + "\n";
        result += R"ebnf(lfmws ::= [ \n\t\r]{0,2}
lfmstr ::= "\"" lfmchars
lfmchars ::= "\"" | lfmchar lfmchars
lfmchar ::= [^"\\\x00-\x1f] | "\\" lfmesc
lfmesc ::= ["\\/bfnrt] | "u" lfmunicode
lfmhex ::= [0-9a-fA-F]
lfmunicode ::= [1-9a-cA-Ce-fE-F] lfmhex lfmhex lfmhex | "0" ([1-9a-fA-F] lfmhex lfmhex | "0" ([1-9a-fA-F] lfmhex | "0" [1-9a-fA-F])) | [dD] [0-7] lfmhex lfmhex | [dD] [89aAbB] lfmhex lfmhex "\\u" [dD] [c-fC-F] lfmhex lfmhex
lfmint ::= "-"? ("0" | [1-9] [0-9]*)
lfmnum ::= lfmint ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
lfmbool ::= "true" | "false"
)ebnf";
        for (const auto& rule : rules) result += rule + "\n";
        return result;
    }
};
inline std::string toolGrammar(const ToolsSchemas_t& tools, bool parallel) {
    return ToolGrammar{}.build(tools, parallel);
}
}  // namespace ovms::lfm2
