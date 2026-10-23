//*****************************************************************************
// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.
//*****************************************************************************

#pragma once

#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rapidjson/document.h>

namespace ovms {

// Historical Gemma fix is JSONSchema(schema, 2): at most two whitespace
// characters between JSON tokens. The 2026.4 GenAI DLL compiles structural
// tags from its own JSONSchema::to_json and does not read max_whitespace_cnt,
// so a raw field on the tag is ignored and the model can fill the open call
// with newlines. The bound below is part of the EBNF text that DLL compiles.
inline constexpr int kGemma4JsonMaxWhitespace = 2;

namespace {

class Gemma4BoundedJsonEmitter {
    std::vector<std::string> rules;
    int serial = 0;
    bool primitives = false;
    std::string anyName;
    int objectSerial = 0;

    std::string fresh(const char* prefix) {
        return std::string(prefix) + std::to_string(serial++);
    }

    void define(const std::string& name, const std::string& body) {
        rules.push_back(name + " ::= " + body);
    }

    static std::string ebnfQuote(const std::string& raw) {
        std::string out = "\"";
        for (unsigned char c : raw) {
            if (c == '\\' || c == '"')
                out.push_back('\\');
            out.push_back(static_cast<char>(c));
        }
        out.push_back('"');
        return out;
    }

    static std::string lit(const std::string& raw) {
        return ebnfQuote(raw);
    }

    static std::string jsonString(const std::string& raw) {
        std::string out = "\"";
        for (unsigned char c : raw) {
            if (c == '\\' || c == '"')
                out.push_back('\\');
            out.push_back(static_cast<char>(c));
        }
        out.push_back('"');
        return out;
    }

    static void rejectStructuralKeyword(const rapidjson::Value& schema) {
        static const char* blocked[] = {
            "$ref", "allOf", "not", "pattern", "patternProperties",
            "dependentSchemas", "dependentRequired", "if", "then", "else",
            "prefixItems", "unevaluatedProperties", "unevaluatedItems"};
        for (const char* key : blocked) {
            if (schema.HasMember(key))
                throw std::invalid_argument(std::string("Gemma4 bounded JSON grammar does not implement schema keyword ") + key);
        }
    }

    void ensurePrimitives() {
        if (primitives)
            return;
        primitives = true;
        define("g4ws", std::string("[ \\n\\t]{0,") + std::to_string(kGemma4JsonMaxWhitespace) + "}");
        define("g4str", std::string(R"("\"")") + " g4chars");
        define("g4chars", std::string(R"("\"")") + " | g4one g4chars");
        define("g4one", std::string(R"([^"\\])") + " | " + R"("\\")" + " g4esc");
        define("g4esc", std::string(R"(["\\/bfnrt])") + " | " + R"("u")" + " [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f]");
        define("g4int", R"ebnf("-"? ("0" | [1-9] [0-9]*))ebnf");
        define("g4num", R"ebnf(g4int ("." [0-9]+)? ([eE] [+-]? [0-9]+)?)ebnf");
        define("g4bool", R"("true" | "false")");
        define("g4null", R"("null")");
    }

    std::string ws() {
        ensurePrimitives();
        return "g4ws";
    }

    std::string anyValue() {
        ensurePrimitives();
        if (!anyName.empty())
            return anyName;
        anyName = "g4any";
        define("g4any", "g4str | g4num | g4bool | g4null | g4anyobj | g4anyarr");
        define("g4anyobj", lit("{") + " " + ws() + " g4anybody");
        define("g4anybody", lit("}") + " | g4str " + ws() + " " + lit(":") + " " + ws() + " g4any g4anytail");
        define("g4anytail", ws() + " " + lit("}") + " | " + ws() + " " + lit(",") + " " + ws() + " g4str " + ws() + " " + lit(":") + " " + ws() + " g4any g4anytail");
        define("g4anyarr", lit("[") + " " + ws() + " g4anyarrb");
        define("g4anyarrb", lit("]") + " | g4any g4anyarrt");
        define("g4anyarrt", ws() + " " + lit("]") + " | " + ws() + " " + lit(",") + " " + ws() + " g4any g4anyarrt");
        return anyName;
    }

    std::string unionOf(const rapidjson::Value& branches) {
        if (!branches.IsArray() || branches.Empty())
            throw std::invalid_argument("Gemma4 bounded JSON grammar requires a non-empty anyOf or oneOf array");
        std::string body;
        for (rapidjson::SizeType i = 0; i < branches.Size(); ++i) {
            if (i)
                body += " | ";
            body += valueOf(branches[i]);
        }
        const std::string name = fresh("g4or");
        define(name, body);
        return name;
    }

    std::string enumOf(const rapidjson::Value& choices) {
        if (!choices.IsArray() || choices.Empty())
            throw std::invalid_argument("Gemma4 bounded JSON grammar requires a non-empty enum array");
        std::string body;
        for (rapidjson::SizeType i = 0; i < choices.Size(); ++i) {
            if (i)
                body += " | ";
            const auto& choice = choices[i];
            if (choice.IsString())
                body += lit(jsonString(choice.GetString()));
            else if (choice.IsBool())
                body += lit(choice.GetBool() ? "true" : "false");
            else if (choice.IsInt64())
                body += lit(std::to_string(choice.GetInt64()));
            else if (choice.IsUint64())
                body += lit(std::to_string(choice.GetUint64()));
            else if (choice.IsDouble())
                body += lit(std::to_string(choice.GetDouble()));
            else if (choice.IsNull())
                body += lit("null");
            else
                throw std::invalid_argument("Gemma4 bounded JSON grammar enum value must be a scalar");
        }
        const std::string name = fresh("g4en");
        define(name, body);
        return name;
    }

    std::string arrayOf(const rapidjson::Value* items) {
        const std::string item = items ? valueOf(*items) : anyValue();
        const std::string body = fresh("g4ab");
        const std::string tail = fresh("g4at");
        const std::string name = fresh("g4arr");
        define(name, lit("[") + " " + ws() + " " + body);
        define(body, lit("]") + " | " + item + " " + tail);
        define(tail, ws() + " " + lit("]") + " | " + ws() + " " + lit(",") + " " + ws() + " " + item + " " + tail);
        return name;
    }

    std::string objectOf(const rapidjson::Value& schema) {
        if (schema.HasMember("properties") && !schema["properties"].IsObject())
            throw std::invalid_argument("Gemma4 bounded JSON grammar properties must be an object");
        if (!schema.HasMember("properties")) {
            anyValue();
            return "g4anyobj";
        }
        const auto& properties = schema["properties"];
        std::vector<std::pair<std::string, bool>> fields;
        for (auto it = properties.MemberBegin(); it != properties.MemberEnd(); ++it)
            fields.emplace_back(it->name.GetString(), false);
        if (schema.HasMember("required")) {
            if (!schema["required"].IsArray())
                throw std::invalid_argument("Gemma4 bounded JSON grammar required must be an array");
            for (const auto& name : schema["required"].GetArray()) {
                if (!name.IsString())
                    throw std::invalid_argument("Gemma4 bounded JSON grammar required entries must be strings");
                bool found = false;
                for (auto& field : fields) {
                    if (field.first == name.GetString()) {
                        field.second = true;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw std::invalid_argument(std::string("Gemma4 bounded JSON grammar required property is missing: ") + name.GetString());
            }
        }
        if (fields.empty())
            return lit("{") + " " + ws() + " " + lit("}");

        const int objId = objectSerial++;
        std::map<std::string, std::string> memo;
        const auto chain = [&](auto&& self, size_t index, bool needComma) -> std::string {
            const std::string key = std::to_string(objId) + ":" + std::to_string(index) + ":" + (needComma ? "1" : "0");
            const auto existing = memo.find(key);
            if (existing != memo.end())
                return existing->second;
            const std::string name = fresh("g4c");
            memo.emplace(key, name);
            if (index == fields.size()) {
                define(name, lit("}"));
                return name;
            }
            const auto& field = fields[index];
            const std::string pair = lit(jsonString(field.first)) + " " + ws() + " " + lit(":") + " " + ws() + " " + valueOf(properties[field.first.c_str()]);
            const std::string taken = self(self, index + 1, true);
            std::string take = pair + " " + ws() + " " + taken;
            if (needComma)
                take = lit(",") + " " + ws() + " " + take;
            if (field.second)
                define(name, take);
            else
                define(name, take + " | " + self(self, index + 1, needComma));
            return name;
        };
        const std::string name = fresh("g4obj");
        define(name, lit("{") + " " + ws() + " " + chain(chain, 0, false));
        return name;
    }

    std::string valueOf(const rapidjson::Value& schema) {
        if (!schema.IsObject())
            throw std::invalid_argument("Gemma4 bounded JSON grammar schema node must be an object");
        rejectStructuralKeyword(schema);
        if (schema.HasMember("enum"))
            return enumOf(schema["enum"]);
        if (schema.HasMember("anyOf"))
            return unionOf(schema["anyOf"]);
        if (schema.HasMember("oneOf"))
            return unionOf(schema["oneOf"]);
        if (schema.HasMember("const")) {
            const auto& held = schema["const"];
            if (held.IsString())
                return lit(jsonString(held.GetString()));
            if (held.IsBool())
                return lit(held.GetBool() ? "true" : "false");
            if (held.IsNull())
                return lit("null");
            throw std::invalid_argument("Gemma4 bounded JSON grammar const must be a string, boolean, or null");
        }
        if (!schema.HasMember("type"))
            return anyValue();
        const auto& type = schema["type"];
        if (type.IsArray()) {
            rapidjson::Document branches;
            branches.SetArray();
            for (const auto& item : type.GetArray()) {
                if (!item.IsString())
                    throw std::invalid_argument("Gemma4 bounded JSON grammar type union entries must be strings");
                rapidjson::Value node(rapidjson::kObjectType);
                node.AddMember("type", rapidjson::Value(item.GetString(), branches.GetAllocator()), branches.GetAllocator());
                branches.PushBack(node, branches.GetAllocator());
            }
            return unionOf(branches);
        }
        if (!type.IsString())
            throw std::invalid_argument("Gemma4 bounded JSON grammar type must be a string or array of strings");
        const std::string kind = type.GetString();
        ensurePrimitives();
        if (kind == "string")
            return "g4str";
        if (kind == "integer")
            return "g4int";
        if (kind == "number")
            return "g4num";
        if (kind == "boolean")
            return "g4bool";
        if (kind == "null")
            return "g4null";
        if (kind == "array") {
            if (schema.HasMember("items") && schema["items"].IsArray())
                throw std::invalid_argument("Gemma4 bounded JSON grammar does not implement tuple items");
            const rapidjson::Value* items = schema.HasMember("items") ? &schema["items"] : nullptr;
            return arrayOf(items);
        }
        if (kind == "object")
            return objectOf(schema);
        throw std::invalid_argument("Gemma4 bounded JSON grammar does not implement type " + kind);
    }

public:
    std::string emit(const rapidjson::Value& schema) {
        const std::string root = valueOf(schema);
        std::ostringstream out;
        out << "root ::= " << root << "\n";
        for (const auto& rule : rules)
            out << rule << "\n";
        return out.str();
    }
};

}  // namespace

inline std::string gemma4BoundedJsonGrammar(const std::string& schemaJson) {
    rapidjson::Document schema;
    if (schema.Parse(schemaJson.c_str()).HasParseError() || !schema.IsObject())
        throw std::invalid_argument("Gemma4 tool schema is not a JSON object");
    return Gemma4BoundedJsonEmitter().emit(schema);
}

}  // namespace ovms
