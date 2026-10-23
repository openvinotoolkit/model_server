//*****************************************************************************
// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************
#include "tool_schema_profile.hpp"

#include <cstring>
#include <unordered_set>

#pragma warning(push)
// Match the existing Qwen restricted-schema compiler settings for MSVC.
#pragma warning(disable : 5232 6313)
#include "rapidjson/schema.h"
#pragma warning(pop)

#include "draft4_meta_schema.hpp"

namespace ovms {
namespace {
bool safeValidationInput(const rapidjson::Value& value, bool compilerInput) {
    if (value.IsString())
        return std::memchr(value.GetString(), '\0', value.GetStringLength()) == nullptr;
    if (value.IsArray()) {
        for (const auto& item : value.GetArray())
            if (!safeValidationInput(item, compilerInput)) return false;
    } else if (value.IsObject()) {
        std::unordered_set<std::string> names;
        for (const auto& member : value.GetObject()) {
            if (!safeValidationInput(member.name, compilerInput)) return false;
            if (!names.emplace(member.name.GetString(), member.name.GetStringLength()).second) return false;
            // RapidJSON recursively compiles schema-map containers too, and
            // assumes this member is object-valued even in a dependency map.
            if (compilerInput && std::strcmp(member.name.GetString(), "patternProperties") == 0 && !member.value.IsObject())
                return false;
            if (!safeValidationInput(member.value, compilerInput)) return false;
        }
    }
    return true;
}

void removeSchemaDefaults(rapidjson::Value& schema) {
    if (!schema.IsObject()) return;
    // Defaults are annotations, never a substitute for required values. Only
    // visit schema-valued locations; a property *named* default must survive.
    while (schema.HasMember("default")) schema.RemoveMember("default");
    for (const auto* key : {"properties", "patternProperties", "definitions", "dependencies"}) {
        if (schema.HasMember(key) && schema[key].IsObject())
            for (auto& member : schema[key].GetObject())
                removeSchemaDefaults(member.value);
    }
    for (const auto* key : {"additionalProperties", "additionalItems", "not"})
        if (schema.HasMember(key)) removeSchemaDefaults(schema[key]);
    for (const auto* key : {"items", "allOf", "anyOf", "oneOf"}) {
        if (!schema.HasMember(key)) continue;
        auto& child = schema[key];
        if (child.IsArray()) {
            for (auto& item : child.GetArray()) removeSchemaDefaults(item);
        } else {
            removeSchemaDefaults(child);
        }
    }
}
}  // namespace


namespace {
// Compile the same restricted Draft 4 schema used for executable output.
// Per-request compiled-schema reuse is a separate measured optimization.
bool verifyQwenXmlToolProfile(const std::string& schemaText, const std::string* argumentsText) {
    if (schemaText.empty() || schemaText.find('\0') != std::string::npos) return false;
    rapidjson::Document schema;
    schema.Parse(schemaText.c_str());
    if (schema.HasParseError() || !schema.IsObject()) return false;
    if (!safeValidationInput(schema, false)) return false;
    static const rapidjson::SchemaDocument metaSchema = []() {
        rapidjson::Document meta;
        meta.Parse(QWEN_DRAFT4_META_SCHEMA);
        auto& allocator = meta.GetAllocator();
        meta.AddMember("additionalProperties", false, allocator);
        // Empty required lists are common in native tool schemas, including
        // legitimate zero-parameter tools; treat them as no required fields.
        meta["definitions"]["stringArray"]["minItems"].SetInt(0);
        // Format assertions and newer dialect keywords are not silently ignored.
        meta["properties"].RemoveMember("format");
        // This RapidJSON version recursively compiles annotation/enum objects
        // as schemas too. Only scalar data can safely reach that compiler.
        rapidjson::Value scalar(rapidjson::kObjectType);
        rapidjson::Value scalarTypes(rapidjson::kArrayType);
        for (const auto* type : {"null", "boolean", "number", "string"})
            scalarTypes.PushBack(rapidjson::Value(type, allocator), allocator);
        scalar.AddMember("type", scalarTypes, allocator);
        meta["properties"]["default"].CopyFrom(scalar, allocator);
        meta["properties"]["enum"].AddMember("items", scalar, allocator);
        rapidjson::Value ref(rapidjson::kObjectType);
        ref.AddMember("type", "string", allocator);
        // Definitions are schema-valued and meta-validated. An arbitrary local
        // pointer could instead reach unchecked data or a schema-map container.
        // Restrict identifiers to exclude URI-decoded extra path segments.
        ref.AddMember("pattern", "^#/definitions/[A-Za-z0-9_.-]+$", allocator);
        meta["properties"].AddMember("$ref", ref, allocator);
        rapidjson::Value dialects(rapidjson::kArrayType);
        dialects.PushBack("http://json-schema.org/draft-04/schema#", allocator);
        dialects.PushBack("https://json-schema.org/draft-04/schema#", allocator);
        meta["properties"]["$schema"].AddMember("enum", dialects, allocator);
        return rapidjson::SchemaDocument(meta);
    }();
    rapidjson::SchemaValidator schemaValidator(metaSchema);
    if (!schema.Accept(schemaValidator)) return false;
    removeSchemaDefaults(schema);
    if (!safeValidationInput(schema, true)) return false;
    rapidjson::SchemaDocument compiledSchema(schema);
    if (!compiledSchema.IsSupportedSpecification() || !compiledSchema.GetError().ObjectEmpty())
        return false;
    if (!argumentsText) return true;

    if (argumentsText->find('\0') != std::string::npos) return false;
    rapidjson::Document args;
    args.Parse(argumentsText->c_str());
    if (args.HasParseError() || !args.IsObject() || !safeValidationInput(args, false))
        return false;
    rapidjson::SchemaValidator argumentsValidator(compiledSchema);
    return args.Accept(argumentsValidator);
}
}  // namespace

bool isSupportedQwenXmlToolSchema(const std::string& schemaText) {
    return verifyQwenXmlToolProfile(schemaText, nullptr);
}

bool validateQwenXmlToolArguments(const std::string& schemaText, const std::string& argumentsText) {
    return verifyQwenXmlToolProfile(schemaText, &argumentsText);
}
}  // namespace ovms
