// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovms::lfm2 {

// A literal subset shared by the LFM Python-style and JSON-style templates.
// Output is constructed from validated tokens, never from evaluated expressions
// or a numeric DOM round trip. All failures discard the entire argument object.
class LiteralReader {
    static constexpr size_t maxBytes = 1024 * 1024;
    static constexpr size_t maxDepth = 64;
    static constexpr size_t maxEntries = 4096;
    std::string_view input;
    size_t position = 0;

    [[noreturn]] void fail() const {
        throw std::invalid_argument("Invalid LFM literal at byte " + std::to_string(position));
    }
    static bool digit(char c) { return c >= '0' && c <= '9'; }
    static bool alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
    void whitespace() {
        while (position < input.size() && (input[position] == ' ' || input[position] == '\t' || input[position] == '\r' || input[position] == '\n')) ++position;
    }
    bool take(char c) {
        whitespace();
        if (position < input.size() && input[position] == c) { ++position; return true; }
        return false;
    }
    void require(char c) { if (!take(c)) fail(); }
    static std::string quoted(const std::string& value) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteValidateEncodingFlag> writer(buffer);
        if (!writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size())))
            throw std::invalid_argument("Invalid UTF-8 in LFM string");
        return std::string(buffer.GetString(), buffer.GetSize());
    }
    uint32_t hexadecimal(size_t count) {
        uint32_t value = 0;
        for (size_t i = 0; i < count; ++i) {
            if (position == input.size()) fail();
            const char c = input[position++];
            value <<= 4;
            if (digit(c)) value += c - '0';
            else if (c >= 'a' && c <= 'f') value += c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') value += c - 'A' + 10;
            else fail();
        }
        return value;
    }
    void appendCodepoint(std::string& value, uint32_t cp) {
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (position + 2 > input.size() || input.substr(position, 2) != "\\u") fail();
            position += 2;
            const uint32_t low = hexadecimal(4);
            if (low < 0xDC00 || low > 0xDFFF) fail();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
        }
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) fail();
        if (cp <= 0x7F) value.push_back(static_cast<char>(cp));
        else if (cp <= 0x7FF) {
            value.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            value.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            value.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            value.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            value.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            value.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    std::string string() {
        whitespace();
        if (position == input.size() || (input[position] != '\'' && input[position] != '"')) fail();
        const char quote = input[position++];
        std::string value;
        while (position < input.size()) {
            const unsigned char c = static_cast<unsigned char>(input[position++]);
            if (c == quote) return value;
            if (c < 0x20) fail();
            if (c != '\\') { value.push_back(static_cast<char>(c)); continue; }
            if (position == input.size()) fail();
            switch (input[position++]) {
            case '\\': value += '\\'; break;
            case '\'': value += '\''; break;
            case '"': value += '"'; break;
            case '/': value += '/'; break;
            case 'a': value += '\a'; break;
            case 'b': value += '\b'; break;
            case 'f': value += '\f'; break;
            case 'n': value += '\n'; break;
            case 'r': value += '\r'; break;
            case 't': value += '\t'; break;
            case 'v': value += '\v'; break;
            case 'x': appendCodepoint(value, hexadecimal(2)); break;
            case 'u': appendCodepoint(value, hexadecimal(4)); break;
            case 'U': appendCodepoint(value, hexadecimal(8)); break;
            default: fail();
            }
        }
        fail();
    }
    std::string number() {
        const size_t start = position;
        if (input[position] == '-') ++position;
        if (position == input.size() || !digit(input[position])) fail();
        if (input[position] == '0') ++position;
        else while (position < input.size() && digit(input[position])) ++position;
        if (position < input.size() && input[position] == '.') {
            ++position;
            const size_t begin = position;
            while (position < input.size() && digit(input[position])) ++position;
            if (begin == position) fail();
        }
        if (position < input.size() && (input[position] == 'e' || input[position] == 'E')) {
            ++position;
            if (position < input.size() && (input[position] == '+' || input[position] == '-')) ++position;
            const size_t begin = position;
            while (position < input.size() && digit(input[position])) ++position;
            if (begin == position) fail();
        }
        return std::string(input.substr(start, position - start));
    }
    std::string value(size_t depth) {
        whitespace();
        if (position == input.size()) fail();
        const char c = input[position];
        if (c == '\'' || c == '"') return quoted(string());
        if (digit(c) || c == '-') return number();
        for (const auto& literal : {std::pair<std::string_view, std::string_view>{"True", "true"},
                 {"False", "false"}, {"None", "null"}, {"true", "true"}, {"false", "false"}, {"null", "null"}}) {
            if (input.substr(position, literal.first.size()) == literal.first) {
                position += literal.first.size();
                return std::string(literal.second);
            }
        }
        if (c != '[' && c != '{') fail();
        if (depth >= maxDepth) fail();
        ++position;
        const char end = c == '[' ? ']' : '}';
        std::string result(1, c);
        std::unordered_set<std::string> keys;
        size_t count = 0;
        if (!take(end)) {
            do {
                if (++count > maxEntries) fail();
                if (count > 1) result += ',';
                if (c == '{') {
                    const auto key = string();
                    if (!keys.insert(key).second) fail();
                    result += quoted(key) + ':';
                    require(':');
                }
                result += value(depth + 1);
                if (take(end)) break;
                require(',');
                if (take(end)) break;
            } while (true);
        }
        return result + end;
    }

public:
    explicit LiteralReader(std::string_view source) : input(source) {
        if (input.size() > maxBytes) fail();
    }
    std::string arguments() {
        std::string result = "{";
        std::unordered_set<std::string> names;
        size_t count = 0;
        whitespace();
        while (position < input.size()) {
            if (++count > maxEntries) fail();
            const size_t begin = position;
            if (!alpha(input[position])) fail();
            while (position < input.size() && (alpha(input[position]) || digit(input[position]))) ++position;
            const std::string name(input.substr(begin, position - begin));
            if (!names.insert(name).second) fail();
            require('=');
            if (count > 1) result += ',';
            result += quoted(name) + ':' + value(0);
            whitespace();
            if (position == input.size()) break;
            require(',');
            whitespace();
        }
        return result + '}';
    }
};

inline std::string parseArguments(std::string_view input) {
    return LiteralReader(input).arguments();
}
}  // namespace ovms::lfm2
