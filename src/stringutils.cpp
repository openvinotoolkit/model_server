//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include "stringutils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <utility>

namespace ovms {

std::string joins(const std::vector<std::string>& listOfStrings, const std::string delimiter) {
    std::stringstream ss;
    auto it = listOfStrings.cbegin();
    if (it == listOfStrings.end()) {
        return "";
    }
    for (; it != (listOfStrings.end() - 1); ++it) {
        ss << *it << delimiter;
    }
    if (it != listOfStrings.end()) {
        ss << *it;
    }
    return ss.str();
}

void ltrim(std::string& str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int c) {
        return !std::isspace(c);
    }));
}

void rtrim(std::string& str) {
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int c) {
        return !std::isspace(c);
    })
                  .base(),
        str.end());
}

void trim(std::string& str) {
    ltrim(str);
    rtrim(str);
}

void erase_spaces(std::string& str) {
    str.erase(std::remove_if(str.begin(), str.end(),
                  [](char c) -> bool {
                      return std::isspace<char>(c, std::locale::classic());
                  }),
        str.end());
}

std::vector<std::string> tokenize(const std::string& str, const char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(str);
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

bool endsWith(const std::string& str, const std::string& match) {
    auto it = match.begin();
    return str.size() >= match.size() &&
           std::all_of(std::next(str.begin(), str.size() - match.size()), str.end(), [&it](const char& c) {
               return ::tolower(c) == ::tolower(*(it++));
           });
}

bool startsWith(const std::string& str, const std::string& prefix) {
    auto it = prefix.begin();
    bool sizeCheck = (str.size() >= prefix.size());
    if (!sizeCheck) {
        return false;
    }
    bool allOf = std::all_of(str.begin(),
        std::next(str.begin(), prefix.size()),
        [&it](const char& c) {
            return c == *(it++);
        });
    return allOf;
}

std::optional<uint32_t> stou32(const std::string& input) {
    std::string str = input;
    ovms::erase_spaces(str);

    if (str.size() > 0 && str[0] == '-') {
        return std::nullopt;
    }

    try {
        uint64_t val = std::stoul(str);
        if (val > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        return {static_cast<uint32_t>(val)};
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int32_t> stoi32(const std::string& str) {
    try {
        return {static_cast<int32_t>(std::stoi(str))};
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int64_t> stoi64(const std::string& str) {
    if (!str.size()) {
        return std::nullopt;
    }
    bool isMinus = (str[0] == '-');
    size_t i = 0;
    if (isMinus) {
        i = 1;
    }
    for (; i < str.size(); ++i) {
        if (!std::isdigit(str[i])) {
            return std::nullopt;
        }
    }
    if (str.size() > 1 && str[isMinus] == '0') {
        return std::nullopt;
    }
    try {
        return std::stoll(str);
    } catch (...) {
        return std::nullopt;
    }
}

// TODO: Write unit test
std::optional<float> stof(const std::string& str) {
    if (str.empty()) {
        return std::nullopt;
    }

    size_t idx = 0;
    try {
        float val = std::stof(str, &idx);
        // Check if the whole string was consumed
        if (idx != str.size()) {
            return std::nullopt;
        }
        // Reject NaN or Inf
        if (std::isnan(val) || std::isinf(val)) {
            return std::nullopt;
        }
        return val;
    } catch (...) {
        return std::nullopt;
    }
}

bool isValidUtf8(const std::string& text) {
    // inspect the chars from the end of the string to test if the utf8 sequence is complete
    size_t byte_counter = 0;
    if (text.size() == 0) {
        return false;
    }
    for (size_t i = text.size(); i-- > 0 && byte_counter <= 3;) {
        int x = static_cast<int>(static_cast<unsigned char>(text[i]));
        if (((x >> 7) == 0b0) && (byte_counter == 0))
            return true;  // last char is a single byte char
        if ((x >> 6) == 0b10)
            byte_counter++;  // octet belong to multibyte sequence
        else if (((x >> 5) == 0b110) && (byte_counter + 1 == 2))
            return true;  // first byte of 2 byte sequence
        else if (((x >> 4) == 0b1110) && (byte_counter + 1 == 3))
            return true;  // first byte of 3 byte sequence
        else if (((x >> 3) == 0b11110) && (byte_counter + 1 == 4))
            return true;  // first byte of 3 byte sequence
        else
            return false;  // invalid utf8 sequence
    }
    return false;
}

std::string toLower(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

}  // namespace ovms
