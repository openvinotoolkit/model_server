//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#pragma once

#include <algorithm> 
#include <cctype>
#include <locale>
#include <string>
#include <vector>

namespace ovms {

/**
 * @brief Trims the string on the left side
 * 
 * @param std::string&
 */
static inline void ltrim(std::string &str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int c) {
        return !std::isspace(c);
    }));
}

/**
 * @brief Trims the string on the right side
 * 
 * @param str
 */
static inline void rtrim(std::string &str) {
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int c) {
        return !std::isspace(c);
    }).base(), str.end());
}

/**
 * @brief Trims the string
 * 
 * @param str
 */
static inline void trim(std::string &str) {
    ltrim(str);
    rtrim(str);
}

/**
 * @brief Erases all whitespace characters from string
 * 
 * @param str
 */
static inline void erase_spaces(std::string &str) {
    str.erase(std::remove_if(str.begin(), str.end(), 
        [](char c) -> bool
        { 
            return std::isspace<char>(c, std::locale::classic()); 
        }), str.end());
}

/**
 * @brief Tokenizes a string into a vector of tokens
 * 
 * @param str 
 * @param delimiter 
 * @return std::vector<std::string> 
 */
static inline
std::vector<std::string> tokenize(const std::string& str, const char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(str);
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

/**
 * @brief Checks if given string ends with another one
 *
 * @param str
 * @param match
 * @return true
 * @return false
 */
static inline
bool endsWith(const std::string& str, const std::string& match)
{
	auto it = match.begin();
	return str.size() >= match.size() &&
		std::all_of(std::next(str.begin(),str.size() - match.size()), str.end(), [&it](const char & c){
			return ::tolower(c) == ::tolower(*(it++))  ;
	    });
}

} // namespace ovms
