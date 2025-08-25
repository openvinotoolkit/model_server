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

#include <optional>
#include <string>
#include <vector>
namespace ovms {

std::string joins(const std::vector<std::string>& listOfStrings, const std::string delimiter);

/**
 * @brief Trims the string on the left side
 * 
 * @param std::string&
 */
void ltrim(std::string& str);

/**
 * @brief Trims the string on the right side
 * 
 * @param str
 */
void rtrim(std::string& str);

/**
 * @brief Trims the string
 * 
 * @param str
 */
void trim(std::string& str);

/**
 * @brief Erases all whitespace characters from string
 * 
 * @param str
 */
void erase_spaces(std::string& str);

/**
 * @brief Tokenizes a string into a vector of tokens
 * 
 * @param str 
 * @param delimiter 
 * @return std::vector<std::string> 
 */
std::vector<std::string> tokenize(const std::string& str, const char delimiter);

/**
 * @brief Checks if given string ends with another one but disregards upper/lower case difference
 *
 * @param str
 * @param match
 * @return true
 * @return false
 */
bool endsWith(const std::string& str, const std::string& match);

/**
 * @brief Checks if given string starts with another one
 *
 * @param str
 * @param match
 * @return true
 * @return false
 */
bool startsWith(const std::string& str, const std::string& prefix);

/**
 * @brief Converts string to uint32, returns 0 or specified default value if conversion failed, fails if negative number is provided
 *
 * @param string input
 * @param default value
 * @return converted value and result indicating if conversion succeeded
 */
std::optional<uint32_t> stou32(const std::string& input);

std::optional<uint64_t> stou64(const std::string& input);

/**
 * @brief Converts string to int32, returns 0 or specified default value if conversion failed
 *
 * @param string input
 * @param default value
 * @return converted value and result indicating if conversion succeeded
 */
std::optional<int32_t> stoi32(const std::string& str);

std::optional<int64_t> stoi64(const std::string& str);

std::optional<float> stof(const std::string& str);

bool isValidUtf8(const std::string& text);

std::string toLower(const std::string& input);

}  // namespace ovms
