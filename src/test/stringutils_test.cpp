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

#include <fstream>
#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../stringutils.hpp"

using namespace testing;

TEST(StringUtils, joins) {
    EXPECT_EQ("", ovms::joins({}, ","));
    EXPECT_EQ("A", ovms::joins({"A"}, ","));
    EXPECT_EQ("A,B", ovms::joins({"A", "B"}, ","));
    EXPECT_EQ("Abe,Bece", ovms::joins({"Abe", "Bece"}, ","));
    EXPECT_EQ("A,B,,D", ovms::joins({"A", "B", "", "D"}, ","));
}

TEST(StringUtils, ltrim) {
    std::string str0 = "";
    std::string str1 = "   {1234 5 67890}   ";
    std::string str2 = "\n\r\t \n\r\t{1234 5 67890}\n\r\t  ";

    ovms::ltrim(str0);
    ovms::ltrim(str1);
    ovms::ltrim(str2);

    EXPECT_EQ(str0, "");
    EXPECT_EQ(str1, "{1234 5 67890}   ");
    EXPECT_EQ(str2, "{1234 5 67890}\n\r\t  ");
}

TEST(StringUtils, rtrim) {
    std::string str0 = "";
    std::string str1 = "   {1234 5 67890}   ";
    std::string str2 = "\n\r\t \n\r\t{1234 5 67890}\n\r\t  ";

    ovms::rtrim(str0);
    ovms::rtrim(str1);
    ovms::rtrim(str2);

    EXPECT_EQ(str0, "");
    EXPECT_EQ(str1, "   {1234 5 67890}");
    EXPECT_EQ(str2, "\n\r\t \n\r\t{1234 5 67890}");
}

TEST(StringUtils, trim) {
    std::string str0 = "";
    std::string str1 = "   {1234 5 67890}   ";
    std::string str2 = "\n\r\t \n\r\t{1234 5 67890}\n\r\t  ";

    ovms::trim(str0);
    ovms::trim(str1);
    ovms::trim(str2);

    EXPECT_EQ(str0, "");
    EXPECT_EQ(str1, "{1234 5 67890}");
    EXPECT_EQ(str2, "{1234 5 67890}");
}

TEST(StringUtils, erase_spaces) {
    std::string str0 = "";
    std::string str1 = "   {1234 5 67890}   ";
    std::string str2 = "\n\r\t \n\r\t{1234 5 67890}\n\r\t  ";

    ovms::erase_spaces(str0);
    ovms::erase_spaces(str1);
    ovms::erase_spaces(str2);

    EXPECT_EQ(str0, "");
    EXPECT_EQ(str1, "{1234567890}");
    EXPECT_EQ(str2, "{1234567890}");
}

TEST(StringUtils, tokenize) {
    std::string str0 = "";
    std::string str1 = "uno dos tres";
    std::string str2 = "   ";
    std::string str3 = "1,2,3,4,,,";

    auto t0 = ovms::tokenize(str0, ';');
    auto t1 = ovms::tokenize(str1, ' ');
    auto t2 = ovms::tokenize(str2, ' ');
    auto t3 = ovms::tokenize(str3, ',');

    EXPECT_EQ(t0.size(), 0);
    EXPECT_THAT(t1, ElementsAre("uno", "dos", "tres"));
    EXPECT_EQ(t2.size(), 3);
    EXPECT_THAT(t3, ElementsAre("1", "2", "3", "4", "", ""));
}

TEST(StringUtils, endsWith) {
    std::string str0 = "";
    std::string str1 = "test case 1";
    std::string str2 = "not really matter 1 }";

    auto b0 = ovms::endsWith(str0, "");
    auto b1 = ovms::endsWith(str0, "/");
    auto b2 = ovms::endsWith(str1, "");
    auto b3 = ovms::endsWith(str1, "1");
    auto b4 = ovms::endsWith(str1, "2");
    auto b5 = ovms::endsWith(str2, " 1 }");
    auto b6 = ovms::endsWith(str2, "11 }");

    EXPECT_EQ(b0, true);
    EXPECT_EQ(b1, false);
    EXPECT_EQ(b2, true);
    EXPECT_EQ(b3, true);
    EXPECT_EQ(b4, false);
    EXPECT_EQ(b5, true);
    EXPECT_EQ(b6, false);
}

TEST(StringUtils, startsWith) {
    using ovms::startsWith;
    using std::string;
    string str0 = "";
    string str1 = "test case 1";
    string str2 = "{ not really matter 1 }";

    EXPECT_EQ(startsWith(str0.c_str(), ""), true);
    EXPECT_EQ(startsWith(str0.c_str(), "/"), false);
    EXPECT_EQ(startsWith(str1.c_str(), ""), true);
    EXPECT_EQ(startsWith(str1.c_str(), "test"), true);
    EXPECT_EQ(startsWith(str1.c_str(), "2"), false);
    EXPECT_EQ(startsWith(str2.c_str(), "{ not "), true);
    EXPECT_EQ(startsWith(str2.c_str(), "{ 1not"), false);
    EXPECT_EQ(startsWith(str2.c_str(), "{ 1not"), false);
    EXPECT_EQ(startsWith(string("TENSOR"), string("TENSOR")), true);
    EXPECT_EQ(startsWith(string("TENSOR").c_str(), string("TENSOR")), true);
    EXPECT_EQ(startsWith("TENSOR", string("TENSOR")), true);
    EXPECT_EQ(startsWith(string("TENSOR").c_str(), "TENSOR"), true);
    EXPECT_EQ(startsWith("TENSOR1", "TENSOR"), true);
    EXPECT_EQ(startsWith("TENSOR_1", "TENSOR"), true);
    EXPECT_EQ(startsWith("TENSORA", "TENSOR"), true);
    EXPECT_EQ(startsWith("TENSO", "TENSOR"), false);
}

TEST(StringUtils, stof) {
    auto result = ovms::stof("  -100 ");
    EXPECT_FALSE(result);

    result = ovms::stof("  -100.0 ");
    EXPECT_FALSE(result);

    result = ovms::stof("-100");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), -100.0f, 0.0001f);

    result = ovms::stof("-100.0");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), -100.0f, 0.0001f);

    result = ovms::stof("100.0");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), 100.0f, 0.0001f);

    result = ovms::stof("100.0000000000001");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), 100.0f, 0.0001f);

    result = ovms::stof("0.01");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), 0.01f, 0.0001f);

    result = ovms::stof("0.0000000000001");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), 0.0f, 0.0001f);

    // with e
    result = ovms::stof("1e-10");
    EXPECT_TRUE(result);
    EXPECT_NEAR(result.value(), 1e-10f, 0.0001f);

    // inf / nan
    result = ovms::stof("inf");
    EXPECT_FALSE(result) << result.value();
    result = ovms::stof("nan");
    EXPECT_FALSE(result) << result.value();
    result = ovms::stof("1.0e+100");
    EXPECT_FALSE(result) << result.value();
    result = ovms::stof("1.0e-100");
    EXPECT_FALSE(result);
}

TEST(StringUtils, stou32) {
    auto result = ovms::stou32("-100");
    EXPECT_FALSE(result);

    result = ovms::stou32("4294967296");
    EXPECT_FALSE(result);

    result = ovms::stou32("4294967295");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 4294967295);
}

TEST(StringUtils, stou64) {
    auto result = ovms::stou64("-100");
    EXPECT_FALSE(result);

    result = ovms::stou64("   100 ");
    EXPECT_FALSE(result);

    result = ovms::stou64("18446744073709551616");
    EXPECT_FALSE(result);

    result = ovms::stou64("18446744073709551615");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 18446744073709551615ULL);
}

TEST(StringUtils, stoi32) {
    auto result = ovms::stoi32("-100");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), -100);

    result = ovms::stoi32("2147483648");
    EXPECT_FALSE(result);

    result = ovms::stoi32("2147483647");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 2147483647);

    result = ovms::stoi32("-2147483649");
    EXPECT_FALSE(result);

    result = ovms::stoi32("-2147483648");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), -2147483648);
}

TEST(StringUtils, stoi64) {
    auto result = ovms::stoi64("0");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 0);

    result = ovms::stoi64("100");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 100);

    result = ovms::stoi64("-100");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), -100);

    result = ovms::stoi64("2147483647");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 2147483647);

    result = ovms::stoi64("0.01");
    EXPECT_FALSE(result) << result.value();

    result = ovms::stoi64("1 1");
    EXPECT_FALSE(result) << result.value();

    result = ovms::stoi64("0018");
    EXPECT_FALSE(result) << result.value();

    result = ovms::stoi64("zero");
    EXPECT_FALSE(result);

    result = ovms::stoi64("9223372036854775807");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 9223372036854775807);

    result = ovms::stoi64("9223372036854775808");
    EXPECT_FALSE(result);

    result = ovms::stoi64("");
    EXPECT_FALSE(result);
}

TEST(StringUtils, isValidUtf8) {
    auto result = ovms::isValidUtf8("\x7a");  // one ASCII char
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\x1a\x2b\x3c");  // three ASCII chars
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\x2b\x3c\x1a\x2b\x3c");  // six ASCII chars
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\x1a\xca\xaa");  // one ASCII char and one UTF-8 char
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xea\xaa\xaa");  // one 3byte long UTF-8 char
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xf5\xab\xab\xac");  // one 4byte long UTF-8 char
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xf5\xab\xab");  // incomplete 4byte long UTF-8 char
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("\xea\xaa");  // incomplete 3byte long UTF-8 char
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("\xf5\xc0");  // incorrect char
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("\x1a\xca");  // ASCII char followed by incomplete UTF-8 char
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("");  // Empty content considered invalid because there is nothing to return as partial response
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("\x7a\xaa\xaa");  // incorrect sequence without length information
    EXPECT_FALSE(result);

    result = ovms::isValidUtf8("\xc3\xa9");
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xe2\x82\xac");
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xf0\x9d\x84\x9e");
    EXPECT_TRUE(result);

    result = ovms::isValidUtf8("\xaa\xaa");  // iterations would decrease i below 0
    EXPECT_FALSE(result);
}
