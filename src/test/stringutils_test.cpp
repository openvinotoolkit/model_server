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

TEST(StringUtils, stou32) {
    auto result = ovms::stou32("-100");
    EXPECT_FALSE(result);

    result = ovms::stou32("4294967296");
    EXPECT_FALSE(result);

    result = ovms::stou32("4294967295");
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 4294967295);
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
