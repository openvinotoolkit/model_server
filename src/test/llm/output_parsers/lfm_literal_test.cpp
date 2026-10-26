// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <fstream>
#include <iterator>
#include <rapidjson/document.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace std::string_literals;
#ifndef LFM_LITERAL_STANDALONE
#include "src/test/platform_utils.hpp"
#endif
#ifdef LFM_LEGACY_REFERENCE
#include "legacy_lfm_normalize.hpp"
#else
#include "src/llm/io_processing/lfm2/python_literal.hpp"
#endif

TEST(LfmLiteralTest, GoldenValuesPreserveTypesAndSpelling) {
    const std::vector<std::pair<std::string, std::string>> fixtures = {
        {"arg=None", R"({"arg":null})"},
        {"arg=[True, False, None, {'key': [1.10, 1e3, 9007199254740993]}]", R"({"arg":[true,false,null,{"key":[1.10,1e3,9007199254740993]}]})"},
        {R"(arg='O\'Brien')", R"({"arg":"O'Brien"})"},
        {R"(arg='C:\\test\\')", R"({"arg":"C:\\test\\"})"},
        {R"(arg='a\n\t\x41\u00e9\U0001F600')", "{\"arg\":\"a\\n\\tA\xc3\xa9\xf0\x9f\x98\x80\"}"},
        {R"(arg='\uD83D\uDE00')", "{\"arg\":\"\xf0\x9f\x98\x80\"}"},
        {R"(arg="it's [a,b] (x=y) <|tool_call_end|>")", R"({"arg":"it's [a,b] (x=y) <|tool_call_end|>"})"},
        {" a = {'x': True, 'n': None,}, b=[1,2,], ", R"({"a":{"x":true,"n":null},"b":[1,2]})"},
        {"arg=-0.00E+10", R"({"arg":-0.00E+10})"},
        {"", "{}"},
        {"arg=''", R"({"arg":""})"},
    };
    for (const auto& fixture : fixtures) {
        EXPECT_EQ(ovms::lfm2::parseArguments(fixture.first), fixture.second) << fixture.first;
    }
}

TEST(LfmLiteralTest, RejectsMalformedOrExecutableSyntax) {
    const std::vector<std::string> bad = {
        "arg=", "arg", "arg=missing", "arg=NoneX", "arg=TRUE", "arg=NaN", "arg=Infinity",
        "arg=1+2", "arg=__import__('os')", "arg=(1,2)", "arg=[x for x in y]", "arg=x.y",
        "arg=b'bytes'", "arg=r'raw'", "arg=01", "arg=1.", "arg=.1", "arg=+1", "arg=1e",
        "arg={'a':1,'a':2}", "a=1,a=2", "arg={1:'x'}", "arg=[1,,2]", "arg={'a' 1}",
        "arg='unclosed", R"(arg='bad\q')", R"(arg='bad\x0z')", R"(arg='\uD800')", R"(arg='\uDE00')",
        R"(arg='\U00110000')", "arg='raw\nnewline'", "arg=true trailing", "arg=[] garbage", "arg=1\0tail"s,
    };
    for (const auto& input : bad) {
        EXPECT_THROW(ovms::lfm2::parseArguments(input), std::invalid_argument) << input;
    }
}

TEST(LfmLiteralTest, BoundsAreEnforced) {
    EXPECT_THROW(ovms::lfm2::parseArguments("arg=" + std::string(65, '[') + "0" + std::string(65, ']')), std::invalid_argument);
    EXPECT_THROW(ovms::lfm2::parseArguments("arg='" + std::string(1024 * 1024, 'x') + "'"), std::invalid_argument);
    std::string items = "arg=[";
    for (int i = 0; i < 4097; ++i) items += "0,";
    EXPECT_THROW(ovms::lfm2::parseArguments(items + "]"), std::invalid_argument);
}

TEST(LfmLiteralTest, FixedSeedStringRoundTripAndHostileMutation) {
    std::mt19937 random(1234);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        std::string value;
        for (unsigned i = 0, n = random() % 80; i < n; ++i) value.push_back(static_cast<char>(random() % 128));
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
        const std::string json(buffer.GetString(), buffer.GetSize());
        EXPECT_EQ(ovms::lfm2::parseArguments("arg=" + json), "{\"arg\":" + json + "}");
        EXPECT_THROW(ovms::lfm2::parseArguments("arg=" + json + "()"), std::invalid_argument);
    }
}

TEST(LfmLiteralTest, ExternalGoldenCorpus) {
#ifdef LFM_LITERAL_STANDALONE
    const std::string path = "src/test/llm/output_parsers/fixtures/lfm_literals.json";
#else
    const std::string path = getGenericFullPathForSrcTest("/ovms/src/test/llm/output_parsers/fixtures/lfm_literals.json", false);
#endif
    std::ifstream file(path);
    ASSERT_TRUE(file.good()) << path;
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    rapidjson::Document corpus;
    corpus.Parse(text.c_str());
    ASSERT_FALSE(corpus.HasParseError());
    ASSERT_TRUE(corpus.IsArray());
    for (const auto& row : corpus.GetArray()) {
        const std::string input(row["input"].GetString(), row["input"].GetStringLength());
        if (row.HasMember("error")) EXPECT_THROW(ovms::lfm2::parseArguments(input), std::invalid_argument) << input;
        else EXPECT_EQ(ovms::lfm2::parseArguments(input), row["expected"].GetString()) << input;
    }
}
