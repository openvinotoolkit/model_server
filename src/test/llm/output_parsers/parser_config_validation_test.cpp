//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <algorithm>
#include <string>

#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>

#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/io_processing/parser_config_validation.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string parserConfigValidationTokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\microsoft\\Phi-4-mini-instruct";
#else
const std::string parserConfigValidationTokenizerPath = "/ovms/src/test/llm_testing/microsoft/Phi-4-mini-instruct";
#endif

static ovms::ToolsSchemas_t PARSER_CONFIG_VALIDATION_EMPTY_TOOLS_SCHEMA = {};
static std::unique_ptr<ov::genai::Tokenizer> parserConfigValidationTokenizer;

class ParserConfigValidationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            parserConfigValidationTokenizer = std::make_unique<ov::genai::Tokenizer>(parserConfigValidationTokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize tokenizer: " << e.what();
        }
    }

    static void TearDownTestSuite() {
        parserConfigValidationTokenizer.reset();
    }
};

TEST_F(ParserConfigValidationTest, RegistryHasExpectedToolParsers) {
    const auto& names = getSupportedToolParserNames();
    for (const auto& expected : {"llama3", "hermes3", "phi4", "mistral", "gptoss",
             "qwen3coder", "devstral", "lfm2", "lfm2.5", "gemma4"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end())
            << "Expected tool parser '" << expected << "' missing from registry";
    }
    EXPECT_TRUE(isSupportedToolParserName("hermes3"));
    EXPECT_FALSE(isSupportedToolParserName("nonexistent"));
    EXPECT_FALSE(isSupportedToolParserName(""));
}

TEST_F(ParserConfigValidationTest, RegistryHasExpectedReasoningParsers) {
    const auto& names = getSupportedReasoningParserNames();
    for (const auto& expected : {"qwen3", "gemma4", "gptoss", "lfm2.5"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end())
            << "Expected reasoning parser '" << expected << "' missing from registry";
    }
    EXPECT_TRUE(isSupportedReasoningParserName("qwen3"));
    EXPECT_FALSE(isSupportedReasoningParserName("nonexistent"));
    EXPECT_FALSE(isSupportedReasoningParserName(""));
}

TEST_F(ParserConfigValidationTest, SupportedNamesStringContainsAllParsers) {
    const std::string toolNames = getSupportedToolParserNamesAsString();
    EXPECT_NE(toolNames.find("hermes3"), std::string::npos);
    EXPECT_NE(toolNames.find("phi4"), std::string::npos);
    const std::string reasoningNames = getSupportedReasoningParserNamesAsString();
    EXPECT_NE(reasoningNames.find("qwen3"), std::string::npos);
    EXPECT_NE(reasoningNames.find("gptoss"), std::string::npos);
    EXPECT_NE(reasoningNames.find("lfm2.5"), std::string::npos);
}

TEST_F(ParserConfigValidationTest, OutputParserThrowsOnUnknownToolParser) {
    try {
        OutputParser parser(*parserConfigValidationTokenizer, "totally_invalid_parser", "", PARSER_CONFIG_VALIDATION_EMPTY_TOOLS_SCHEMA);
        FAIL() << "OutputParser should have thrown for unknown tool parser name";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Unsupported tool parser"), std::string::npos) << what;
        EXPECT_NE(what.find("totally_invalid_parser"), std::string::npos) << what;
        EXPECT_NE(what.find("hermes3"), std::string::npos)
            << "Error message should list supported parsers, got: " << what;
    }
}

TEST_F(ParserConfigValidationTest, OutputParserThrowsOnUnknownReasoningParser) {
    try {
        OutputParser parser(*parserConfigValidationTokenizer, "", "totally_invalid_reasoning", PARSER_CONFIG_VALIDATION_EMPTY_TOOLS_SCHEMA);
        FAIL() << "OutputParser should have thrown for unknown reasoning parser name";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Unsupported reasoning parser"), std::string::npos) << what;
        EXPECT_NE(what.find("totally_invalid_reasoning"), std::string::npos) << what;
        EXPECT_NE(what.find("qwen3"), std::string::npos)
            << "Error message should list supported parsers, got: " << what;
    }
}

TEST_F(ParserConfigValidationTest, OutputParserAcceptsEmptyNames) {
    EXPECT_NO_THROW({
        OutputParser parser(*parserConfigValidationTokenizer, "", "", PARSER_CONFIG_VALIDATION_EMPTY_TOOLS_SCHEMA);
        EXPECT_FALSE(parser.isToolParserAvailable());
        EXPECT_FALSE(parser.isReasoningParserAvailable());
    });
}
