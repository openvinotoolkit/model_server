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
#include <string>

#include <gtest/gtest.h>
#include <openvino/genai/chat_history.hpp>

#include "../../../llm/io_processing/input_processors/raw_prompt_extractor.hpp"
#include "../../../llm/io_processing/input_request.hpp"

using namespace ovms;

TEST(RawPromptExtractorTest, MovesPromptToPromptText) {
    InputRequest req;
    req.input = std::string("Explain quantum computing.");

    RawPromptExtractor extractor;
    const auto status = extractor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(req.promptText, "Explain quantum computing.");
}

TEST(RawPromptExtractorTest, EmptyStringProducesEmptyPromptText) {
    InputRequest req;
    req.input = std::string("");

    RawPromptExtractor extractor;
    const auto status = extractor.process(req);

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(req.promptText.empty());
}

// Note: tests for ChatTemplateProcessor and TokenizationProcessor that require
// an actual tokenizer are covered by integration tests in
// src/test/llm/input_processing/chat_template_processor_test.cpp and
// src/test/llm/input_processing/tokenization_processor_test.cpp.
