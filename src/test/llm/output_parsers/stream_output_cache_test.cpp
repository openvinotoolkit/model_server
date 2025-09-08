//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <optional>
#include "../../../llm/io_processing/output_parser.hpp"

using namespace ovms;

TEST(StreamOutputCacheTest, LookupTag) {
    OutputParser::StreamOutputCache cache;
    cache.add("functoo");
    EXPECT_EQ(cache.lookupTag("func"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    EXPECT_EQ(cache.lookupTag("to"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    EXPECT_EQ(cache.lookupTag("functools"), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    EXPECT_EQ(cache.lookupTag("functor"), OutputParser::TagLookupStatus::NOT_FOUND);
    EXPECT_EQ(cache.lookupTag("functorrrrrr"), OutputParser::TagLookupStatus::NOT_FOUND);
    cache.add("ls");
    EXPECT_EQ(cache.lookupTag("functools"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    cache.add("[\"");
    EXPECT_EQ(cache.lookupTag("functools"), OutputParser::TagLookupStatus::FOUND_COMPLETE);

    cache.clear();
    // Not realistic but tests the logic
    EXPECT_EQ(cache.lookupTag("func"), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);

    cache.add("functools");
    EXPECT_EQ(cache.lookupTag("functools"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    EXPECT_EQ(cache.lookupTag("functoo"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    EXPECT_EQ(cache.lookupTag("tools"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    EXPECT_EQ(cache.lookupTag("functools["), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    EXPECT_EQ(cache.lookupTag("toools"), OutputParser::TagLookupStatus::NOT_FOUND);
    EXPECT_EQ(cache.lookupTag("functoool"), OutputParser::TagLookupStatus::NOT_FOUND);

    cache.clear();
    cache.add("end. ");
    EXPECT_EQ(cache.lookupTag("</think>"), OutputParser::TagLookupStatus::NOT_FOUND);
    cache.add("\n</");
    EXPECT_EQ(cache.lookupTag("</think>"), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    cache.add("think");
    EXPECT_EQ(cache.lookupTag("</think>"), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    cache.add(">");
    EXPECT_EQ(cache.lookupTag("</think>"), OutputParser::TagLookupStatus::FOUND_COMPLETE);

    cache.clear();
    cache.add("<thin");
    EXPECT_EQ(cache.lookupTag("<think>"), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    cache.add("k>\n text");
    EXPECT_EQ(cache.lookupTag("<think>"), OutputParser::TagLookupStatus::FOUND_COMPLETE);
    cache.clear();
}

TEST(StreamOutputCacheTest, LookupTags) {
    OutputParser::StreamOutputCache cache;
    cache.add("{\"name\":");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::FOUND_COMPLETE);

    cache.clear();
    cache.add("some text <|python");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    cache.add("_tag|> more text");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::FOUND_COMPLETE);

    cache.clear();
    cache.add("<|python{");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::FOUND_COMPLETE);

    cache.clear();
    cache.add("<|python_tag|");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::FOUND_INCOMPLETE);
    cache.add("text");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::NOT_FOUND);
    cache.add("|>");
    EXPECT_EQ(cache.lookupTags({"<|python_tag|>", "{"}), OutputParser::TagLookupStatus::NOT_FOUND);
    cache.clear();
}
