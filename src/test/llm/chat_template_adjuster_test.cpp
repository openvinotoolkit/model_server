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
//*****************************************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../llm/io_processing/chat_template/adjuster.hpp"

using namespace ovms;

TEST(ChatTemplateAdjusterTest, probesAndRemovesFilters) {
    std::string tmpl = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";
    ChatTemplateAdjusterConfig config;
    config.removeFunctions = {"min", "from_json", "missing_filter"};

    const ChatTemplateAdjuster adjuster(config);
    adjuster.adjust(tmpl);

    EXPECT_EQ(config.removeFunctions, (std::vector<std::string>{"min", "from_json", "missing_filter"}));
    EXPECT_EQ(tmpl, "{{ value | tojson(ensure_ascii=False) }} {{ values }} {{ data }}");
}

TEST(ChatTemplateAdjusterTest, usesMinjaConfiguration) {
    std::string tmpl = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";

    const ChatTemplateAdjuster adjuster(ChatTemplateAdjusterConfig::forMinja());
    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ value | tojson }} {{ values }} {{ data }}");
}

TEST(ChatTemplateAdjusterTest, removesFunctionParamsWithoutRemovingFilter) {
    std::string tmpl = "{{ tool | tojson(ensure_ascii=False) }} {{ tool | tojson(indent=2) }}";
    const RemoveFunctionParamsAdjustment adjustment({"tojson"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{{ tool | tojson }} {{ tool | tojson }}");
}

TEST(ChatTemplateAdjusterTest, removesNestedParamsWithQuotedParentheses) {
    std::string tmpl = R"({{ tool | tojson(option=nested(')', "("), indent=2) | other }})";
    const RemoveFunctionParamsAdjustment adjustment({"tojson"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{{ tool | tojson | other }}");
}

TEST(ChatTemplateAdjusterTest, paramRemovalPreservesUnconfiguredAndBareFilters) {
    const std::string original = "{{ tool | tojson }} {{ tool | tojson_extra(indent=2) }} {{ tool | other(indent=2) }}";
    std::string tmpl = original;
    const RemoveFunctionParamsAdjustment adjustment({"tojson"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, paramRemovalPreservesUnclosedArgumentList) {
    const std::string original = "{{ tool | tojson(indent=2 }}";
    std::string tmpl = original;
    const RemoveFunctionParamsAdjustment adjustment({"tojson"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, removesParamsBeforeRemovingFilter) {
    std::string tmpl = "{{ tool | tojson(ensure_ascii=False) }}";
    ChatTemplateAdjusterConfig config;
    config.removeFunctions = {"tojson"};
    config.removeFunctionsParams = {"tojson"};
    const ChatTemplateAdjuster adjuster(config);

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ tool }}");
}

TEST(ChatTemplateAdjusterTest, emptyConfigurationPreservesSource) {
    const std::string original = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";
    std::string tmpl = original;
    const ChatTemplateAdjusterConfig config{{}};

    const ChatTemplateAdjuster adjuster(config);
    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, jinjaConfigurationPreservesSource) {
    const std::string original = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";
    std::string tmpl = original;

    const ChatTemplateAdjuster adjuster(ChatTemplateAdjusterConfig::forJinja());
    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, removesFiltersIndependently) {
    std::string tmpl = "{{ value | min | minimum }}";
    const RemoveFiltersAdjustment adjustment({"min"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{{ value | minimum }}");
}

TEST(ChatTemplateAdjusterTest, removesConfiguredFilter) {
    std::string tmpl = "{{ value | tojson }}";
    const ChatTemplateAdjuster adjuster({{"tojson"}});

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ value }}");
}

TEST(ChatTemplateAdjusterTest, ownsConfigurationAndCanBeReused) {
    ChatTemplateAdjusterConfig config{{"min"}};
    const ChatTemplateAdjuster adjuster(config);
    config.removeFunctions.clear();
    const std::string original = "{{ value | tojson(indent=2) | min }}";
    std::string first = original;
    std::string second = original;

    adjuster.adjust(first);
    adjuster.adjust(second);

    EXPECT_EQ(first, "{{ value | tojson(indent=2) }}");
    EXPECT_EQ(second, first);
}

TEST(ChatTemplateAdjusterTest, removesLeadingSpacesAndTabsWithConsecutiveFilters) {
    std::string tmpl = "{{ value \t  | min \t | from_json | minimum }}";
    const ChatTemplateAdjuster adjuster(ChatTemplateAdjusterConfig::forMinja());

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ value | minimum }}");
}

TEST(ChatTemplateAdjusterTest, preservesLineBreaksAndSurroundingText) {
    std::string tmpl = "Before  {{ value\n\t | min\n }}  After";
    const RemoveFiltersAdjustment adjustment({"min"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "Before  {{ value\n\n }}  After");
}

TEST(ChatTemplateAdjusterTest, removesBracketsWhenRemovingFunction) {
    std::string tmpl = "{{ value | min() }}";
    const ChatTemplateAdjuster adjuster({{"min"}});

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ value }}");
}

TEST(ChatTemplateAdjusterTest, removesFunctionWithArguments) {
    std::string tmpl = "{{ value | min(some_param=42) }}";
    const ChatTemplateAdjuster adjuster({{"min"}});

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{{ value }}");
}

TEST(ChatTemplateAdjusterTest, notRemovesFunctionFromString) {
    std::string tmpl = "{{ \"Use | min went you want to have lowest number from an array.\" }}";
    const RemoveFiltersAdjustment adjustment({"min"});
    std::string original = tmpl;

    adjustment.apply(tmpl);

    EXPECT_EQ(original, tmpl);
}

TEST(ChatTemplateAdjusterTest, removesCompactFilterWithoutAddingWhitespace) {
    std::string tmpl = "{{value|min}}";
    const RemoveFiltersAdjustment adjustment({"min"});

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{{value}}");
}

TEST(ChatTemplateAdjusterTest, concatenatesMultiLineStrings) {
    std::string tmpl = "\"line1\"\n\"line2\"\n\"line3\"";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "\"line1line2line3\"");
}

TEST(ChatTemplateAdjusterTest, concatenatesTwoLineStrings) {
    std::string tmpl = "\"line1\"\n\"line2\"";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "\"line1line2\"");
}

TEST(ChatTemplateAdjusterTest, concatenatesTwoLineEmptyStrings) {
    std::string tmpl = "\"\"\n\"\"";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "\"\"");
}

TEST(ChatTemplateAdjusterTest, concatenatesMultilineWithIndentAndSpaces) {
    std::string tmpl = "\"line1\"\n\t\t\t\t\t\"line2\"\n\t\t\t\t\"line3\"";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "\"line1line2line3\"");
}

TEST(ChatTemplateAdjusterTest, concatenatesAdjacentStringsAcrossMixedWhitespace) {
    std::string tmpl = "\"line1\" \t\n  \"line2\"";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "\"line1line2\"");
}

TEST(ChatTemplateAdjusterTest, omitsEmptyStrings) {
    std::string tmpl = "{%- set emptyString = \"\" -%}";
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{%- set emptyString = \"\" -%}");
}

TEST(ChatTemplateAdjusterTest, preservesEmptyAssignmentBetweenStringExpressions) {
    const std::string original = R"({{ "header" }}
{%- set developer_message = "" %}
{{ "tail" }})";
    std::string tmpl = original;
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, replacesUndefinedChecks) {
    std::string tmpl = "{% if value is undefined or message.content\tis\nundefined %}missing{% endif %}";
    const ReplaceUndefinedChecksAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, "{% if not value is defined or not message.content is defined %}missing{% endif %}");
}

TEST(ChatTemplateAdjusterTest, minjaConfigurationReplacesUndefinedChecks) {
    std::string tmpl = "{% if value is undefined %}missing{% endif %}";
    const ChatTemplateAdjuster adjuster(ChatTemplateAdjusterConfig::forMinja());

    adjuster.adjust(tmpl);

    EXPECT_EQ(tmpl, "{% if not value is defined %}missing{% endif %}");
}

TEST(ChatTemplateAdjusterTest, undefinedCheckReplacementIsDisabledByDefaultAndForJinja) {
    const std::string original = "{% if value is undefined %}missing{% endif %}";
    for (const auto& config : {ChatTemplateAdjusterConfig{}, ChatTemplateAdjusterConfig::forJinja()}) {
        std::string tmpl = original;
        const ChatTemplateAdjuster adjuster(config);

        adjuster.adjust(tmpl);

        EXPECT_EQ(tmpl, original);
    }
}

TEST(ChatTemplateAdjusterTest, undefinedCheckReplacementPreservesDefinedChecks) {
    const std::string original = "{% if value is defined and not message.content is defined %}present{% endif %}";
    std::string tmpl = original;
    const ReplaceUndefinedChecksAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, preservesDoubleQuotesInsideSingleQuotedLiteral) {
    const std::string original = R"({{ '"first" "second"' }})";
    std::string tmpl = original;
    const ConcatMultiLineStringAdjustment adjustment;

    adjustment.apply(tmpl);

    EXPECT_EQ(tmpl, original);
}
