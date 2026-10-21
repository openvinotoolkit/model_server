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
    config.removeFunctionsParams = {"tojson", "missing_params"};
    config.removeFunctions = {"min", "from_json", "missing_filter"};

    ChatTemplateAdjuster::adjust(tmpl, config);

    EXPECT_EQ(config.removeFunctionsParams, (std::vector<std::string>{"tojson", "missing_params"}));
    EXPECT_EQ(config.removeFunctions, (std::vector<std::string>{"min", "from_json", "missing_filter"}));
    EXPECT_EQ(tmpl, "{{ value | tojson }} {{ values  }} {{ data  }}");
}

TEST(ChatTemplateAdjusterTest, usesMinjaConfiguration) {
    std::string tmpl = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";

    ChatTemplateAdjuster::adjust(tmpl, ChatTemplateAdjusterConfig::forMinja());

    EXPECT_EQ(tmpl, "{{ value | tojson }} {{ values  }} {{ data  }}");
}

TEST(ChatTemplateAdjusterTest, emptyConfigurationPreservesSource) {
    const std::string original = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";
    std::string tmpl = original;
    const ChatTemplateAdjusterConfig config{{}, {}};

    ChatTemplateAdjuster::adjust(tmpl, config);

    EXPECT_EQ(tmpl, original);
}

TEST(ChatTemplateAdjusterTest, jinjaConfigurationPreservesSource) {
    const std::string original = R"({{ value | tojson(ensure_ascii=False) }} {{ values | min }} {{ data | from_json }})";
    std::string tmpl = original;

    ChatTemplateAdjuster::adjust(tmpl, ChatTemplateAdjusterConfig::forJinja());

    EXPECT_EQ(tmpl, original);
}