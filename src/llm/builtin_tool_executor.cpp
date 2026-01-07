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

#include "builtin_tool_executor.hpp"

#include <sstream>

#include "../logging.hpp"
#include "src/port/rapidjson_document.hpp"

namespace ovms {

BuiltInToolExecutor::BuiltInToolExecutor() {
    // Register default mock handlers for built-in tools
    // Tool names follow the pattern: <category>.<action> (e.g., browser.search, browser.open)
    handlers["browser.search"] = handleBrowserSearch;
    handlers["browser.open"] = handleBrowserOpen;
    handlers["code_interpreter"] = handleCodeInterpreter;
    handlers["code_interpreter.run"] = handleCodeInterpreter;
    handlers["file_search"] = handleFileSearch;
    handlers["file_search.search"] = handleFileSearch;
    handlers["image_generation"] = handleImageGeneration;
    handlers["image_generation.generate"] = handleImageGeneration;
}

BuiltInToolResults_t BuiltInToolExecutor::execute(const ToolCalls_t& builtInToolCalls) {
    BuiltInToolResults_t results;
    results.reserve(builtInToolCalls.size());

    for (const auto& toolCall : builtInToolCalls) {
        BuiltInToolResult result;
        result.toolCallId = toolCall.id;
        result.toolName = toolCall.name;

        auto it = handlers.find(toolCall.name);
        if (it != handlers.end()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Executing built-in tool: {} with arguments: {}", toolCall.name, toolCall.arguments);
            try {
                result.content = it->second(toolCall.arguments);
                result.success = true;
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Built-in tool {} executed successfully, result: {}", toolCall.name, result.content);
            } catch (const std::exception& e) {
                result.content = std::string("Error executing tool: ") + e.what();
                result.success = false;
                SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Built-in tool {} execution failed: {}", toolCall.name, e.what());
            }
        } else {
            // Unknown built-in tool - provide a generic mock response
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Unknown built-in tool: {}, providing generic mock response", toolCall.name);
            result.content = "Mock response for unknown tool: " + toolCall.name;
            result.success = true;
        }

        results.push_back(std::move(result));
    }

    return results;
}

bool BuiltInToolExecutor::isBuiltInTool(const std::string& toolName) const {
    // Built-in tools typically have a category prefix like "browser.", "code_interpreter", etc.
    // or are explicitly registered
    if (handlers.find(toolName) != handlers.end()) {
        return true;
    }
    // Check for category-based built-in tools
    return toolName.find("browser.") == 0 ||
           toolName.find("code_interpreter") == 0 ||
           toolName.find("file_search") == 0 ||
           toolName.find("image_generation") == 0;
}

void BuiltInToolExecutor::registerHandler(const std::string& toolName, ToolHandler handler) {
    handlers[toolName] = std::move(handler);
}

std::string BuiltInToolExecutor::getArgumentValue(const std::string& arguments, const std::string& key) {
    rapidjson::Document doc;
    if (doc.Parse(arguments.c_str()).HasParseError()) {
        return "";
    }
    if (!doc.IsObject()) {
        return "";
    }
    auto it = doc.FindMember(key.c_str());
    if (it == doc.MemberEnd()) {
        return "";
    }
    if (it->value.IsString()) {
        return it->value.GetString();
    }
    if (it->value.IsNumber()) {
        return std::to_string(it->value.GetInt());
    }
    return "";
}

std::string BuiltInToolExecutor::handleBrowserSearch(const std::string& arguments) {
    std::string query = getArgumentValue(arguments, "query");
    if (query.empty()) {
        query = getArgumentValue(arguments, "q");
    }

    // Mock search results - in a real implementation, this would call a search API
    std::stringstream ss;
    ss << R"({
  "results": [
    {
      "title": "Mock Search Result 1 for: )" << query << R"(",
      "url": "https://example.com/result1",
      "snippet": "This is a mock search result providing information about )" << query << R"(. The content is generated for testing purposes."
    },
    {
      "title": "Mock Search Result 2 for: )" << query << R"(",
      "url": "https://example.com/result2",
      "snippet": "Another relevant result about )" << query << R"(. This mock data simulates what a real search engine would return."
    },
    {
      "title": "Wikipedia - )" << query << R"(",
      "url": "https://en.wikipedia.org/wiki/)" << query << R"(",
      "snippet": ")" << query << R"( is a topic with extensive information available. This mock Wikipedia entry provides an overview."
    }
  ],
  "total_results": 3,
  "search_time_ms": 42
})";
    return ss.str();
}

std::string BuiltInToolExecutor::handleBrowserOpen(const std::string& arguments) {
    std::string url = getArgumentValue(arguments, "url");
    if (url.empty()) {
        url = getArgumentValue(arguments, "link");
    }

    // Mock page content - in a real implementation, this would fetch the actual page
    std::stringstream ss;
    ss << R"({
  "url": ")" << url << R"(",
  "title": "Mock Page Title",
  "content": "This is mock content from the webpage at )" << url << R"(. In a real implementation, this would contain the actual page content extracted and summarized for the AI to process. The content includes various sections discussing the topic in detail.",
  "status": 200,
  "load_time_ms": 150
})";
    return ss.str();
}

std::string BuiltInToolExecutor::handleCodeInterpreter(const std::string& arguments) {
    std::string code = getArgumentValue(arguments, "code");
    if (code.empty()) {
        code = getArgumentValue(arguments, "input");
    }

    // Mock code execution - in a real implementation, this would run the code in a sandbox
    std::stringstream ss;
    ss << R"({
  "status": "success",
  "output": "Mock execution output for the provided code.\nThe code was analyzed and would produce the following results:\n- Variable assignments completed\n- Functions defined successfully\n- No errors detected",
  "execution_time_ms": 25,
  "memory_used_mb": 12.5
})";
    return ss.str();
}

std::string BuiltInToolExecutor::handleFileSearch(const std::string& arguments) {
    std::string query = getArgumentValue(arguments, "query");
    if (query.empty()) {
        query = getArgumentValue(arguments, "search");
    }

    // Mock file search results
    std::stringstream ss;
    ss << R"({
  "matches": [
    {
      "file": "document1.pdf",
      "page": 5,
      "relevance": 0.95,
      "excerpt": "Mock excerpt containing information about )" << query << R"( found in the uploaded documents."
    },
    {
      "file": "notes.txt",
      "line": 42,
      "relevance": 0.87,
      "excerpt": "Another relevant section mentioning )" << query << R"( with additional context."
    }
  ],
  "total_matches": 2
})";
    return ss.str();
}

std::string BuiltInToolExecutor::handleImageGeneration(const std::string& arguments) {
    std::string prompt = getArgumentValue(arguments, "prompt");
    if (prompt.empty()) {
        prompt = getArgumentValue(arguments, "description");
    }

    // Mock image generation - in a real implementation, this would call an image generation API
    std::stringstream ss;
    ss << R"({
  "status": "success",
  "image_url": "https://mock-image-service.example.com/generated/image_12345.png",
  "prompt": ")" << prompt << R"(",
  "dimensions": {
    "width": 1024,
    "height": 1024
  },
  "generation_time_ms": 3500
})";
    return ss.str();
}

}  // namespace ovms
