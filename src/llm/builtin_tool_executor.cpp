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
  "requestId": "7e4f9a2d-8c3b-4a1e-9f5d-2b6c8e0a3d7f",
  "autopromptString": "2026 AI artificial intelligence developments breaking news",
  "results": [
    {
      "title": "Anthropic Unveils Claude 4 with Autonomous Agent Capabilities",
      "url": "https://www.anthropic.com/news/claude-4-autonomous-agents",
      "publishedDate": "2026-01-06T16:45:00.000Z",
      "author": "Dario Amodei",
      "score": 0.9891,
      "text": "Anthropic has released Claude 4, featuring breakthrough autonomous agent capabilities that allow the model to complete multi-hour tasks independently while maintaining human oversight through a new constitutional AI framework. The model demonstrates unprecedented performance on agentic benchmarks, completing complex software engineering tasks with 94% accuracy without human intervention. Enterprise customers can now deploy Claude 4 agents that handle entire workflows from customer support to code review pipelines."
    },
    {
      "title": "Google DeepMind Achieves AGI Milestone with Gemini Ultra 2",
      "url": "https://deepmind.google/research/gemini-ultra-2-agi-breakthrough",
      "publishedDate": "2026-01-05T11:30:00.000Z",
      "author": "Demis Hassabis",
      "score": 0.9756,
      "text": "Google DeepMind announced that Gemini Ultra 2 has passed a comprehensive battery of tests measuring artificial general intelligence, including novel scientific reasoning and cross-domain transfer learning assessments. The model successfully designed and validated a new pharmaceutical compound in collaboration with researchers at Stanford Medical School. DeepMind emphasized that the system operates under strict safety constraints developed over three years of alignment research."
    },
    {
      "title": "OpenAI GPT-6 Introduces Real-Time Multimodal Reasoning Across Video Streams",
      "url": "https://openai.com/blog/gpt-6-multimodal-video-reasoning",
      "publishedDate": "2026-01-04T09:00:00.000Z",
      "author": "OpenAI Research",
      "score": 0.9623,
      "text": "OpenAI launched GPT-6 with native real-time video understanding, enabling users to have conversations about live video feeds with sub-second latency. The model can analyze security footage, assist with live surgeries, and provide real-time sports commentary with expert-level accuracy. Initial availability is limited to enterprise customers with pricing starting at $0.15 per minute of video processed."
    },
    {
      "title": "China's Baidu Releases Ernie 5.0 Claiming Parity with Western Frontier Models",
      "url": "https://www.scmp.com/tech/baidu-ernie-5-release-january-2026",
      "publishedDate": "2026-01-03T22:15:00.000Z",
      "author": "South China Morning Post Technology Desk",
      "score": 0.9234,
      "text": "Baidu officially released Ernie 5.0, claiming performance parity with GPT-6 and Claude 4 on Chinese language benchmarks and competitive results on English evaluations. The model was trained on a new domestically produced AI accelerator cluster, reducing China's dependency on NVIDIA hardware. Independent researchers have begun evaluating the claims, with early results suggesting strong performance on mathematical reasoning and coding tasks."
    },
    {
      "title": "US Senate Passes Comprehensive AI Regulation Framework",
      "url": "https://www.wsj.com/politics/us-senate-ai-regulation-bill-2026",
      "publishedDate": "2026-01-02T14:00:00.000Z",
      "author": "Wall Street Journal",
      "score": 0.8945,
      "text": "The US Senate passed the Artificial Intelligence Safety and Innovation Act with bipartisan support, establishing mandatory safety testing requirements for frontier AI models and creating a new federal agency to oversee AI development. Companies training models above 10^26 FLOPs must now submit to government safety evaluations before deployment. The legislation also allocates $50 billion for public AI research and workforce retraining programs over the next five years."
    }
  ]
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
