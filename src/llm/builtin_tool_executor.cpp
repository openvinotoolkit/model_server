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

#include <mcp_sse_client.h>
#include <mcp_message.h>
#include <mcp_logger.h>

#include "../logging.hpp"
#include "src/port/rapidjson_document.hpp"

namespace ovms {

BuiltInToolExecutor::BuiltInToolExecutor() {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "BuiltInToolExecutor: Initializing with default mock handlers");
    // Register default mock handlers for built-in tools
    // Tool names follow the pattern: <category>.<action> (e.g., browser.search, browser.open)
    handlers["browser.search"] = handleBrowserSearch;
    handlers["browser.open"] = handleBrowserOpen;
    // code_interpreter uses mock by default, will be overridden when MCP client is initialized
    handlers["code_interpreter"] = handleCodeInterpreterMock;
    handlers["code_interpreter.run"] = handleCodeInterpreterMock;
    handlers["file_search"] = handleFileSearch;
    handlers["file_search.search"] = handleFileSearch;
    handlers["image_generation"] = handleImageGeneration;
    handlers["image_generation.generate"] = handleImageGeneration;
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "BuiltInToolExecutor: Registered {} handlers (MCP not connected, using mocks)", handlers.size());
}

BuiltInToolExecutor::~BuiltInToolExecutor() {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "BuiltInToolExecutor: Destructor called, cleaning up");
    disconnectMcpClient();
}

bool BuiltInToolExecutor::initializeMcpClient(const std::string& url, const std::string& sseEndpoint) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "BuiltInToolExecutor::initializeMcpClient called with url={}, sseEndpoint={}", url, sseEndpoint);
    std::lock_guard<std::mutex> lock(mcpClientMutex);

    if (mcpClientInitialized) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "MCP client already initialized, skipping");
        return true;
    }

    try {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Creating MCP SSE client instance...");
        mcpClient = std::make_unique<mcp::sse_client>(url, sseEndpoint);
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP SSE client instance created successfully");

        // Set capabilities
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Setting MCP client capabilities...");
        mcp::json capabilities = {
            {"roots", {{"listChanged", true}}}};
        mcpClient->set_capabilities(capabilities);
        mcpClient->set_timeout(30);  // 30 second timeout
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "MCP client timeout set to 30 seconds");

        // Initialize the connection
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Calling MCP client initialize with client name 'ovms-builtin-tool-executor'...");
        bool initialized = mcpClient->initialize("ovms-builtin-tool-executor", mcp::MCP_VERSION);

        SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP client initialize returned: {}", initialized);
        if (initialized) {
            // Verify connection with ping
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Verifying MCP connection with ping...");
            bool pingResult = mcpClient->ping();
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP ping result: {}", pingResult ? "SUCCESS" : "FAILED");
            if (pingResult) {
                mcpClientInitialized = true;

                // Register the real Python execution handler
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Registering real Python execution handlers...");
                auto pythonHandler = [this](const std::string& args) {
                    return this->handlePythonExecution(args);
                };
                handlers["python"] = pythonHandler;
                handlers["code_interpreter"] = pythonHandler;
                handlers["code_interpreter.run"] = pythonHandler;

                SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP client initialized successfully! Python/code_interpreter tools now use REAL execution via MCP server");
                return true;
            } else {
                SPDLOG_LOGGER_ERROR(llm_calculator_logger, "MCP client ping failed - server may not be responding");
                mcpClient.reset();
                return false;
            }
        } else {
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "MCP client initialization failed - check server URL and availability");
            mcpClient.reset();
            return false;
        }
    } catch (const mcp::mcp_exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "MCP exception during initialization: {}", e.what());
        mcpClient.reset();
        return false;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Exception during MCP client initialization: {}", e.what());
        mcpClient.reset();
        return false;
    }
}

bool BuiltInToolExecutor::isMcpClientReady() const {
    std::lock_guard<std::mutex> lock(mcpClientMutex);
    return mcpClientInitialized && mcpClient != nullptr;
}

void BuiltInToolExecutor::disconnectMcpClient() {
    std::lock_guard<std::mutex> lock(mcpClientMutex);
    if (mcpClient) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Disconnecting MCP client and reverting to mock handlers...");
        mcpClient.reset();
        mcpClientInitialized = false;

        // Reset handlers back to mock implementations
        handlers["code_interpreter"] = handleCodeInterpreterMock;
        handlers["code_interpreter.run"] = handleCodeInterpreterMock;
        handlers.erase("python");
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP client disconnected, Python/code_interpreter tools reverted to MOCK mode");
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "disconnectMcpClient called but no MCP client was connected");
    }
}

// Helper to normalize tool name by stripping "functions." prefix if present
static std::string normalizeToolName(const std::string& toolName) {
    static const std::string functions_prefix = "functions.";
    if (toolName.compare(0, functions_prefix.length(), functions_prefix) == 0) {
        return toolName.substr(functions_prefix.length());
    }
    return toolName;
}

BuiltInToolResults_t BuiltInToolExecutor::execute(const ToolCalls_t& builtInToolCalls) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "BuiltInToolExecutor::execute called with {} tool calls, MCP connected: {}",
                       builtInToolCalls.size(), mcpClientInitialized ? "YES" : "NO");
    BuiltInToolResults_t results;
    results.reserve(builtInToolCalls.size());

    for (const auto& toolCall : builtInToolCalls) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Processing tool call: id={}, name={}, args_length={}",
                           toolCall.id, toolCall.name, toolCall.arguments.length());
        
        // Normalize tool name (strip "functions." prefix if present)
        std::string normalizedName = normalizeToolName(toolCall.name);
        if (normalizedName != toolCall.name) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Normalized tool name '{}' -> '{}'", toolCall.name, normalizedName);
        }
        
        BuiltInToolResult result;
        result.toolCallId = toolCall.id;
        result.toolName = normalizedName;

        auto it = handlers.find(normalizedName);
        if (it != handlers.end()) {
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Found handler for tool '{}', executing...", normalizedName);
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool arguments: {}", toolCall.arguments);
            try {
                result.content = it->second(toolCall.arguments);
                result.success = true;
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Tool '{}' executed successfully, result length: {} chars", normalizedName, result.content.length());
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool '{}' result: {}", normalizedName, result.content);
            } catch (const std::exception& e) {
                result.content = std::string("Error executing tool: ") + e.what();
                result.success = false;
                SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Tool '{}' execution FAILED with exception: {}", normalizedName, e.what());
            }
        } else {
            // Unknown built-in tool - provide a generic mock response
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "No handler found for tool '{}', providing generic mock response", normalizedName);
            result.content = "Mock response for unknown tool: " + normalizedName;
            result.success = true;
        }

        results.push_back(std::move(result));
    }

    return results;
}

bool BuiltInToolExecutor::isBuiltInTool(const std::string& toolName) const {
    // Normalize tool name first (strip "functions." prefix if present)
    std::string normalizedName = normalizeToolName(toolName);
    
    // Built-in tools typically have a category prefix like "browser.", "code_interpreter", etc.
    // or are explicitly registered
    if (handlers.find(normalizedName) != handlers.end()) {
        return true;
    }
    // Check for category-based built-in tools
    return normalizedName.find("browser.") == 0 ||
           normalizedName.find("code_interpreter") == 0 ||
           normalizedName.find("file_search") == 0 ||
           normalizedName.find("image_generation") == 0 ||
           normalizedName == "python";
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

std::string BuiltInToolExecutor::handlePythonExecution(const std::string& arguments) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "handlePythonExecution called, arguments: [{}]", arguments);
    std::string code = arguments;//getArgumentValue(arguments, "code");
    if (code.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "No 'code' argument found, trying 'input'");
        code = getArgumentValue(arguments, "input");
    }

    if (code.empty()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "No code provided in arguments, returning error");
        return R"({"status": "error", "error": "No code provided"})";
    }

    SPDLOG_LOGGER_INFO(llm_calculator_logger, "Python code to execute ({} chars): {}", code.length(), code.substr(0, 200));

    std::lock_guard<std::mutex> lock(mcpClientMutex);

    if (!mcpClientInitialized || !mcpClient) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "MCP client not ready (initialized={}, client={}), falling back to mock response",
                          mcpClientInitialized, mcpClient ? "exists" : "null");
        return handleCodeInterpreterMock(arguments);
    }

    try {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Calling MCP server to execute Python code...");

        mcp::json args = mcp::json::object();
        args["code"] = code;

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "MCP call_tool('python', ...) starting");
        mcp::json result = mcpClient->call_tool("python", args);
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "MCP call_tool completed");

        SPDLOG_LOGGER_INFO(llm_calculator_logger, "MCP Python execution raw result: {}", result.dump());

        // Extract the result from MCP response
        // MCP response format: {"content": [{"type": "text", "text": "..."}], "structuredContent": {...}, "isError": bool}
        std::stringstream ss;
        ss << R"({"status": )";

        bool isError = false;
        if (result.contains("isError") && result["isError"].is_boolean()) {
            isError = result["isError"].get<bool>();
        }

        if (isError) {
            ss << R"("error")";
        } else {
            ss << R"("success")";
        }

        // Extract output text
        std::string outputText;
        if (result.contains("content") && result["content"].is_array()) {
            for (const auto& content : result["content"]) {
                if (content.contains("type") && content["type"] == "text" && content.contains("text")) {
                    outputText += content["text"].get<std::string>();
                }
            }
        } else if (result.contains("structuredContent") && result["structuredContent"].contains("result")) {
            outputText = result["structuredContent"]["result"].get<std::string>();
        }

        // Escape the output for JSON
        std::string escapedOutput;
        for (char c : outputText) {
            switch (c) {
            case '"':
                escapedOutput += "\\\"";
                break;
            case '\\':
                escapedOutput += "\\\\";
                break;
            case '\n':
                escapedOutput += "\\n";
                break;
            case '\r':
                escapedOutput += "\\r";
                break;
            case '\t':
                escapedOutput += "\\t";
                break;
            default:
                escapedOutput += c;
            }
        }

        ss << R"(, "output": ")" << escapedOutput << R"("})";
        std::string finalResult = ss.str();
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Python execution completed successfully, output length: {} chars", outputText.length());
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Final formatted result: {}", finalResult);
        return finalResult;

    } catch (const mcp::mcp_exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "MCP exception during Python execution: {} (type: mcp_exception)", e.what());
        std::stringstream ss;
        ss << R"({"status": "error", "error": "MCP error: )" << e.what() << R"("})";
        return ss.str();
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Standard exception during Python execution: {} (type: std::exception)", e.what());
        std::stringstream ss;
        ss << R"({"status": "error", "error": ")" << e.what() << R"("})";
        return ss.str();
    }
}

std::string BuiltInToolExecutor::handleCodeInterpreterMock(const std::string& arguments) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "handleCodeInterpreterMock called (MOCK MODE - MCP not connected)");
    std::string code = getArgumentValue(arguments, "code");
    if (code.empty()) {
        code = getArgumentValue(arguments, "input");
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Mock code_interpreter received code: {}", code.substr(0, 200));

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
