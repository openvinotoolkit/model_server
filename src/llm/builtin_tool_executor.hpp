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
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "io_processing/base_output_parser.hpp"

namespace ovms {

// Result of executing a built-in tool
struct BuiltInToolResult {
    std::string toolCallId;   // ID of the tool call this result corresponds to
    std::string toolName;     // Name of the tool that was executed
    std::string content;      // Result content from the tool execution
    bool success;             // Whether the tool execution was successful
};

using BuiltInToolResults_t = std::vector<BuiltInToolResult>;

// Built-in tool executor that handles execution of tools like browser, code_interpreter, etc.
// Currently provides mock implementations for development and testing purposes.
class BuiltInToolExecutor {
public:
    using ToolHandler = std::function<std::string(const std::string& arguments)>;

    BuiltInToolExecutor();

    // Execute all built-in tool calls and return results
    BuiltInToolResults_t execute(const ToolCalls_t& builtInToolCalls);

    // Check if a tool name corresponds to a built-in tool
    bool isBuiltInTool(const std::string& toolName) const;

    // Register a custom handler for a built-in tool (useful for testing or future extensions)
    void registerHandler(const std::string& toolName, ToolHandler handler);

private:
    std::unordered_map<std::string, ToolHandler> handlers;

    // Default mock handlers for built-in tools
    static std::string handleBrowserSearch(const std::string& arguments);
    static std::string handleBrowserOpen(const std::string& arguments);
    static std::string handleCodeInterpreter(const std::string& arguments);
    static std::string handleFileSearch(const std::string& arguments);
    static std::string handleImageGeneration(const std::string& arguments);

    // Helper to parse JSON arguments
    static std::string getArgumentValue(const std::string& arguments, const std::string& key);
};

}  // namespace ovms
