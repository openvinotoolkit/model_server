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
#include "tool_parser.hpp"

namespace ovms {

void Lfm2ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;
    size_t pos = 0;
    
    const std::string toolListStartIndicator = "[";
    const std::string toolListEndIndicator = "]";
    const std::string toolEndIndicator = ")";
    const std::string toolSeparatorStr = ", ";

    while (true) {
        size_t start = parsedOutput.content.find(toolCallStartTag, pos);
        if (start == std::string::npos) {
            break;
        }
        start += toolCallStartTag.length();
        size_t end = parsedOutput.content.find(toolCallEndTag, start);
        if(end == std::string::npos) {
            break;
        }
        std::string toolListStr = parsedOutput.content.substr(start + toolListStartIndicator.length(), end - start - toolListStartIndicator.length() - toolListEndIndicator.length());
        uint32_t guard = 0;
        //current state func_name1(arg1=value1), func_name2(arg1=value1) or func_name(arg1=value1, arg2=value2)
        while (!toolListStr.empty() && guard < 100) {
            size_t toolEndPos = toolListStr.find(toolEndIndicator);
            std::string singleTool;
            if (toolEndPos != std::string::npos) {
                singleTool = toolListStr.substr(0, toolEndPos);
                if(toolEndPos + toolEndIndicator.length() < toolListStr.length()) {
                    toolListStr = toolListStr.substr(toolEndPos + toolEndIndicator.length() + toolSeparatorStr.length());
                } else {
                    toolListStr.clear();
                }
            }
            guard++;
        }
        

        for (std::string& tool : tools) {
            size_t args_pos = tool.find('(');
            if (args_pos != std::string::npos) {
                std::string tool_name = tool.substr(0, args_pos);
                std::string args_str = tool.substr(args_pos+1, tool.length() - args_pos - 2);

                std::vector<std::string> args;
                size_t arg_pos = 0;
                while (arg_pos < args_str.length()) {
                    size_t comma_pos = args_str.find(',', arg_pos);
                    if (comma_pos == std::string::npos) {
                        args.push_back(args_str.substr(arg_pos));
                        break;
                    }
                    args.push_back(args_str.substr(arg_pos, comma_pos - arg_pos));
                    arg_pos = comma_pos + 1;
                }
            }
        }


        if (!singleTool.empty()) {
            tools.push_back(singleTool);
        }
    }
}