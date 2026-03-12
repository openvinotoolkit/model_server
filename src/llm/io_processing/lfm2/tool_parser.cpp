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

Argument Lfm2ToolParser::parseSingleArgument(std::string& argumentStr){
    Argument argument;

    size_t equalPos = argumentStr.find('=');
    if (equalPos != std::string::npos) {
        argument.name = argumentStr.substr(0, equalPos);
        if (argumentStr[equalPos + 1] == '\"') {
            argument.value = argumentStr.substr(equalPos + 2, argumentStr.length() - equalPos - 3);
            argument.type = ParameterType::STRING;
        // probably it should be also a case for object or array - no info yet 
        } else {
            argument.value = argumentStr.substr(equalPos + 1);
            if (argument.value == "true" || argument.value == "false") {
                argument.type = ParameterType::BOOLEAN;
            } else {
                argument.type = ParameterType::NUMBER;
            }
        }
    } else {
        argument.name = argumentStr;
        argument.value = "";
        argument.type = ParameterType::UNKNOWN;
    }
    return argument;
}

std::vector<Argument> Lfm2ToolParser::parseArguments(std::string& argumentsStr) {
    const std::string toolArgSeparatorStr = ", ";
    std::vector<std::string> args;
    std::vector<Argument> parsedArgs;

    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = argumentsStr.find(toolArgSeparatorStr, argPos);
        if (commaPos == std::string::npos) {
            args.push_back(argumentsStr.substr(argPos));
            break;
        }
        args.push_back(argumentsStr.substr(argPos, commaPos - argPos));
        argPos = commaPos + toolArgSeparatorStr.length();
    }

    for (std::string& arg : args) {
        parsedArgs.push_back(parseSingleArgument(arg));
    }
    return parsedArgs;
}

void Lfm2ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;
    size_t pos = 0;
    
    const std::string toolListStartIndicator = "[";
    const std::string toolListEndIndicator = "]";
    const std::string toolArgsStartIndicator = "(";
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
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed tool list string: {}", toolListStr);
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
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed single tool string nr {}: {}", guard, singleTool);
            }
            guard++;
            if (singleTool.empty()) {
                continue;
            } else {
                tools.push_back(singleTool);
            }
        }
        

        for (std::string& tool : tools) {
            ToolCall toolCall;
            size_t argsPos = tool.find(toolArgsStartIndicator);
            if (argsPos != std::string::npos) {
                std::string toolName = tool.substr(0, argsPos);                
                size_t argsStartPos = singleTool.find(toolArgsStartIndicator);
                std::string toolName = singleTool.substr(0, argsStartPos);
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed tool name: {}", toolName);
                std::string argsStr = singleTool.substr(argsStartPos + toolArgsStartIndicator.length(), singleTool.length() - argsStartPos - toolArgsStartIndicator.length() - toolEndIndicator.length());
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed args string: {}", argsStr);
                std::vector<Argument> arguments = parseArguments(argsStr);
                
                toolCall.name = toolName;
                rapidjson::Document argsDoc(rapidjson::kObjectType);
                rapidjson::StringBuffer sb;
                rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
                argsWriter.StartObject();
                for (const Argument& argument : arguments) {
                    argsWriter.Key(argument.name.c_str());
                    argsWriter.StartObject();
                    if (argument.type == ParameterType::STRING) {
                        argsWriter.String(argument.value.c_str());
                    } else if (argument.type == ParameterType::BOOLEAN) {
                        bool boolValue = (argument.value == "true");
                        argsWriter.Bool(boolValue);
                    } else if (argument.type == ParameterType::NUMBER) {
                        argsWriter.RawNumber(argument.value.c_str(), argument.value.length());
                    }
                }                
                argsWriter.EndObject();
                toolCall.arguments = sb.GetString();
                toolCall.id = generateRandomId();
                parsedOutput.toolCalls.push_back(toolCall);
            }
        }
    }
}