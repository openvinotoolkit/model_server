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
#include "../utils.hpp"
#include "../../../logging.hpp"

namespace ovms {

void Lfm2ToolParser::writeArgumentOfAnyType(const rapidjson::Value& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer) {
    if (arg.IsString()) {
        writer.String(arg.GetString());
    } else if (arg.IsInt64()) {
        writer.Int64(arg.GetInt64());
    } else if (arg.IsDouble()) {
        writer.Double(arg.GetDouble());
    } else if (arg.IsBool()) {
        writer.Bool(arg.GetBool());
    } else if (arg.IsArray()) {
        writer.StartArray();
        for (auto& elem : arg.GetArray()) {
            writeArgumentOfAnyType(elem, writer);
        }
        writer.EndArray();
    } else if (arg.IsObject()) {
        writer.StartObject();
        for (auto it = arg.MemberBegin(); it != arg.MemberEnd(); ++it) {
            writer.Key(it->name.GetString());
            writeArgumentOfAnyType(it->value, writer);
        }
        writer.EndObject();
    } else {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Argument has unsupported type.");
    }
}

void Lfm2ToolParser::writeArgumentOfAnyType(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer) {
    std::string normalized = arg;
    if (arg.find('{') != std::string::npos || arg.find('}') != std::string::npos) {
        std::replace(normalized.begin(), normalized.end(), '\'', '"');
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Argument contains curly braces, replaced single quotes with double quotes for JSON parsing. Modified string: {}", normalized);
    } 

    rapidjson::Document doc;
    doc.Parse(normalized.c_str());
    
    if (doc.HasParseError()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to parse argument string as JSON. Argument string: {}", normalized);
        return;
    }

    rapidjson::Value& argumentDoc = doc;
    writeArgumentOfAnyType(argumentDoc, writer);
}

Lfm2ToolParser::Argument Lfm2ToolParser::parseSingleArgument(const std::string& argumentStr){
    Lfm2ToolParser::Argument argument;

    size_t equalPos = argumentStr.find('=');
    if (equalPos != std::string::npos) {
        argument.name = argumentStr.substr(0, equalPos);
        argument.value = argumentStr.substr(equalPos + 1);
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed argument - name: {}, value: {}", argument.name, argument.value);
    } else {
        argument.name = argumentStr;
        argument.value = "";
        argument.isValid = false;
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Argument string: {} does not contain '=', setting name as entire string and value as empty", argumentStr);
    }
    return argument;
}

std::vector<Lfm2ToolParser::Argument> Lfm2ToolParser::parseArguments(const std::string& argumentsStr) {
    std::vector<std::string> args;
    std::vector<Lfm2ToolParser::Argument> parsedArgs;

    size_t argPos = 0;
    while (argPos < argumentsStr.length()) {
        size_t commaPos = std::string::npos;
        int bracketDepth = 0;
        int braceDepth = 0;
        
        for (size_t i = argPos; i < argumentsStr.length(); ++i) {
            if (argumentsStr[i] == '{') {
                braceDepth++;
            } else if (argumentsStr[i] == '}') {
                braceDepth--;
            } else if (argumentsStr[i] == '[') {
                bracketDepth++;
            } else if (argumentsStr[i] == ']') {
                bracketDepth--;
            } else if (argumentsStr.substr(i, toolSeparatorStr.length()) == toolSeparatorStr && 
                       bracketDepth == 0 && braceDepth == 0) {
                commaPos = i;
                break;
            }
        }
        
        if (commaPos == std::string::npos) {
            auto remainingStr = argumentsStr.substr(argPos);
            args.push_back(remainingStr);
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "No more commas found, adding remaining argument string: {}", remainingStr);
            break;
        }
        auto argStr = argumentsStr.substr(argPos, commaPos - argPos);
        args.push_back(argStr);
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed argument string: {}", argStr);
        argPos = commaPos + toolSeparatorStr.length();
    }

    for (const std::string& arg : args) {
        parsedArgs.push_back(parseSingleArgument(arg));
    }
    return parsedArgs;
}

bool Lfm2ToolParser::parseNewContent() {
    switch(this->currentState) { // to think about spliting every state into separate functions if it grows too much
        case State::Content: {
            auto currentStr = this->streamingContent.substr(this->streamingPosition);
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Current state: Content. Streaming content from current position: {}", currentStr);
            size_t pos = this->streamingContent.find(toolCallStartTag, this->streamingPosition);
            size_t toolCallEndTagPos = this->streamingContent.find(toolCallEndTag, this->streamingPosition);
            if (toolCallEndTagPos != std::string::npos && pos == std::string::npos) {
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Detected end of tool call at position: {}", toolCallEndTagPos);
                this->streamingPosition = toolCallEndTagPos + toolCallEndTag.length();
                return false;
            }
            if (pos != std::string::npos) {
                this->streamingPosition = pos + toolCallStartTag.length() + toolListStartIndicator.length();
                this->currentState = State::ToolCallStarted;
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Detected start of tool call at position: {}", pos);
                return false;
            }

            return true;
            break;
        }
        case State::ToolCallStarted: {
            size_t pos = this->streamingContent.find(toolArgsStartIndicator, this->streamingPosition);
            if (pos != std::string::npos) {
                std::string toolName = this->streamingContent.substr(this->streamingPosition, pos - this->streamingPosition);
                this->toolCall = ToolCall{generateRandomId(), toolName, ""};
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed tool name: {}", toolName);
                this->streamingPosition = pos + toolArgsStartIndicator.length();
                this->currentState = State::ToolCallParameters;
                toolCallIndex++;
                return true;
            }
            break;
        }
        case State::ToolCallParameters: {
            size_t pos = this->streamingContent.find(toolArgsEndIndicator, this->streamingPosition);
            if (pos != std::string::npos) {
                std::string argumentsStr = this->streamingContent.substr(this->streamingPosition, pos - this->streamingPosition);
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed arguments string: {}", argumentsStr);
                std::vector<Argument> arguments = parseArguments(argumentsStr);

                rapidjson::Document argsDoc(rapidjson::kObjectType);
                rapidjson::StringBuffer sb;
                rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
                argsWriter.StartObject();
                for (const Argument& argument : arguments) {
                    argsWriter.Key(argument.name.c_str());
                    writeArgumentOfAnyType(argument.value, argsWriter);
                }                
                argsWriter.EndObject();
                this->toolCall.arguments = sb.GetString();
                this->toolCalls.push_back(this->toolCall);
                this->currentState = State::ToolCallEnded;
                this->streamingPosition = pos + toolArgsEndIndicator.length();
                return true;
            }
            break;
        }
        case State::ToolCallEnded: {
            size_t pos = this->streamingContent.find(toolListEndIndicator, this->streamingPosition);
            size_t toolSeparatorPos = this->streamingContent.find(toolSeparatorStr, this->streamingPosition);
            size_t toolCallEndTagPos = this->streamingContent.find(toolCallEndTag, this->streamingPosition);
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Current state: ToolCallEnded. Streaming content from current position: {}", this->streamingContent.substr(this->streamingPosition));
            if (pos == std::string::npos && toolSeparatorPos == std::string::npos && toolCallEndTagPos == std::string::npos) {
                return false;
            } else if (toolSeparatorPos != std::string::npos && toolSeparatorPos < pos) {
                this->streamingPosition = toolSeparatorPos + toolSeparatorStr.length();
                this->currentState = State::ToolCallStarted;
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Detected separator between tool calls at position: {}, expecting another tool call to start", toolSeparatorPos);
            } else if (toolCallEndTagPos != std::string::npos) {
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Detected end of tool call at position: {}", toolCallEndTagPos);
                this->streamingPosition = toolCallEndTagPos + toolCallEndTag.length();
                this->currentState = State::AfterToolCall;
            } else {
                this->streamingPosition = pos + toolListEndIndicator.length();
                this->currentState = State::AfterToolCall;
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Detected end of tool list at position: {}, returning to content state", pos);
            }
            return true;
        }
        case State::AfterToolCall: break;
    }
    return false;
}

rapidjson::Document Lfm2ToolParser::wrapDeltaContent(const std::string& content)
{
    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Value deltaObj(rapidjson::kObjectType);
    deltaObj.AddMember("content", rapidjson::Value(content.c_str(), doc.GetAllocator()), doc.GetAllocator());
    doc.AddMember("delta", deltaObj, doc.GetAllocator());
    return doc;
}

rapidjson::Document Lfm2ToolParser::wrapDeltaArgs(const std::string& argsStr, int toolCallIndex)
{
    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Document argsDoc;
    argsDoc.Parse(argsStr.c_str());
    if (!argsDoc.HasParseError()) {
        doc.AddMember("arguments", argsDoc, doc.GetAllocator());
    } else {
        doc.AddMember("arguments", rapidjson::Value(argsStr.c_str(), doc.GetAllocator()), doc.GetAllocator());
    }
    return BaseOutputParser::wrapDelta(doc, toolCallIndex);
}

std::optional<rapidjson::Document> Lfm2ToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    if (chunk.empty()) {
        return std::nullopt;
    }

    this->streamingContent += chunk;

    if (parseNewContent()) {
        if (this->currentState == State::ToolCallParameters) {
            return BaseOutputParser::wrapFirstDelta(this->toolCall.name, toolCallIndex);
        }
        if (this->currentState == State::ToolCallEnded) {
            return wrapDeltaArgs(this->toolCall.arguments, toolCallIndex);
        }
        if (this->currentState == State::Content) {
            auto content = this->streamingContent.substr(this->streamingPosition);
            return wrapDeltaContent(content);
        }
        if (this->currentState == State::AfterToolCall) {
            this->currentState = State::Content;
        }
    }

    return std::nullopt;
}

bool Lfm2ToolParser::parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall) {
    size_t argsPos = toolStr.find(toolArgsStartIndicator);
    if (argsPos != std::string::npos) {
        std::string toolName = toolStr.substr(0, argsPos);                
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed tool name: {}", toolName);

        int argsStrLen = toolStr.length() - argsPos - toolArgsStartIndicator.length() - toolArgsEndIndicator.length();
        std::string argsStr = toolStr.substr(argsPos + toolArgsStartIndicator.length(), argsStrLen);
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed args string: {}", argsStr);
        std::vector<Lfm2ToolParser::Argument> arguments = parseArguments(argsStr);

        toolCall.name = toolName;
        rapidjson::Document argsDoc(rapidjson::kObjectType);
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> argsWriter(sb);
        argsWriter.StartObject();
        for (const Lfm2ToolParser::Argument& argument : arguments) {
            argsWriter.Key(argument.name.c_str());
            writeArgumentOfAnyType(argument.value, argsWriter);
        }                
        argsWriter.EndObject();
        toolCall.arguments = sb.GetString();
        toolCall.id = generateRandomId();
        return true;
    }
    return false;
}

void Lfm2ToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    std::vector<std::string> tools;
    std::vector<std::pair<size_t, size_t>> toolCallPositions;
    size_t pos = 0;
    uint8_t main_guard = 0;
    while (pos != std::string::npos && main_guard < 100) {
        std::pair<size_t, size_t> toolCallPosition;
        size_t start = parsedOutput.content.find(toolCallStartTag, pos);
        if (start == std::string::npos) {
            break;
        }
        toolCallPosition.first = start;
        start += toolCallStartTag.length();
        size_t end = parsedOutput.content.find(toolCallEndTag, start);
        if(end == std::string::npos) {
            end = parsedOutput.content.rfind(toolListEndIndicator, end);
            if (end == std::string::npos) {
                SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Malformed tool call in content, no tool end tag or tool list end tag found for tool call starting at position {}", start);
                break;
            }
           toolCallPosition.second = end + toolListEndIndicator.length();
           end += toolListEndIndicator.length();
           SPDLOG_LOGGER_INFO(llm_calculator_logger, "No tool call end tag found, but found tool list end tag, treating content between start and this position as tool list");
        } else { 
            toolCallPosition.second = end + toolCallEndTag.length();
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Found tool call end tag for tool call starting at position {}", start);
        }
        toolCallPositions.push_back(toolCallPosition);
        std::string toolListStr = parsedOutput.content.substr(start + toolListStartIndicator.length(), end - start - toolListStartIndicator.length() - toolListEndIndicator.length());
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed tool list string: {}", toolListStr);
        uint8_t guard = 0;
        while (!toolListStr.empty() && guard < 100) {
            size_t toolEndPos = toolListStr.find(toolArgsEndIndicator);
            std::string singleTool;
            if (toolEndPos != std::string::npos) {
                singleTool = toolListStr.substr(0, toolEndPos + toolArgsEndIndicator.length());
                if(toolEndPos + toolArgsEndIndicator.length() < toolListStr.length()) {
                    toolListStr = toolListStr.substr(toolEndPos + toolArgsEndIndicator.length() + toolSeparatorStr.length());
                } else {
                    toolListStr.clear();
                }
                SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed single tool string nr {}: {}", guard, singleTool);
            }
            guard++;
            if (singleTool.empty()) {
                continue;
            } else {
                tools.push_back(singleTool);
            }
        }
        if (guard == 100) {
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Guard limit reached while parsing tools, possible malformed tool list string: {}", toolListStr);
        }

        pos = toolCallPositions.empty() ? std::string::npos : toolCallPositions.back().second;
        main_guard++;
    }

    for (const std::string& tool : tools) {
        ToolCall toolCall;
        auto wasToolCallParsed = parseSingleToolCall(tool, toolCall);
        if (wasToolCallParsed) {
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed tool call - name: {}, args: {}", toolCall.name, toolCall.arguments);
            parsedOutput.toolCalls.push_back(toolCall);
        } else
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Failed to parse tool call from string: {}", tool);
    }

    for(int i = toolCallPositions.size() - 1; i >= 0; i--) {
         const std::pair<size_t, size_t>& toolCallPosition = toolCallPositions[i];
         parsedOutput.content.erase(toolCallPosition.first, toolCallPosition.second - toolCallPosition.first);
    }
}
}