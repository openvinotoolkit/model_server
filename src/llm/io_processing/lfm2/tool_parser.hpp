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

#include "src/llm/io_processing/base_output_parser.hpp"


namespace ovms {
class Lfm2ToolParser : public BaseOutputParser {
protected:
    const std::string toolCallStartTag = "<|tool_call_start|>";
    const std::string toolCallEndTag = "<|tool_call_end|>";
    const std::string toolListStartTag = "<|tool_list_start|>";
    const std::string toolListEndTag = "<|tool_list_end|>";
    const std::string toolRepsonseStartTag = "<|tool_response_start|>";
    const std::string toolResponseEndTag = "<|tool_response_end|>";

    const std::string toolListStartIndicator = "[";
    const std::string toolListEndIndicator = "]";
    const std::string toolArgsStartIndicator = "(";
    const std::string toolEndIndicator = ")";
    const std::string toolSeparatorStr = ", ";

    enum class State {
        Content,
        ToolCallStarted,
        ToolCallFunctionName,
        ToolCallParameters,
        ToolCallEnded
    };

public:
    struct Argument {
        std::string name;
        std::string value;
        bool isValid = true;
    };
    Lfm2ToolParser() = delete;
    explicit Lfm2ToolParser(ov::genai::Tokenizer& tokenizer) : BaseOutputParser(tokenizer) {}

    void parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) override;
    std::optional<ToolCalls_t> parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) override;
    const std::vector<std::string>& getParsingStartTags() const override {
        static const std::vector<std::string> parsingStartTags = {toolCallStartTag};
        return parsingStartTags;
    }

    const std::vector<std::string>& getSpecialParsingStartTags() const override {
        static const std::vector<std::string> beginningOnlyTags = {};
        return beginningOnlyTags;
    }

    const std::string& getParsingEndTag() const override {
        return toolCallEndTag;
    }

private:
    void writeArgumentOfAnyType(const std::string& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);
    void writeArgumentOfAnyType(const rapidjson::Value& arg, rapidjson::Writer<rapidjson::StringBuffer>& writer);
    Argument parseSingleArgument(const std::string& argumentStr);
    std::vector<Argument> parseArguments(const std::string& argumentsStr);
    bool parseSingleToolCall(const std::string& toolStr, ToolCall& toolCall);

    std::string streamingContent;
};
}