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
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/onyx/onyx_tool_parser.hpp"
#include "src/llm/io_processing/output_parser.hpp"
#include "src/logging.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

// TODO @atobiszei replace when tokenizer is available
#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\facebook\\opt-125m";
#else
const std::string tokenizerPath = "/ovms/src/test/llm_testing/facebook/opt-125m";
#endif

static std::unique_ptr<ov::genai::Tokenizer> opt125mTokenizer;

// Tool schemas drive argument typing (string vs integer vs object) exactly like
// Qwen3CoderOutputParserTest -- Onyx's ATEM parameter values are untyped raw text, so the
// parser must consult these to decide how to serialize each value into the JSON args blob.
static std::map<std::string, std::string> toolSchemasInput = {
    {"get_weather", R"({"properties":{"location":{"type":"string","description":"City name."},"unit":{"type":"string","description":"Temperature unit."}},"required":["location"]})"},
    {"get_time", R"({"properties":{"city":{"type":"string","description":"City name."}},"required":["city"]})"},
    {"get_current_location", R"({"properties":{},"required":[]})"},
    {"string_tool", R"({"properties":{"arg1":{"type":"string","description":"A string argument."}},"required":["arg1"]})"},
    {"cd", R"({"properties":{"folder":{"type":"string","description":"Path"}},"required":["folder"]})"},
    {"string_int_tool", R"({"properties":{"arg1":{"type":"string","description":"A string argument."},"arg2":{"type":"integer","description":"An integer argument."}},"required":["arg1","arg2"]})"}};

static std::vector<std::unique_ptr<rapidjson::Document>> schemaDocsStorage;

static ToolsSchemas_t convertStringToolSchemasStringToToolsSchemas(
    const std::map<std::string, std::string>& input) {
    ToolsSchemas_t result;
    schemaDocsStorage.clear();
    for (const auto& [name, schemaStr] : input) {
        auto schemaDoc = std::make_unique<rapidjson::Document>();
        if (schemaDoc->Parse(schemaStr.c_str()).HasParseError()) {
            throw std::runtime_error("Failed to parse schema for tool: " + name);
        }
        result[name] = {schemaDoc.get(), schemaStr};
        schemaDocsStorage.push_back(std::move(schemaDoc));
    }
    return result;
}

static ovms::ToolsSchemas_t toolsSchemas = convertStringToolSchemasStringToToolsSchemas(toolSchemasInput);

// -----------------------------------------------------------------------------
// ATEM block builders -- keep tool-call test inputs readable. A single tool call
// renders (mirroring the served template's render_atem, newlines included) as:
//   <atem:function_calls>\n<atem:invoke name="NAME">\n
//   <atem:parameter name="K">V</atem:parameter>\n   (repeated)
//   </atem:invoke>\n</atem:function_calls>
// The full generated turn wraps that in the harmony envelope
// " to=NAME<|message|>...{<|eom|>|<|eot|>}".
// -----------------------------------------------------------------------------
using AtemParams = std::vector<std::pair<std::string, std::string>>;

static std::string atemBlock(const std::string& name, const AtemParams& params) {
    std::string s = "<atem:function_calls>\n<atem:invoke name=\"" + name + "\">\n";
    for (const auto& [k, v] : params) {
        s += "<atem:parameter name=\"" + k + "\">" + v + "</atem:parameter>\n";
    }
    s += "</atem:invoke>\n</atem:function_calls>";
    return s;
}

// A full assistant tool-call turn as the model emits it: the leading " to=<name>" recipient,
// "<|message|>", the ATEM block, then the turn terminator.
static std::string onyxToolTurn(const std::string& name, const AtemParams& params, const std::string& terminator = "<|eom|>") {
    return " to=" + name + "<|message|>" + atemBlock(name, params) + terminator;
}

// Pre-parsed parameter-type map for the direct-impl parametrized test below (mirrors
// Qwen3CoderOutputParserTest's toolsParametersTypeMap) -- drives schema-based typing of the
// otherwise-untyped ATEM parameter values.
static ovms::ToolsParameterTypeMap_t onyxToolsParametersTypeMap = {
    {"string_tool", {{"arg1", ovms::ParameterType::STRING}}},
    {"int_tool", {{"arg1", ovms::ParameterType::NUMBER}}},
    {"float_tool", {{"arg1", ovms::ParameterType::NUMBER}}},
    {"bool_tool", {{"arg1", ovms::ParameterType::BOOLEAN}}},
    {"object_tool", {{"arg1", ovms::ParameterType::OBJECT}}},
    {"list_tool", {{"arg1", ovms::ParameterType::ARRAY}}}};

class OnyxOutputParserTest : public ::testing::Test {
protected:
    std::unique_ptr<OutputParser> outputParser;

    static void SetUpTestSuite() {
        try {
            opt125mTokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath);
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize opt-125m tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize opt-125m tokenizer due to unknown error.";
        }
    }

    static void TearDownTestSuite() {
        opt125mTokenizer.reset();
    }

    void SetUp() override {
        outputParser = std::make_unique<OutputParser>(*opt125mTokenizer, "onyx", "onyx", toolsSchemas);
    }

    ParsedOutput generateParsedOutput(const std::string& input, bool toolsAvailable = true) {
        auto generatedTensor = opt125mTokenizer->encode(input, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> generatedTokens(generatedTensor.data<int64_t>(), generatedTensor.data<int64_t>() + generatedTensor.get_size());
        return outputParser->parse(generatedTokens, toolsAvailable);
    }

    // Wraps a raw (unescaped) string value into a JSON object {"arg1":"<escaped>"},
    // using rapidjson's serializer to handle all escaping. This lets us write raw PLC/Python
    // code in tests without manually counting backslashes.
    static std::string wrapRawCodeAsToolArgs(const std::string& rawCode) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Value val(rawCode.c_str(), static_cast<rapidjson::SizeType>(rawCode.size()), doc.GetAllocator());
        doc.AddMember("arg1", val, doc.GetAllocator());
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        doc.Accept(writer);
        return buf.GetString();
    }

    // Builds the expected arguments delta JSON string for a given tool call index and
    // raw (unescaped) code content: {"delta":{"tool_calls":[{"index":N,"function":{"arguments":"..."}}]}}
    static std::string expectedArgsDelta(int index, const std::string& rawCode) {
        std::string argsJson = wrapRawCodeAsToolArgs(rawCode);
        // argsJson is a valid JSON object string -- we need to embed it as an escaped
        // string value inside the outer delta JSON. Use rapidjson to build the whole thing.
        rapidjson::Document outer;
        outer.SetObject();
        rapidjson::Document::AllocatorType& alloc = outer.GetAllocator();
        rapidjson::Value delta(rapidjson::kObjectType);
        rapidjson::Value toolCalls(rapidjson::kArrayType);
        rapidjson::Value tc(rapidjson::kObjectType);
        tc.AddMember("index", index, alloc);
        rapidjson::Value func(rapidjson::kObjectType);
        rapidjson::Value argsVal(argsJson.c_str(), static_cast<rapidjson::SizeType>(argsJson.size()), alloc);
        func.AddMember("arguments", argsVal, alloc);
        tc.AddMember("function", func, alloc);
        toolCalls.PushBack(tc, alloc);
        delta.AddMember("tool_calls", toolCalls, alloc);
        outer.AddMember("delta", delta, alloc);
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        outer.Accept(writer);
        return buf.GetString();
    }
};

// =============================================================================
// Reasoning / final-answer framing (UNCHANGED by the new drop).
// A single generate() call stops at the first "<|eom|>"/"<|eot|>" (both are configured as
// eos tokens for Onyx), so only one of these segment shapes is ever produced at a time.
// =============================================================================
TEST_F(OnyxOutputParserTest, FinalAnswerWithExplicitUserRecipient) {
    ParsedOutput parsedOutput = generateParsedOutput(" to=user<|message|>It's 65F in SF.<|eot|>");

    EXPECT_EQ(parsedOutput.content, "It's 65F in SF.");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(OnyxOutputParserTest, FinalAnswerWithoutRecipient) {
    // The template only renders " to=" when the caller supplies a recipient; the model may
    // also just emit "<|message|>...<|eot|>" directly with no recipient at all.
    ParsedOutput parsedOutput = generateParsedOutput("<|message|>It's 65F in SF.<|eot|>");

    EXPECT_EQ(parsedOutput.content, "It's 65F in SF.");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(OnyxOutputParserTest, LiteralToEqualsInContentNotStripped) {
    // "to=" appearing in the answer body must not be mistaken for an envelope prefix.
    ParsedOutput parsedOutput = generateParsedOutput(
        " to=user<|message|>Send email to=admin for help.<|eot|>");

    EXPECT_EQ(parsedOutput.content, "Send email to=admin for help.");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(OnyxOutputParserTest, PrivateReasoningOnly) {
    ParsedOutput parsedOutput = generateParsedOutput(" to=self<|message|>Let me think about this.<|eom|>");

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "Let me think about this.");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

// =============================================================================
// Unary tool-call parsing (ATEM). Arguments are typed via the tool schema.
// =============================================================================
TEST_F(OnyxOutputParserTest, ToolCallWithAtemArguments) {
    // Live-model shape: " to=<name><|message|><atem:function_calls>...</atem:function_calls><|eom|>".
    // Both params are strings per get_weather's schema, so both stay quoted.
    ParsedOutput parsedOutput = generateParsedOutput(
        onyxToolTurn("get_weather", {{"location", "Paris"}, {"unit", "celsius"}}));

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"location":"Paris","unit":"celsius"})");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(OnyxOutputParserTest, ToolCallWithNoArguments) {
    // No-argument call, exactly as captured live (get_location_gps/get_current_time).
    ParsedOutput parsedOutput = generateParsedOutput(onyxToolTurn("get_current_location", {}));

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_current_location");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{}");
}

TEST_F(OnyxOutputParserTest, ToolCallWithSchemaTypedIntegerArgument) {
    // arg2 is declared integer in the schema, so the untyped ATEM value "3141522" must be
    // serialized as a JSON number (not a quoted string), while arg1 stays a quoted string.
    ParsedOutput parsedOutput = generateParsedOutput(
        onyxToolTurn("string_int_tool", {{"arg1", "hello"}, {"arg2", "3141522"}}));

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "string_int_tool");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"arg1":"hello","arg2":3141522})");
}

TEST_F(OnyxOutputParserTest, PrivateReasoningThenToolCall) {
    // Realistic live shape (see chat_with_tools.py): in a SINGLE generation the model first emits
    // a private reasoning turn ("to=self...<|eom|>") and then a tool-call turn. The reasoning
    // parser must lift the reasoning out AND strip the following turn's harmony envelope, leaving
    // only the ATEM block for the tool parser to consume -- so content ends up empty. This is the
    // path the eos-token fix unblocked (previously "<|eom|>" was an eos token and generation
    // stopped after the reasoning turn, so the tool call was never produced).
    ParsedOutput parsedOutput = generateParsedOutput(
        " to=self<|message|>Let me check the location first.<|eom|>" +
        onyxToolTurn("get_current_location", {}, "<|eot|>"));

    EXPECT_EQ(parsedOutput.reasoning, "Let me check the location first.");
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_current_location");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{}");
}

TEST_F(OnyxOutputParserTest, PrivateReasoningThenToolCallWithArgs) {
    // Same combined reasoning+tool-call turn, but the tool call carries schema-typed arguments.
    ParsedOutput parsedOutput = generateParsedOutput(
        " to=self<|message|>I should look up Paris weather.<|eom|>" +
        onyxToolTurn("get_weather", {{"location", "Paris"}, {"unit", "celsius"}}, "<|eot|>"));

    EXPECT_EQ(parsedOutput.reasoning, "I should look up Paris weather.");
    EXPECT_EQ(parsedOutput.content, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"location":"Paris","unit":"celsius"})");
}

TEST_F(OnyxOutputParserTest, ReasoningAndToolCallAndContentIsolated) {
    // Reasoning + tool call + final answer in one generation: verify no cross-contamination.
    ParsedOutput parsedOutput = generateParsedOutput(
        " to=self<|message|>I need the weather first.<|eom|>" +
        onyxToolTurn("get_weather", {{"location", "Paris"}}) +
        "<|start|>assistant to=user<|message|>I will provide result when I have tool call result.<|eot|>");

    EXPECT_EQ(parsedOutput.reasoning, "I need the weather first.");
    EXPECT_EQ(parsedOutput.content, "I will provide result when I have tool call result.");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"location":"Paris"})");
    // Ensure no leakage of reasoning or ATEM XML into content.
    EXPECT_EQ(parsedOutput.content.find("self"), std::string::npos);
    EXPECT_EQ(parsedOutput.content.find("atem"), std::string::npos);
    EXPECT_EQ(parsedOutput.content.find("weather"), std::string::npos);
    // Ensure no leakage of content or ATEM XML into reasoning.
    EXPECT_EQ(parsedOutput.reasoning.find("result"), std::string::npos);
    EXPECT_EQ(parsedOutput.reasoning.find("atem"), std::string::npos);
}

TEST_F(OnyxOutputParserTest, ToolCallNotParsedWhenToolsUnavailable) {
    // OutputParser::parse() only invokes the tool parser when toolsAvailable is true, so the
    // ATEM block is never extracted and remains (verbatim) somewhere in content.
    std::string turn = onyxToolTurn("get_weather", {{"location", "Paris"}});
    ParsedOutput parsedOutput = generateParsedOutput(turn, /*toolsAvailable=*/false);

    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
    // Known current limitation mirrored from the previous drop: with the tool parser
    // disabled, the raw ATEM block is surfaced as-is in content. Use a substring check so
    // the test does not over-constrain how much of the " to=..."/"<|message|>" envelope the
    // reasoning parser strips around it.
    EXPECT_NE(parsedOutput.content.find(atemBlock("get_weather", {{"location", "Paris"}})), std::string::npos)
        << parsedOutput.content;
}

TEST_F(OnyxOutputParserTest, MalformedOutputWithoutMessageTagLeftUntouched) {
    ParsedOutput parsedOutput = generateParsedOutput("some unexpected raw text without any framing");

    EXPECT_EQ(parsedOutput.content, "some unexpected raw text without any framing");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

// =============================================================================
// Streaming. Implemented on top of OnyxToolParserImpl, a state machine that
// accumulates raw text and hands back a fully assembled tool call once its ATEM end
// tag ("</atem:function_calls>") is seen -- mirroring Qwen3CoderToolParserImpl. Unary
// parse() above drives that same impl with the whole content as one chunk, so it is the
// single-shot degenerate case of streaming, not a parallel implementation of the tag walk.
//
// Like qwen3coder, the function name streams as its own first delta once
// "<atem:invoke name=...>" closes, and the fully-typed arguments blob is sent as a single
// delta once the tool call closes (there is no per-parameter incremental streaming).
//
// Chunk boundaries below are deliberately awkward (splitting tags and values mid-token) to
// exercise the Content/InsideToolCall/InsideFunctionName/InsideFunction/InsideParameter*/
// AfterFunction state machine, mirroring Qwen3CoderOutputParserTest.StreamingSimpleToolCall.
//
// NOTE on the harmony envelope: the leading " to=<name><|message|>" that precedes the ATEM
// block in real output is treated as generic content by the shared OutputParser streaming
// framework (the tool parser's start tag is "<atem:function_calls>"). Streaming-mode
// stripping of that envelope is a separate, not-yet-designed concern (see the TODO in
// OnyxReasoningParser::parseChunk), so it is intentionally out of scope here -- this test
// focuses on the ATEM tool-call state machine itself, exactly as qwen3coder's does.
// =============================================================================
TEST_F(OnyxOutputParserTest, StreamingSimpleToolCall) {
    // Raw PLC structured text code (mirrors qwen3coder's FC_CreateJsonPayload). Written as a
    // raw string literal so it's human-readable; wrapRawCodeAsToolArgs() handles JSON escaping.
    const std::string plcCode = R"(FUNCTION FC_CreateJsonPayload : STRING
VAR_INPUT
    Value1 : REAL;
    Value2 : INT;
    Value3 : BOOL;
    Value4 : STRING(100);
END_VAR
VAR_OUTPUT
    JsonPayload : STRING(1000);
END_VAR
VAR
    TempStr : STRING(100);
END_VAR

    JsonPayload := '{';
    JsonPayload := JsonPayload + '"value1":' + REAL_TO_STRING(Value1, '', 2) + ',';
    JsonPayload := JsonPayload + '"value2":' + INT_TO_STRING(Value2) + ',';
    JsonPayload := JsonPayload + '"value3":' + BOOL_TO_STRING(Value3) + ',';
    JsonPayload := JsonPayload + '"value4":"' + Value4 + '"';
    JsonPayload := JsonPayload + '}';

END_FUNCTION)";

    // Raw Python code with triple-quotes, f-strings, escape sequences (mirrors qwen3coder).
    const std::string pythonCode = R"(
if __name__ == "__main__":
    addresses = {}
    addresses["Hodor"] = """The door"""
    addresses["Arya"] = "Winterfell"
    for name, address in addresses.items():
        print(f'\n\t{name} lives at {address}\n\r'))";

    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Content before any tool call -- OutputParser sees no start tag match, emits content.
        {"<|start|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"assistant ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"to", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=user", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        {"to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"get_weather", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<atem:func", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tion_calls>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // "<atem:invoke name=\"get_weather\">" split mid-tag and mid-name -- name delta emitted
        // once the full opening tag (up to the closing "\">") lands.
        {"<atem:invoke na", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"me=\"get_", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"weath", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"er\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        // Parameter "location" -> "Paris" (string per schema, stays quoted). Split awkwardly.
        {"<atem:parameter na", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"me=\"location\">Pa", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ris</atem:param", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"eter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Parameter "unit" -> "celsius".
        {"<atem:parameter name=\"unit\">celsius</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // "</atem:invoke>" then "</atem:function_calls>" split mid-tag -- closes the tool call,
        // full typed args delta emitted once the end tag completes.
        {"</atem:inv", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"oke>\n</atem:func", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tion_calls>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Paris\",\"unit\":\"celsius\"}"}}]}})"},
        // Harmony envelope + content between tool calls -- swallowed while waiting for the next
        // "<atem:function_calls>" (mirrors qwen3coder's "POTENTIALLY EXISINT CONTENT").
        {"<|eom|><|start|>assistant to=string_tool<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Second tool call -- start tag + invoke name split across tiny chunks.
        {"<atem:function_calls>\n<atem:invoke na", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"me=\"string_tool\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"string_tool"}}]}})"},
        // arg1 (string) split across chunks.
        {"<atem:parameter name=\"arg1\">STRI", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"NG_VALUE</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</atem:invoke>\n</atem:func", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tion_calls>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"arg1\":\"STRING_VALUE\"}"}}]}})"},
        // More envelope/content between tool calls.
        {"<|eom|><|start|>assistant to=string_int_tool<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Third tool call -- string_int_tool: arg1 string (with a leading escaped newline in
        // the value) + arg2 integer (stays numeric per schema).
        {"<atem:function_calls>\n<atem:invoke name=\"string_int_tool\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"string_int_tool"}}]}})"},
        {"<atem:parameter name=\"arg1\">\\nANOTHER_STRING_VALUE</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<atem:parameter name=\"arg2\">314", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"1522</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // arg1's value is a literal backslash-n (C++ "\\n"), a STRING param. It round-trips
        // through two JSON layers: value -> arguments string ("\\n") -> delta string ("\\\\n").
        {"</atem:invoke>\n</atem:function_calls>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\"arg1\":\"\\\\nANOTHER_STRING_VALUE\",\"arg2\":3141522}"}}]}})"},
        // Envelope/content before a "bfcl draft" style call -- cd tool, arriving with preceding
        // text like qwen3coder's "part of bfcl 'draft'." pattern.
        {"<|eom|><|start|>assistant to=cd<|message|>part of bfcl draft.\n\n<atem:function_calls>\n<atem:invoke name=\"cd\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":3,"function":{"name":"cd"}}]}})"},
        {"<atem:parameter name=\"folder\">ResearchDocs</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</atem:invoke>\n</atem:function_calls>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":3,"function":{"arguments":"{\"folder\":\"ResearchDocs\"}"}}]}})"},
        // PLC structured text code as a string tool argument (mirrors qwen3coder's
        // FC_CreateJsonPayload test). Raw code defined above; wrapRawCodeAsToolArgs() /
        // expectedArgsDelta() handle JSON escaping. Sent as a single chunk -- the interesting
        // complexity here is the value escaping, not chunk-boundary splitting.
        {"<|eom|><|start|>assistant to=string_tool<|message|><atem:function_calls>\n<atem:invoke name=\"string_tool\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":4,"function":{"name":"string_tool"}}]}})"},
        {"<atem:parameter name=\"arg1\">" + plcCode + "</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</atem:invoke>\n</atem:function_calls>", ov::genai::GenerationFinishReason::NONE, expectedArgsDelta(4, plcCode)},
        // Python code with triple-quotes, f-strings, escape sequences (mirrors qwen3coder's
        // last case). Also single-chunk value; finishes generation with STOP.
        {"<|eom|><|start|>assistant to=string_tool<|message|><atem:function_calls>\n<atem:invoke name=\"string_tool\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":5,"function":{"name":"string_tool"}}]}})"},
        {"<atem:parameter name=\"arg1\">" + pythonCode + "</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"</atem:invoke>\n</atem:function_calls>", ov::genai::GenerationFinishReason::STOP, expectedArgsDelta(5, pythonCode)},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        i++;
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, {}, /*toolsAvailable=*/true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;  // Both are nullopt, OK
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            const std::string idKey = "\"id\":\"";
            auto docIdPos = docStr.find(idKey);
            auto expectedIdPos = expected.find(idKey);
            if (docIdPos != std::string::npos && expectedIdPos != std::string::npos) {
                auto docIdStart = docIdPos + idKey.size();
                auto docIdEnd = docStr.find("\"", docIdStart);
                auto expectedIdStart = expectedIdPos + idKey.size();
                auto expectedIdEnd = expected.find("\"", expectedIdStart);
                ASSERT_NE(docIdEnd, std::string::npos);
                ASSERT_NE(expectedIdEnd, std::string::npos);
                std::string docId = docStr.substr(docIdStart, docIdEnd - docIdStart);
                std::string expectedId = expected.substr(expectedIdStart, expectedIdEnd - expectedIdStart);
                EXPECT_EQ(docId.size(), expectedId.size()) << "ID length mismatch for chunk[" << i << "]: " << chunk;
                EXPECT_TRUE(std::all_of(docId.begin(), docId.end(), ::isalnum)) << "ID not alphanumeric for chunk[" << i << "]: " << chunk;
                std::string docStrNoId = docStr;
                std::string expectedNoId = expected;
                docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk[" << i << "] (ignoring id value): " << chunk;
            } else {
                SPDLOG_ERROR("Expected:\n{}", expected);
                SPDLOG_ERROR("Got:\n{}", docStr);
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk[" << i << "]: " << chunk;
                // Validate that arguments fields are valid JSON
                if (expected.find("arguments") != std::string::npos) {
                    auto docJsonIt = doc->FindMember("delta");
                    ASSERT_NE(docJsonIt, doc->MemberEnd());
                    auto toolCallsIt = docJsonIt->value.FindMember("tool_calls");
                    ASSERT_NE(toolCallsIt, docJsonIt->value.MemberEnd());
                    for (const auto& toolCall : toolCallsIt->value.GetArray()) {
                        auto functionIt = toolCall.FindMember("function");
                        ASSERT_NE(functionIt, toolCall.MemberEnd());
                        auto argumentsIt = functionIt->value.FindMember("arguments");
                        ASSERT_NE(argumentsIt, functionIt->value.MemberEnd());
                        const std::string& argumentsStr = argumentsIt->value.GetString();
                        rapidjson::Document argsDoc;
                        argsDoc.Parse(argumentsStr.c_str());
                        EXPECT_FALSE(argsDoc.HasParseError()) << "Arguments is not valid JSON for chunk[" << i << "]: " << chunk << "\nArguments string:\n"
                                                              << argumentsStr;
                    }
                }
            }
        } else {
            EXPECT_TRUE(false) << "Mismatch between expectedDelta and doc for id: " << i << " chunk:\n"
                               << chunk
                               << "\nexpectedDelta:\n"
                               << (expectedDelta.has_value() ? expectedDelta.value() : "EMPTY_DELTA")
                               << "\nGot doc:\n"
                               << (doc.has_value() ? [&]() {
                                      rapidjson::StringBuffer buffer;
                                      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                                      doc->Accept(writer);
                                      return std::string(buffer.GetString());
                                  }()
                                                   : "NO_DOC");
            FAIL() << "Mismatch between expectedDelta and doc for chunk[" << i << "]: " << chunk;
        }
    }
}

// =============================================================================
// Streaming reasoning followed by a tool call. Verifies that:
// - reasoning_content deltas are emitted for reasoning body chunks
// - framing tags (to=self, <|message|>, <|eom|>) are swallowed (nullopt)
// - after reasoning ends, the tool call streams normally with name + args deltas
// - no reasoning leaks into content or tool_calls deltas
// =============================================================================
TEST_F(OnyxOutputParserTest, StreamingReasoningThenToolCall) {
    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning start tag -- framework detects "to=self", enters REASONING phase.
        // OnyxReasoningParser::parseChunk sees the tag and returns nullopt (swallows framing).
        {"to=self", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // <|message|> separator -- also swallowed by the reasoning parser.
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Actual reasoning body chunks -- emitted as reasoning_content deltas.
        {"Let me think", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":"Let me think"}})"},
        {" about the weather.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" about the weather."}})"},
        // Reasoning end tag -- swallowed, framework transitions back to UNKNOWN.
        {"<|eom|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool call envelope (harmony prefix) -- must be swallowed, not leaked as content.
        // BUG: framework transitions from REASONING→UNKNOWN after <|eom|>, and in UNKNOWN the
        // envelope doesn't match any start tag ("to=self" for reasoning, "<atem:function_calls>"
        // for tools), so it's flushed as content. Gptoss avoids this because its tool start tag
        // IS the envelope prefix ("<|channel|>commentary to="). Onyx needs equivalent handling.
        {" to=get_weather<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // ATEM start tag -- enters TOOL_CALLS phase.
        {"<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        // Parameter.
        {"<atem:parameter name=\"location\">Paris</atem:parameter>\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Close tool call.
        {"</atem:invoke>\n</atem:function_calls>", ov::genai::GenerationFinishReason::STOP, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Paris\"}"}}]}})"},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        i++;
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, {}, /*toolsAvailable=*/true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            // Normalize tool call IDs (same approach as StreamingSimpleToolCall).
            const std::string idKey = "\"id\":\"";
            auto docIdPos = docStr.find(idKey);
            auto expectedIdPos = expected.find(idKey);
            if (docIdPos != std::string::npos && expectedIdPos != std::string::npos) {
                auto docIdStart = docIdPos + idKey.size();
                auto docIdEnd = docStr.find("\"", docIdStart);
                auto expectedIdStart = expectedIdPos + idKey.size();
                auto expectedIdEnd = expected.find("\"", expectedIdStart);
                std::string docStrNoId = docStr;
                std::string expectedNoId = expected;
                docStrNoId.replace(docIdStart, docIdEnd - docIdStart, std::string(docIdEnd - docIdStart, '*'));
                expectedNoId.replace(expectedIdStart, expectedIdEnd - expectedIdStart, std::string(expectedIdEnd - expectedIdStart, '*'));
                EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk[" << i << "]: " << chunk;
            } else {
                EXPECT_EQ(docStr, expected) << "Mismatch for chunk[" << i << "]: " << chunk;
            }
        } else {
            EXPECT_TRUE(false) << "Mismatch for chunk[" << i << "]: " << chunk
                               << "\nexpectedDelta: " << (expectedDelta.has_value() ? expectedDelta.value() : "nullopt")
                               << "\nGot doc: " << (doc.has_value() ? [&]() {
                                      rapidjson::StringBuffer b;
                                      rapidjson::Writer<rapidjson::StringBuffer> w(b);
                                      doc->Accept(w);
                                      return std::string(b.GetString());
                                  }() : "nullopt");
        }
    }
}

// =============================================================================
// Streaming reasoning followed by content (final answer). Verifies that:
// - reasoning_content deltas are emitted for reasoning body chunks
// - framing tags (to=self, <|message|>, <|eom|>) are swallowed (nullopt)
// - after reasoning ends, the content streams normally as content deltas
// - no reasoning leaks into content deltas
// =============================================================================
TEST_F(OnyxOutputParserTest, StreamingReasoningThenContent) {
    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning start tag -- swallowed.
        {"to=self", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // <|message|> separator -- swallowed.
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Reasoning body chunks -- emitted as reasoning_content deltas.
        {"Let me think", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":"Let me think"}})"},
        {" carefully.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"reasoning_content":" carefully."}})"},
        // Reasoning end tag -- swallowed.
        {"<|eom|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Content envelope -- swallowed (harmony framing before the actual answer).
        {"<|start|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"assistant ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"to", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=user", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Content body chunks -- emitted as content deltas.
        {"The weather in", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"The weather in"}})"},
        {" Paris is sunny.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" Paris is sunny."}})"},
        {"<|eot|>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        i++;
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, {}, /*toolsAvailable=*/true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            EXPECT_EQ(docStr, expected) << "Mismatch for chunk[" << i << "]: " << chunk;
        } else {
            EXPECT_TRUE(false) << "Mismatch for chunk[" << i << "]: " << chunk
                               << "\nexpectedDelta: " << (expectedDelta.has_value() ? expectedDelta.value() : "nullopt")
                               << "\nGot doc: " << (doc.has_value() ? [&]() {
                                      rapidjson::StringBuffer b;
                                      rapidjson::Writer<rapidjson::StringBuffer> w(b);
                                      doc->Accept(w);
                                      return std::string(b.GetString());
                                  }() : "nullopt");
        }
    }
}

TEST_F(OnyxOutputParserTest, StreamingContentOnly) {
    int i = -1;
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        {"to=user", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Your ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"Your "}})"},
        {"tweet", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"tweet"}})"},
        {" has", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" has"}})"},
        {" been", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" been"}})"},
        {" posted.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" posted."}})"},
        {" Let me know", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" Let me know"}})"},
        {" if you need", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" if you need"}})"},
        {" I can do", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" I can do"}})"},
        {" anything ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":" anything "}})"},
        {"to ", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"to "}})"},
        {"help.", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"help."}})"},
        {"<|eot|>", ov::genai::GenerationFinishReason::STOP, std::nullopt},
    };

    for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVec) {
        i++;
        std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, {}, /*toolsAvailable=*/true, finishReason);
        if (!expectedDelta.has_value() && !doc.has_value()) {
            continue;
        }
        if (expectedDelta.has_value() && doc.has_value()) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            doc->Accept(writer);
            std::string docStr = buffer.GetString();
            std::string expected = expectedDelta.value();
            EXPECT_EQ(docStr, expected) << "Mismatch for chunk[" << i << "]: " << chunk;
        } else {
            EXPECT_TRUE(false) << "Mismatch for chunk[" << i << "]: " << chunk
                               << "\nexpectedDelta: " << (expectedDelta.has_value() ? expectedDelta.value() : "nullopt")
                               << "\nGot doc: " << (doc.has_value() ? [&]() {
                                      rapidjson::StringBuffer b;
                                      rapidjson::Writer<rapidjson::StringBuffer> w(b);
                                      doc->Accept(w);
                                      return std::string(b.GetString());
                                  }() : "nullopt");
        }
    }
}

// =============================================================================
// Regression test for an agentic-streaming bug: OutputParser/OnyxToolParser is often
// constructed BEFORE the request's tools are known (toolSchemas is a reference to a map
// that starts empty and is filled in by the caller afterwards -- see the constructor
// comment on OnyxToolParser::toolSchemas). getParsingStartTags() must lazily rebuild its
// "to=<name>" tags from whatever toolSchemas holds at call time (not just what it held at
// construction time), otherwise the harmony envelope preceding a real tool call is never
// recognized as a start tag and leaks into content as raw " to=<name><|message|>" text.
// =============================================================================
TEST_F(OnyxOutputParserTest, StreamingToolEnvelopeNotLeakedWhenSchemasFilledAfterConstruction) {
    ToolsSchemas_t lateSchemas;  // empty when OutputParser/OnyxToolParser are constructed
    OutputParser parser(*opt125mTokenizer, "onyx", "onyx", lateSchemas);

    // Populate the SAME map object only now -- OnyxToolParser keeps a reference to it, so
    // this mirrors production code filling request.toolNameSchemaMap after construction.
    lateSchemas = toolsSchemas;

    // Harmony envelope for a tool call, split the way a real generation streams it. If
    // getParsingStartTags() were still frozen at the empty set captured at construction
    // time, none of these chunks would match a start tag and they would be flushed as content.
    auto doc = parser.parseChunk(" to=get_weather", {}, /*toolsAvailable=*/true, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(doc.has_value()) << "envelope prefix must not be flushed as content";

    doc = parser.parseChunk("<|message|>", {}, /*toolsAvailable=*/true, ov::genai::GenerationFinishReason::NONE);
    EXPECT_FALSE(doc.has_value()) << "\"<|message|>\" separator must not be flushed as content";

    doc = parser.parseChunk("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n", {}, /*toolsAvailable=*/true, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(doc.has_value());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc->Accept(writer);
    std::string docStr = buffer.GetString();
    EXPECT_NE(docStr.find(R"("tool_calls")"), std::string::npos) << docStr;
    EXPECT_NE(docStr.find(R"("name":"get_weather")"), std::string::npos) << docStr;
    EXPECT_EQ(docStr.find(R"("content")"), std::string::npos) << "envelope leaked into content: " << docStr;
}

// =============================================================================
// Proves the "unary is an edge case of streaming" property holds structurally, not
// just by coincidence: OnyxToolParser::parse() drives the same OnyxToolParserImpl used
// by parseChunk() (see onyx_tool_parser.cpp), so this re-checks that the unary entry
// point wires into the same state machine covered above.
// =============================================================================
TEST_F(OnyxOutputParserTest, UnaryToolCallMatchesStreamingReuse) {
    ParsedOutput parsedOutput = generateParsedOutput(
        onyxToolTurn("get_weather", {{"location", "Paris"}, {"unit", "celsius"}}));

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"location":"Paris","unit":"celsius"})");
}

TEST_F(OnyxOutputParserTest, UnaryTwoSequentialToolCalls) {
    // Two tool-call turns back to back, as the model emits them with ignore_eos (each turn is
    // re-introduced by "<|start|>assistant to=<name><|message|>"). Both take a single string arg.
    ParsedOutput parsedOutput = generateParsedOutput(
        onyxToolTurn("get_weather", {{"location", "SF"}}) +
        "<|start|>assistant" + onyxToolTurn("get_time", {{"city", "SF"}}));

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, R"({"location":"SF"})");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "get_time");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, R"({"city":"SF"})");
}

// =============================================================================
// Direct OnyxToolParserImpl unit tests -- exercise the state machine directly, below
// OutputParser/OnyxToolParser (mirrors Qwen3CoderOutputParserTest's TestJustParserImpl*
// layer). These use plain-string parameter values whose correct JSON serialization does
// not depend on schema typing, so they can drive the impl through its current public API
// (parseChunk / getCurrentFunctionName / removeToolCallsFromContentIfNeeded) directly.
// NOTE: when the impl is made schema-driven like qwen3coder (its ctor then taking a
// ToolsParameterTypeMap_t), these direct constructions will need that argument.
// =============================================================================
TEST_F(OnyxOutputParserTest, TestJustParserImplUnaryToolCall) {
    const std::string input = onyxToolTurn("get_weather", {{"location", "Paris"}});
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"location":"Paris"})");
}

TEST_F(OnyxOutputParserTest, TestJustParserImplUnaryWithNoToolCall) {
    const std::string input = "Unexpected void found. Philosophical crisis imminent.";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_FALSE(callsOpt.has_value());
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    EXPECT_EQ(content, input);
}

TEST_F(OnyxOutputParserTest, TestJustParserImplUnaryWithTwoToolCalls) {
    const std::string input =
        onyxToolTurn("get_weather", {{"location", "SF"}}) +
        "<|start|>assistant" + onyxToolTurn("get_time", {{"city", "SF"}});
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    ASSERT_EQ(calls.size(), 2) << input;
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, R"({"location":"SF"})");
    EXPECT_EQ(calls[1].name, "get_time");
    EXPECT_EQ(calls[1].arguments, R"({"city":"SF"})");
}

TEST_F(OnyxOutputParserTest, TestJustParserImplUnaryToolCallThenTruncatedOpen) {
    // A completed tool call followed by a second call truncated before "</atem:function_calls>"
    // (generation cut off mid-arguments). The completed call is still extracted, and content
    // removal must drop BOTH the completed block and the dangling open block -- not bail out and
    // leave every block in content (the pre-fix behavior on a begin/end tag-count mismatch).
    const std::string input =
        onyxToolTurn("get_weather", {{"location", "SF"}}) +
        " to=get_time<|message|><atem:function_calls>\n<atem:invoke name=\"get_time\">\n<atem:parameter name=\"city\">S";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ASSERT_EQ(callsOpt.value().size(), 1) << input;
    EXPECT_EQ(callsOpt.value()[0].name, "get_weather");
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    // Both ATEM blocks are gone (the impl strips only the blocks, not the surrounding harmony
    // envelope -- that is the reasoning parser's job at the OutputParser level).
    EXPECT_EQ(content.find("<atem:"), std::string::npos) << content;
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithNoStateChange) {
    const std::string input = "Some content without tool calls";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithPartialToolCall) {
    // Function name is known once "<atem:invoke name=\"get_weather\">" is seen, even though the
    // tool call has not closed yet (no "</atem:function_calls>").
    const std::string input = " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"location\">";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
    ASSERT_TRUE(parser.getCurrentFunctionName().has_value());
    EXPECT_EQ(parser.getCurrentFunctionName().value(), "get_weather");
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithToolCallNoArgs) {
    const std::string input = onyxToolTurn("get_current_location", {});
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto stepResult = parser.parseChunk(content);
    ASSERT_TRUE(stepResult.has_value());
    auto& calls = stepResult.value();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "get_current_location");
    EXPECT_EQ(calls[0].arguments, "{}");
}

// =============================================================================
// Parametrized readability test for schema-driven argument typing -- direct analogue of
// Qwen3CoderOutputParserParametrizedTest.TestJustParserImplWithVariousArgumentTypes, adapted
// to the ATEM block. A single <atem:parameter> value is typed according to the tool schema:
// string->quoted, integer/number->numeric, boolean->true/false (also normalizing Python-style
// True/False), object/array->parsed JSON. Feeds a bare ATEM block (no harmony envelope) so
// removeToolCallsFromContentIfNeeded() leaves content empty, mirroring qwen3coder feeding a
// bare <tool_call> block.
// =============================================================================
class OnyxOutputParserParametrizedTest : public OnyxOutputParserTest, public ::testing::WithParamInterface<std::tuple<std::string, std::string, std::string, std::string>> {
};

TEST_P(OnyxOutputParserParametrizedTest, TestJustParserImplWithVariousArgumentTypes) {
    const std::string& toolName = std::get<0>(GetParam());
    const std::string& argName = std::get<1>(GetParam());
    const std::string& paramValue = std::get<2>(GetParam());
    const std::string& expectedArguments = std::get<3>(GetParam());

    const std::string input = atemBlock(toolName, {{argName, paramValue}});
    auto content = input;
    ovms::OnyxToolParserImpl parser(onyxToolsParametersTypeMap);
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value()) << input;
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, toolName);
    EXPECT_EQ(calls[0].arguments, expectedArguments) << input;
    EXPECT_EQ(parser.getCurrentState(), ovms::OnyxToolParserImpl::State::Content) << input;
    EXPECT_EQ(content, "") << input;
}

const std::vector<std::tuple<std::string, std::string, std::string, std::string>> onyxParamValueAndExpectedArgumentsVec = {
    {"string_tool", "arg1", "value1", R"({"arg1":"value1"})"},
    {"int_tool", "arg1", "42", R"({"arg1":42})"},
    {"float_tool", "arg1", "52.32", R"({"arg1":52.32})"},
    {"bool_tool", "arg1", "true", R"({"arg1":true})"},
    {"bool_tool", "arg1", "false", R"({"arg1":false})"},
    {"bool_tool", "arg1", "True", R"({"arg1":true})"},
    {"bool_tool", "arg1", "False", R"({"arg1":false})"},
    {"object_tool", "arg1", R"({"a":1,"b":{"c":"asd"}})", R"({"arg1":{"a":1,"b":{"c":"asd"}}})"},
    {"list_tool", "arg1", "[1, 2, 3]", R"({"arg1":[1,2,3]})"},
    {"list_tool", "arg1", R"(["a","b","c"])", R"({"arg1":["a","b","c"]})"},
    {"object_tool", "arg1", R"([{"a":1},{"b":2}])", R"({"arg1":[{"a":1},{"b":2}]})"}};

INSTANTIATE_TEST_SUITE_P(
    OnyxOutputParserParametrizedTestInstance,
    OnyxOutputParserParametrizedTest,
    ::testing::ValuesIn(onyxParamValueAndExpectedArgumentsVec),
    [](const ::testing::TestParamInfo<OnyxOutputParserParametrizedTest::ParamType>& info) {
        std::string name = std::get<0>(info.param) + "_" + std::get<2>(info.param);
        std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
        return name;
    });
