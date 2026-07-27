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
#include <vector>

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/llm/io_processing/onyx/onyx_tool_parser.hpp"
#include "src/llm/io_processing/output_parser.hpp"
#include "src/logging.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

// Onyx does not ship a converted HF tokenizer in this early preview, and none of the
// segments the parser looks for ("to=functions.", "<|message|>", "<|eom|>", "<|eot|>",
// "to=self") are real special tokens of the model this parser is designed for -- they are
// plain text sequences that must round-trip losslessly through encode()+decode() on ANY
// tokenizer. facebook/opt-125m is already used the same way for chat-template testing
// (see ChatTemplateEndToEndMinjaTest), so it is reused here to avoid pulling in a new
// model fixture.
// TODO @atobiszei replace when tokenizer is available
#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\facebook\\opt-125m";
#else
const std::string tokenizerPath = "/ovms/src/test/llm_testing/facebook/opt-125m";
#endif

static std::unique_ptr<ov::genai::Tokenizer> opt125mTokenizer;

// Onyx never consults tool schemas (arguments are forwarded as a raw JSON blob verbatim,
// no per-parameter type coercion like Qwen3CoderToolParser) -- these mirror
// Qwen3CoderOutputParserTest's toolSchemasInput/toolsSchemas setup purely so the test
// fixture shape matches, and so a real schema is on hand if that ever changes.
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

    // Shared by streaming tests: compares a parseChunk() delta against the expected JSON,
    // masking the randomly generated tool call id (kept as one helper rather than the same
    // id-masking block duplicated per test, mirroring Qwen3CoderOutputParserTest usage).
    static void assertDeltaMatches(const std::optional<rapidjson::Document>& doc, const std::optional<std::string>& expectedDelta, const std::string& chunk);

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

// A single generate() call stops at the first "<|eom|>"/"<|eot|>" (both are configured as
// eos tokens for Onyx), so only one of these three segment shapes is ever produced at a time.

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

TEST_F(OnyxOutputParserTest, PrivateReasoningOnly) {
    ParsedOutput parsedOutput = generateParsedOutput(" to=self<|message|>Let me think about this.<|eom|>");

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "Let me think about this.");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

TEST_F(OnyxOutputParserTest, ToolCallWithRawJsonArguments) {
    ParsedOutput parsedOutput = generateParsedOutput(" to=functions.get_weather<|message|>{\"location\":\"Paris\",\"unit\":\"celsius\"}<|eom|>");

    EXPECT_EQ(parsedOutput.content, "");
    EXPECT_EQ(parsedOutput.reasoning, "");
    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    // Onyx passes arguments through verbatim -- no schema-driven reformatting like qwen3coder.
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\":\"Paris\",\"unit\":\"celsius\"}");
    EXPECT_EQ(parsedOutput.toolCalls[0].id.empty(), false);
}

TEST_F(OnyxOutputParserTest, ToolCallNotParsedWhenToolsUnavailable) {
    // OutputParser::parse() only invokes the tool parser when toolsAvailable is true.
    ParsedOutput parsedOutput = generateParsedOutput(" to=functions.get_weather<|message|>{\"location\":\"Paris\"}<|eom|>", /*toolsAvailable=*/false);

    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
    // Known current limitation: the reasoning parser (which always runs) intentionally
    // leaves "to=functions." segments untouched so the tool parser can claim them -- but
    // if the tool parser never runs, the raw wrapped segment is surfaced as-is in content.
    EXPECT_EQ(parsedOutput.content, " to=functions.get_weather<|message|>{\"location\":\"Paris\"}<|eom|>");
}

TEST_F(OnyxOutputParserTest, MalformedOutputWithoutMessageTagLeftUntouched) {
    ParsedOutput parsedOutput = generateParsedOutput("some unexpected raw text without any framing");

    EXPECT_EQ(parsedOutput.content, "some unexpected raw text without any framing");
    EXPECT_EQ(parsedOutput.reasoning, "");
    EXPECT_EQ(parsedOutput.toolCalls.size(), 0);
}

// =============================================================================
// Streaming is implemented on top of OnyxToolParserImpl, a pure state machine that
// accumulates raw text and hands back a fully assembled tool call once its "<|eom|>"
// end tag is seen (mirroring Qwen3CoderToolParserImpl) -- unary parse() below drives
// that same impl with the whole content as a single chunk, so it is the single-shot
// degenerate case of streaming, not a parallel implementation of the tag walk.
//
// Because Onyx's arguments are already a complete raw JSON blob needing no per-parameter
// schema coercion, they are still sent to the client as a single delta once the tool call
// closes (matching Qwen3CoderToolParser's sendFullDelta) rather than streamed incrementally
// as raw text arrives -- only the function name streams as its own delta once known.
//
// Chunk boundaries below are deliberately awkward (splitting the function name and the
// JSON arguments mid-token) to exercise the Content/InsideName/InsideArguments state
// machine, mirroring Qwen3CoderOutputParserTest.StreamingSimpleToolCall.
// =============================================================================
void OnyxOutputParserTest::assertDeltaMatches(const std::optional<rapidjson::Document>& doc, const std::optional<std::string>& expectedDelta, const std::string& chunk) {
    if (!expectedDelta.has_value()) {
        EXPECT_FALSE(doc.has_value()) << "Expected nullopt for chunk: " << chunk;
        return;
    }
    ASSERT_TRUE(doc.has_value()) << "Expected a delta for chunk: " << chunk;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc->Accept(writer);
    std::string docStr = buffer.GetString();
    std::string expected = expectedDelta.value();
    // If both strings contain "id":"...", compare id values by length and alphanumeric, else compare whole strings
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
        EXPECT_EQ(docId.size(), expectedId.size()) << "ID length mismatch for chunk: " << chunk;
        EXPECT_TRUE(std::all_of(docId.begin(), docId.end(), ::isalnum)) << "ID not alphanumeric for chunk: " << chunk;
        docStr.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
        expected.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
    }
    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: " << chunk;
}

TEST_F(OnyxOutputParserTest, StreamingSimpleToolCall) {
    // Mirrors Qwen3CoderOutputParserTest.StreamingSimpleToolCall's rigor: adversarial
    // chunk boundaries, content before/between tool calls, complex argument values
    // (PLC structured text, Python with triple-quotes/f-strings/escape sequences),
    // adapted to Onyx's "to=functions.<name><|message|><json_args><|eom|>" format.
    //
    // Unlike qwen3coder there is no per-parameter incremental streaming to test
    // (qwen3coder's <parameter=...> tags) since Onyx's arguments are a single raw JSON blob.
    // However, content before/between tool calls IS tested because the OutputParser
    // streaming framework handles that generically (UNKNOWN -> CONTENT transition when no
    // start tag is found).
    //
    // Key structural differences from qwen3coder:
    //  - Start tag is "to=functions." (not "<tool_call>")
    //  - Name delimiter is "<|message|>" (not ">")
    //  - End tag is "<|eom|>" (not "</tool_call>")
    //  - Arguments are a single raw JSON blob (not per-parameter XML tags)
    //  - PLC/Python code must be JSON-escaped within the arguments blob

    // Raw PLC structured text code (mirrors qwen3coder's FC_CreateJsonPayload).
    // Written as a raw string literal so it's human-readable; wrapRawCodeAsToolArgs()
    // handles all the JSON escaping at runtime.
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

    // Raw Python code with triple-quotes, f-strings, escape sequences (mirrors
    // qwen3coder's last test case).
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
        {"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"content":"JUST_SOME_STRING_BEFORE_SPECIAL_STARTING_TAG"}})"},
        // Start tag "to=functions." split across several arbitrarily small chunks.
        // Note: leading space before "to=" is just normal content/separator; the start
        // tag the framework looks for is "to=functions." without the space.
        {" to", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions.", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Function name streams in across several small chunks -- still no delta
        // (OnyxToolParserImpl is in InsideName state, accumulating).
        {"get", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"_", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"weath", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"er", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // "<|message|>" itself split mid-tag -- name delta emitted once the full tag lands.
        {"<|mess", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"age|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":0,"function":{"name":"get_weather"}}]}})"},
        // Raw JSON argument text (with a nested object) split at awkward byte boundaries.
        {"{\"locat", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ion\":\"Pa", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ris\",\"opt", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ions\":{\"unit", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\":\"cel", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"sius\"}}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // "<|eom|>" split mid-tag -- closes the tool call once complete.
        {"<|e", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"om|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Paris\",\"options\":{\"unit\":\"celsius\"}}"}}]}})"},
        // Content between tool calls (mirrors qwen3coder's "POTENTIALLY EXISINT CONTENT").
        // In TOOL_CALLS_WAITING_FOR_TOOL phase, text without start tag match waits for more.
        {"POTENTIALLY EXISINT CONTENT", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Second tool call -- start tag + name + <|message|> split across tiny chunks.
        {" to", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"=functi", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ons.str", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ing_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|messa", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ge|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":1,"function":{"name":"string_tool"}}]}})"},
        // Arguments split across chunks.
        {"{\"arg1\":", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"\"STRI", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"NG_VALUE\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|eo", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"m|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"arg1\":\"STRING_VALUE\"}"}}]}})"},
        // More content between tool calls (mirrors "CONTENT_AFTER_TOOL_CALL").
        {"CONTENT_AFTER_TOOL_CALL", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Third tool call -- string_int_tool with two parameters in JSON (integer stays
        // numeric). Start tag + name + <|message|> split differently from previous calls.
        {" to=func", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"tions.strin", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"g_int_tool<|", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"message|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":2,"function":{"name":"string_int_tool"}}]}})"},
        // Arguments with a leading \n in the string value (matches qwen3coder's
        // "\nANOTHER_STRING_VALUE" pattern) and an integer parameter.
        {"{\"arg1\":\"\\n", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ANOTHER_STRING_VALUE\",\"ar", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"g2\":314", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"1522}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|eom|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\"arg1\":\"\\nANOTHER_STRING_VALUE\",\"arg2\":3141522}"}}]}})"},
        // "NOTHING IMPORTANT HERE" content between calls (mirrors qwen3coder).
        {"NOTHING IMPORTANT HERE", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // A "bfcl draft" style call -- cd tool. Start tag arrives with some preceding
        // text just like qwen3coder's "part of bfcl 'draft'.\n\n<function=cd>\n" pattern.
        {"part of bfcl 'draft'.\n\n to=functions.cd<|message|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":3,"function":{"name":"cd"}}]}})"},
        {"{\"folder\":\"ResearchDocs\"}", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|eom|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":3,"function":{"arguments":"{\"folder\":\"ResearchDocs\"}"}}]}})"},
        // PLC structured text code as a tool argument (mirrors qwen3coder's
        // FC_CreateJsonPayload test). Raw code is defined above as plcCode; the helper
        // wrapRawCodeAsToolArgs() handles all JSON escaping via rapidjson so we don't
        // need to manually count backslashes. Sent as a single chunk since the interesting
        // escaping complexity is in the content, not in chunk-boundary splitting.
        {" to=functions.string_tool", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":4,"function":{"name":"string_tool"}}]}})"},
        {wrapRawCodeAsToolArgs(plcCode), ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|eom|>", ov::genai::GenerationFinishReason::NONE, expectedArgsDelta(4, plcCode)},
        // Python code with triple-quotes, f-strings, escape sequences (mirrors
        // qwen3coder's last test case). Also sent as a single chunk -- the chunk-boundary
        // adversarial testing is covered by the earlier tool calls above.
        {" to=functions.string_tool<|mess", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"age|>", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"id":"XXXXXXXXX","type":"function","index":5,"function":{"name":"string_tool"}}]}})"},
        {wrapRawCodeAsToolArgs(pythonCode), ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|eom|>", ov::genai::GenerationFinishReason::STOP, expectedArgsDelta(5, pythonCode)},
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
// Proves the "unary is an edge case of streaming" property holds structurally, not
// just by coincidence: OnyxToolParser::parse() literally drives the same
// OnyxToolParserImpl used by parseChunk() (see onyx_tool_parser.cpp), so this is
// really just re-checking that the unary entry point wires into the same state
// machine already covered above.
// =============================================================================
TEST_F(OnyxOutputParserTest, UnaryToolCallMatchesStreamingReuse) {
    ParsedOutput parsedOutput = generateParsedOutput(" to=functions.get_weather<|message|>{\"location\":\"Paris\",\"unit\":\"celsius\"}<|eom|>");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 1);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"location\":\"Paris\",\"unit\":\"celsius\"}");
}

TEST_F(OnyxOutputParserTest, UnaryTwoSequentialToolCalls) {
    ParsedOutput parsedOutput = generateParsedOutput(
        " to=functions.get_weather<|message|>{\"city\":\"SF\"}<|eom|> to=functions.get_time<|message|>{\"city\":\"SF\"}<|eom|>");

    ASSERT_EQ(parsedOutput.toolCalls.size(), 2);
    EXPECT_EQ(parsedOutput.toolCalls[0].name, "get_weather");
    EXPECT_EQ(parsedOutput.toolCalls[0].arguments, "{\"city\":\"SF\"}");
    EXPECT_EQ(parsedOutput.toolCalls[1].name, "get_time");
    EXPECT_EQ(parsedOutput.toolCalls[1].arguments, "{\"city\":\"SF\"}");
}

// =============================================================================
// Direct OnyxToolParserImpl unit tests -- mirrors Qwen3CoderOutputParserTest's
// TestJustParserImplUnary*/TestJustParserImplStreamStep* layer (which exercises the
// state machine directly, below OutputParser/OnyxToolParser), previously untested here.
// =============================================================================
TEST_F(OnyxOutputParserTest, TestJustParserImplUnaryToolCall) {
    const std::string input = " to=functions.get_weather<|message|>{\"location\":\"Paris\"}<|eom|>";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 1) << input;
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    EXPECT_EQ(content, "");
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
    const std::string input = " to=functions.get_weather<|message|>{\"city\":\"SF\"}<|eom|> to=functions.get_time<|message|>{\"city\":\"SF\"}<|eom|>";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto callsOpt = parser.parseChunk(content);
    ASSERT_TRUE(callsOpt.has_value());
    ToolCalls_t& calls = callsOpt.value();
    auto status = parser.removeToolCallsFromContentIfNeeded(content);
    EXPECT_TRUE(status.ok()) << status.string();
    ASSERT_EQ(calls.size(), 2) << input;
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[0].arguments, "{\"city\":\"SF\"}");
    EXPECT_EQ(calls[1].name, "get_time");
    EXPECT_EQ(calls[1].arguments, "{\"city\":\"SF\"}");
    EXPECT_EQ(content, "");
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithNoStateChange) {
    const std::string input = "Some content without tool calls";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithPartialToolCall) {
    const std::string input = " to=functions.get_weather<|message|>{\"location\":";
    auto content = input;
    ovms::OnyxToolParserImpl parser;
    auto stepResult = parser.parseChunk(content);
    ASSERT_FALSE(stepResult.has_value());
    ASSERT_TRUE(parser.getCurrentFunctionName().has_value());
    EXPECT_EQ(parser.getCurrentFunctionName().value(), "get_weather");
}

TEST_F(OnyxOutputParserTest, TestJustParserImplStreamStepWithToolCallNoArgs) {
    const std::string input = " to=functions.get_current_location<|message|>{}<|eom|>";
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
// Qwen3CoderOutputParserTest test cases, for reference/parity comparison (this file
// intentionally does not have a 1:1 test for every one of these -- see inline notes
// on why some don't apply to Onyx's simpler, non-schema-driven, single-JSON-blob
// argument format):
//   Parse1ToolCall1Function1ArgumentTagsNewline
//   Parse1ToolCall1Function1ArgumentNoProperBeginTag
//   Parse1ToolCallNestedXmlNotFromSchema
//   ParseTwoToolCalls1Function1ArgumentTagsNoNewline
//   Parse1ToolCall1Function1ArgumentTagsNoNewline
//   Parse1ToolCall1Function1ArgumentMultilineValue
//   TestJustParserImplUnaryToolCall                         -- covered above
//   TestJustParserImplUnaryWithNoToolCall                   -- covered above
//   TestJustParserImplUnaryWithContent                      -- N/A: Onyx's grammar never
//       has plain content before/after a tool-call tag within the same generated turn
//   TestJustParserImplUnaryWithThreeParameters               -- N/A: no per-parameter
//       schema-driven typing; arguments are always a single opaque JSON blob
//   TestJustParserImplUnaryWithEnforcementOfStringParameter  -- N/A, same reason
//   TestJustParserImplUnaryWithNotPresentToolSchema          -- N/A, same reason (Onyx
//       never even looks at tool schemas -- see ToolCallWithRawJsonArguments above)
//   TestJustParserImplUnaryWithJsonObjectArgument            -- covered by nested-object
//       case in StreamingSimpleToolCall above
//   TestJustParserImplUnaryWithTwoToolCalls                  -- covered above
//   TestJustParserImplUnaryToolCallNoMatchingToolParameterTypeMapEntry -- N/A, same reason
//   TestJustParserImplUnaryToolCallWithRepeatedArgument      -- N/A, same reason (no
//       per-parameter parsing to have a "repeated argument" concept at all)
//   TestJustParserImplStreamStepWithMoreThan1StateChange     -- covered by
//       TestJustParserImplUnaryWithTwoToolCalls above (both calls resolve in one parseChunk)
//   TestJustParserImplStreamStepWithNoStateChange            -- covered above
//   TestJustParserImplStreamStepWithPartialToolCall          -- covered above
//   TestJustParserImplStreamStepWithTwoToolCalls             -- covered by
//       TestJustParserImplUnaryWithTwoToolCalls above
//   TestJustParserImplStreamStepWithToolCallNoArgs           -- covered above
//   Qwen3CoderOutputParserParametrizedTest.TestJustParserImplWithVariousArgumentTypes -- N/A:
//       parametrized over per-parameter type coercion (string/int/float/bool/object/list),
//       which does not exist for Onyx (raw JSON passthrough only)
//   StreamingSimpleToolCall                                  -- covered above (adapted;
//       see comment on that test for what was intentionally omitted/adjusted)
// =============================================================================

