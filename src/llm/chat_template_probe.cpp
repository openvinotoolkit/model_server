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
#include "chat_template_probe.hpp"

#include <chrono>
#include <future>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#if (PYTHON_DISABLE == 0)
#include "py_jinja_template_processor.hpp"
#endif

namespace ovms {

bool probeChatTemplateBasicRender(ov::genai::Tokenizer& tokenizer) {
    if (tokenizer.get_chat_template().empty()) {
        return true;
    }

    static const std::string contentNeedle = "probe_basic_7Qw2";

    try {
        ov::genai::ChatHistory history;
        history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Hi"})"));
        history.push_back(ov::genai::JsonContainer::from_json_string(
            R"({"role":"assistant","content":")" + contentNeedle + R"("})"));

        auto t0 = std::chrono::steady_clock::now();
        std::string output = tokenizer.apply_chat_template(history, true);
        auto t1 = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Basic render probe: {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        // Check if assistant content appears in output
        if (output.find(contentNeedle) == std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: assistant content not found in output. "
                                                      "Template may be incompatible with minja.");
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Basic render probe output: {}", output);
            return false;
        }

        // Check for raw JSON dump markers — if minja silently failed, it dumps
        // the message object as JSON containing "content": and "role":
        if (output.find("\"content\": \"" + contentNeedle + "\"") != std::string::npos ||
            output.find("\"content\":\"" + contentNeedle + "\"") != std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: output contains raw JSON dump of message object. "
                                                      "Template is not supported by minja.");
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Basic render probe output: {}", output);
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: exception: {}", e.what());
        return false;
    } catch (...) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: unknown exception");
        return false;
    }
}

static const std::string PROBE_NEEDLE = "probe_needle_xK9m";

// Analyze dry-run probe outputs and update caps accordingly.
// Returns false if the template silently failed (tool calls not supported).
static bool analyzeProbeResults(bool strOk, const std::string& strOut,
    bool objOk, const std::string& objOut,
    ChatTemplateCaps& caps) {
    auto rendersNativeArgs = [](const std::string& output) -> bool {
        return output.find("\"" + PROBE_NEEDLE + "\": \"") != std::string::npos ||
               output.find("\"" + PROBE_NEEDLE + "\":\"") != std::string::npos ||
               output.find("<parameter=" + PROBE_NEEDLE + ">") != std::string::npos ||
               output.find(PROBE_NEEDLE + ":<|") != std::string::npos ||
               output.find(PROBE_NEEDLE + "=") != std::string::npos;
    };

    bool strArgsRendersNative = strOk && rendersNativeArgs(strOut);
    bool objArgsRendersNative = objOk && rendersNativeArgs(objOut);

    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe: strRendersNative={}, objRendersNative={}",
        strArgsRendersNative, objArgsRendersNative);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe strArgs output: {}", strOut);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe objArgs output: {}", objOut);

    // Detect silent failure: output contains raw JSON dump of tool_calls
    static const std::string silentFailureMarker = "\"tool_calls\": [";
    bool strArgsFailed = strOk && strOut.find(silentFailureMarker) != std::string::npos;
    bool objArgsFailed = objOk && objOut.find(silentFailureMarker) != std::string::npos;

    if (strArgsFailed || objArgsFailed) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Dry-run probe: template silently failed to render tool calls");
        caps.supportsToolCalls = false;
        caps.supportsTools = false;
        return false;
    }

    if (strArgsRendersNative || objArgsRendersNative) {
        bool probeResult = objArgsRendersNative;
        if (probeResult != caps.requiresObjectArguments) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe overrides requiresObjectArguments: {} -> {}",
                caps.requiresObjectArguments, probeResult);
        }
        caps.requiresObjectArguments = probeResult;
    } else {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe: template does not render tool_call arguments natively");
    }
    return true;
}

bool probeChatTemplateCapsMinja(ov::genai::Tokenizer& tokenizer, ChatTemplateCaps& caps) {
    if (tokenizer.get_chat_template().empty()) {
        return true;
    }
    if (!caps.supportsToolCalls) {
        return true;
    }

    // Async because apply_chat_template is slow: CVS-189192
    auto strArgsFuture = std::async(std::launch::async, [&tokenizer]() -> std::pair<bool, std::string> {
        try {
            ov::genai::ChatHistory history;
            history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Hello"})"));
            history.push_back(ov::genai::JsonContainer::from_json_string(
                R"({"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":"{\")" + PROBE_NEEDLE + R"(\":\"val\"}"}}]})"));
            std::string output = tokenizer.apply_chat_template(history, true);
            return {true, std::move(output)};
        } catch (...) {
            return {false, ""};
        }
    });

    auto objArgsFuture = std::async(std::launch::async, [&tokenizer]() -> std::pair<bool, std::string> {
        try {
            ov::genai::ChatHistory history;
            history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Hello"})"));
            history.push_back(ov::genai::JsonContainer::from_json_string(
                R"({"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":{")" + PROBE_NEEDLE + R"(":"val"}}}]})"));
            std::string output = tokenizer.apply_chat_template(history, true);
            return {true, std::move(output)};
        } catch (...) {
            return {false, ""};
        }
    });

    auto [strOk, strOut] = strArgsFuture.get();
    auto [objOk, objOut] = objArgsFuture.get();

    return analyzeProbeResults(strOk, strOut, objOk, objOut, caps);
}

#if (PYTHON_DISABLE == 0)

bool probeChatTemplateCapsJinja(PyJinjaTemplateProcessor& templateProcessor, const std::string& modelsPath, ChatTemplateCaps& caps) {
    if (!caps.supportsToolCalls) {
        return true;
    }

    std::string strArgsJson = R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":"{\")" + PROBE_NEEDLE + R"(\":\"val\"}"}}]}]})";
    std::string objArgsJson = R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":{")" + PROBE_NEEDLE + R"(":"val"}}}]}]})";

    std::string strOut, objOut;
    bool strOk = false, objOk = false;

    try {
        strOk = PyJinjaTemplateProcessor::applyChatTemplate(templateProcessor, strArgsJson, strOut);
    } catch (...) {
    }

    try {
        objOk = PyJinjaTemplateProcessor::applyChatTemplate(templateProcessor, objArgsJson, objOut);
    } catch (...) {
    }

    return analyzeProbeResults(strOk, strOut, objOk, objOut, caps);
}
#endif

}  // namespace ovms
