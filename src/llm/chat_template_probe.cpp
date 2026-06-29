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
        std::string output = tokenizer.apply_chat_template(history, false);
        auto t1 = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Basic render probe: {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        // Check if assistant content appears in output
        if (output.find(contentNeedle) == std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: assistant content not found in output. "
                                                      "Template may be incompatible with minja.");
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Basic render probe output: {}", output);
            return false;
        }

        // Check for raw JSON dump markers — if minja silently failed, it dumps
        // the message object as JSON containing "content": and "role":
        if (output.find("\"content\": \"" + contentNeedle + "\"") != std::string::npos ||
            output.find("\"content\":\"" + contentNeedle + "\"") != std::string::npos) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger, "Basic render probe: output contains raw JSON dump of message object. "
                                                      "Template is not supported by minja.");
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Basic render probe output: {}", output);
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

bool probeChatTemplateCaps(ov::genai::Tokenizer& tokenizer, ChatTemplateCaps& caps) {
    if (tokenizer.get_chat_template().empty()) {
        return true;
    }
    if (!caps.supportsToolCalls) {
        return true;
    }

    auto probeStart = std::chrono::steady_clock::now();
    const std::string argNeedle = "probe_needle_xK9m";

    // Async because apply_chat_template is slow: CVS-189192
    auto strArgsFuture = std::async(std::launch::async, [&tokenizer, &argNeedle]() -> std::pair<bool, std::string> {
        try {
            ov::genai::ChatHistory history;
            history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Hello"})"));
            history.push_back(ov::genai::JsonContainer::from_json_string(
                R"({"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":"{\")" + argNeedle + R"(\":\"val\"}"}}]})"));
            auto t0 = std::chrono::steady_clock::now();
            std::string output = tokenizer.apply_chat_template(history, false);
            auto t1 = std::chrono::steady_clock::now();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe minja (string args): {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            return {true, std::move(output)};
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe minja (string args): exception: {}", e.what());
            return {false, ""};
        } catch (...) {
            return {false, ""};
        }
    });

    // Async because apply_chat_template is slow: CVS-189192
    auto objArgsFuture = std::async(std::launch::async, [&tokenizer, &argNeedle]() -> std::pair<bool, std::string> {
        try {
            ov::genai::ChatHistory history;
            history.push_back(ov::genai::JsonContainer::from_json_string(R"({"role":"user","content":"Hello"})"));
            history.push_back(ov::genai::JsonContainer::from_json_string(
                R"({"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":{")" + argNeedle + R"(":"val"}}}]})"));
            auto t0 = std::chrono::steady_clock::now();
            std::string output = tokenizer.apply_chat_template(history, false);
            auto t1 = std::chrono::steady_clock::now();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe minja (object args): {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            return {true, std::move(output)};
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe minja (object args): exception: {}", e.what());
            return {false, ""};
        } catch (...) {
            return {false, ""};
        }
    });

    auto [strOk, strOut] = strArgsFuture.get();
    auto [objOk, objOut] = objArgsFuture.get();

    auto rendersNativeArgs = [&argNeedle](const std::string& output) -> bool {
        return output.find("\"" + argNeedle + "\": \"") != std::string::npos ||         // JSON key: ["needle": "]
               output.find("\"" + argNeedle + "\":\"") != std::string::npos ||         // JSON key: ["needle":"]
               //output.find("'" + argNeedle + "': ") != std::string::npos ||           // Python dict: []'needle': ]
               output.find("<parameter=" + argNeedle + ">") != std::string::npos ||     // Qwen3-Coder XML
               output.find(argNeedle + ":<|") != std::string::npos ||                   // Gemma4: [needle:<|]
               output.find(argNeedle + "=") != std::string::npos;                       // Function-style: [needle=] for LFM2.5
    };

    bool strArgsRendersNative = strOk && rendersNativeArgs(strOut);
    bool objArgsRendersNative = objOk && rendersNativeArgs(objOut);

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe requiresObjectArguments: strRendersNative={}, objRendersNative={}",
        strArgsRendersNative, objArgsRendersNative);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe strArgs output: {}", strOut);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe objArgs output: {}", objOut);

    // Detect minja silent failure: if the output contains "tool_calls": [ it means
    // minja didn't process the template's tool call logic and just dumped raw JSON.
    static const std::string silentFailureMarker = "\"tool_calls\": [";
    bool strArgsFailed = strOk && strOut.find(silentFailureMarker) != std::string::npos;
    bool objArgsFailed = objOk && objOut.find(silentFailureMarker) != std::string::npos;

    if (strArgsFailed || objArgsFailed) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Dry-run probe: minja silently failed to render tool calls "
                                                  "(output contains raw JSON dump). Template is not supported by minja for tool calls.");
        auto probeEnd = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe completed in {} us. Result: FAILURE",
            std::chrono::duration_cast<std::chrono::microseconds>(probeEnd - probeStart).count());
        return false;
    }

    if (strArgsRendersNative || objArgsRendersNative) {
        bool probeResult = objArgsRendersNative;
        if (probeResult != caps.requiresObjectArguments) {
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Dry-run probe overrides requiresObjectArguments: {} -> {}",
                caps.requiresObjectArguments, probeResult);
        }
        caps.requiresObjectArguments = probeResult;
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe: template does not render tool_call arguments in native format, keeping string-matching result for requiresObjectArguments");
    }

    auto probeEnd = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe completed in {} us. Final result: requiresObjectArguments={}",
        std::chrono::duration_cast<std::chrono::microseconds>(probeEnd - probeStart).count(),
        caps.requiresObjectArguments);
    return true;
}

#if (PYTHON_DISABLE == 0)

bool probeChatTemplateCapsJinja(PyJinjaTemplateProcessor& templateProcessor, const std::string& modelsPath, ChatTemplateCaps& caps) {
    if (!caps.supportsToolCalls) {
        return true;
    }

    auto probeStart = std::chrono::steady_clock::now();
    const std::string argNeedle = "probe_needle_xK9m";

    std::string strArgsJson = R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":"{\")" + argNeedle + R"(\":\"val\"}"}}]}]})";
    std::string objArgsJson = R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0_ab","type":"function","function":{"name":"probe_fn","arguments":{")" + argNeedle + R"(":"val"}}}]}]})";

    std::string strOut, objOut;
    bool strOk = false, objOk = false;

    try {
        auto t0 = std::chrono::steady_clock::now();
        strOk = PyJinjaTemplateProcessor::applyChatTemplate(templateProcessor, modelsPath, strArgsJson, strOut);
        auto t1 = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja (string args): {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja (string args): exception: {}", e.what());
    } catch (...) {
    }

    try {
        auto t0 = std::chrono::steady_clock::now();
        objOk = PyJinjaTemplateProcessor::applyChatTemplate(templateProcessor, modelsPath, objArgsJson, objOut);
        auto t1 = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja (object args): {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja (object args): exception: {}", e.what());
    } catch (...) {
    }

    auto rendersNativeArgs = [&argNeedle](const std::string& output) -> bool {\
        return output.find("\"" + argNeedle + "\": \"") != std::string::npos ||         // JSON key: ["needle": "]
               output.find("\"" + argNeedle + "\":\"") != std::string::npos ||         // JSON key: ["needle":"]
               //output.find("'" + argNeedle + "': ") != std::string::npos ||           // Python dict: []'needle': ]
               output.find("<parameter=" + argNeedle + ">") != std::string::npos ||     // Qwen3-Coder XML
               output.find(argNeedle + ":<|") != std::string::npos ||                   // Gemma4: [needle:<|]
               output.find(argNeedle + "=") != std::string::npos;                       // Function-style: [needle=] for LFM2.5
    };

    bool strArgsRendersNative = strOk && rendersNativeArgs(strOut);
    bool objArgsRendersNative = objOk && rendersNativeArgs(objOut);

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja requiresObjectArguments: strRendersNative={}, objRendersNative={}",
        strArgsRendersNative, objArgsRendersNative);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe Jinja strArgs output: {}", strOut);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Dry-run probe Jinja objArgs output: {}", objOut);

    // Detect silent failure (raw JSON dump)
    static const std::string silentFailureMarker = "\"tool_calls\": [";
    bool strArgsFailed = strOk && strOut.find(silentFailureMarker) != std::string::npos;
    bool objArgsFailed = objOk && objOut.find(silentFailureMarker) != std::string::npos;

    if (strArgsFailed || objArgsFailed) {
        SPDLOG_LOGGER_WARN(llm_calculator_logger, "Dry-run probe Jinja: silently failed to render tool calls "
                                                  "(output contains raw JSON dump). Template is not supported for tool calls.");
        auto probeEnd = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja completed in {} us. Result: FAILURE",
            std::chrono::duration_cast<std::chrono::microseconds>(probeEnd - probeStart).count());
        return false;
    }

    if (strArgsRendersNative || objArgsRendersNative) {
        bool probeResult = objArgsRendersNative;
        if (probeResult != caps.requiresObjectArguments) {
            SPDLOG_LOGGER_INFO(llm_calculator_logger, "Dry-run probe Jinja overrides requiresObjectArguments: {} -> {}",
                caps.requiresObjectArguments, probeResult);
        }
        caps.requiresObjectArguments = probeResult;
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja: template does not render tool_call arguments in native format, keeping string-matching result for requiresObjectArguments");
    }

    auto probeEnd = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Dry-run probe Jinja completed in {} us. Final result: requiresObjectArguments={}",
        std::chrono::duration_cast<std::chrono::microseconds>(probeEnd - probeStart).count(),
        caps.requiresObjectArguments);
    return true;
}
#endif

}  // namespace ovms
