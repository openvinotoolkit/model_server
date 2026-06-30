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
#include "chat_template_analyzer.hpp"

#include <string>

namespace ovms {

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// TODO: remove comments before analysis
// TODO: expect GenAI to fix bug + dry-runs on separate threads?
ChatTemplateAnalysisResult ChatTemplateAnalyzer::analyze(const std::string& templateSource) {
    ChatTemplateAnalysisResult result;
    if (templateSource.empty()) {
        return result;
    }

    // GPT-OSS detection — must be before other checks as it has a unique marker
    if (contains(templateSource, "<|channel|>")) {
        result.detectedModelFamily = "gptoss";
        result.detectedToolParser = "gptoss";
        result.detectedReasoningParser = "gptoss";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        return result;
    }

    // Gemma4 detection
    if (contains(templateSource, "'<|tool_call>call:'") || contains(templateSource, "<|tool_call>call:")) {
        result.detectedModelFamily = "gemma4";
        result.detectedToolParser = "gemma4";
        result.detectedReasoningParser = "gemma4";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        result.caps.requiresObjectArguments = true;
        return result;
    }

    // Qwen3-Coder detection — uses <parameter= XML style
    if (contains(templateSource, "<parameter=") && contains(templateSource, "</parameter>") && contains(templateSource, "<function=")) {
        result.detectedModelFamily = "qwen3coder";
        result.detectedToolParser = "qwen3coder";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        // Check for reasoning support (think tags)
        if (contains(templateSource, "<think>") || contains(templateSource, "</think>")) {
            result.detectedReasoningParser = "qwen3";
        }
        return result;
    }

    // LFM2 detection
    if (contains(templateSource, "<|assistant_tool_call|>") || contains(templateSource, "<|tool_call_start|>")) {
        result.detectedModelFamily = "lfm2";
        result.detectedToolParser = "lfm2";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        return result;
    }

    // Phi-4 detection — uses "functools[" marker for tool calls
    if (contains(templateSource, "functools")) {
        result.detectedModelFamily = "phi4";
        result.detectedToolParser = "phi4";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        return result;
    }

    // Devstral detection — uses [TOOL_CALLS]name[ARGS]json format (unique [ARGS] marker)
    if (contains(templateSource, "[TOOL_CALLS]") && contains(templateSource, "[ARGS]")) {
        result.detectedModelFamily = "devstral";
        result.detectedToolParser = "devstral";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        return result;
    }

    // Mistral detection — uses [TOOL_CALLS] without [TOOL_RESULTS] or uses [AVAILABLE_TOOLS]
    if (contains(templateSource, "[TOOL_CALLS]") || (contains(templateSource, "[AVAILABLE_TOOLS]") && contains(templateSource, "[/AVAILABLE_TOOLS]"))) {
        result.detectedModelFamily = "mistral";
        result.detectedToolParser = "mistral";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        return result;
    }

    // Llama3 detection — <|python_tag|>
    if (contains(templateSource, "<|python_tag|>")) {
        result.detectedModelFamily = "llama3";
        result.detectedToolParser = "llama3";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        result.caps.requiresNonNullContent = true;
        return result;
    }

    // Hermes3/Qwen detection — <tool_call> / </tool_call> (without <parameter= which is Qwen3-Coder, already checked above)
    if (contains(templateSource, "<tool_call>") && contains(templateSource, "</tool_call>")) {
        result.detectedModelFamily = "hermes3";
        result.detectedToolParser = "hermes3";
        result.caps.supportsToolCalls = true;
        result.caps.supportsTools = true;
        result.caps.supportsToolResponses = true;
        // Check for reasoning support (think tags in Qwen3)
        if (contains(templateSource, "<think>") || contains(templateSource, "content.split('</think>')")) {
            result.detectedReasoningParser = "qwen3";
        }
        return result;
    }

    // Reasoning-only detection (no tool parser matched but template has reasoning tags)
    if (contains(templateSource, "<think>") || contains(templateSource, "content.split('</think>')")) {
        result.detectedReasoningParser = "qwen3";
    }

    return result;
}

}  // namespace ovms
