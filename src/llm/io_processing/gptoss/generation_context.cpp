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

#include "generation_context.hpp"

#include <string_view>

namespace ovms {

GptOssGenerationContext GptOssGenerationContextAdapter::adapt(const ModelGenerationContext& context) {
    using Header = GptOssGenerationContext::Header;
    GptOssGenerationContext state;
    const std::string_view prompt = context.renderedPrompt;
    constexpr std::string_view startTag = "<|start|>";
    const std::size_t start = prompt.rfind(startTag);
    if (start == std::string_view::npos) {
        return state;
    }

    constexpr std::string_view assistant = "<|start|>assistant";
    const std::size_t afterAssistant = start + assistant.size();
    if (prompt.compare(start, assistant.size(), assistant) != 0 ||
        (afterAssistant < prompt.size() && prompt[afterAssistant] != ' ' && prompt[afterAssistant] != '<') ||
        prompt.find("<|message|>", start) != std::string_view::npos || prompt.find("<|end|>", start) != std::string_view::npos ||
        prompt.find("<|return|>", start) != std::string_view::npos || prompt.find("<|call|>", start) != std::string_view::npos) {
        return state;
    }

    std::string_view remainder = prompt.substr(afterAssistant);
    if (remainder.empty()) {
        state.header = Header::Assistant;
        return state;
    }

    constexpr std::string_view channelPrefix = "<|channel|>";
    if (remainder.size() >= channelPrefix.size() && remainder.compare(0, channelPrefix.size(), channelPrefix) == 0) {
        remainder.remove_prefix(channelPrefix.size());
        const std::size_t channelEnd = remainder.find_first_of(" \t\r\n<");
        state.channel = std::string(remainder.substr(0, channelEnd));
        if (state.channel != "analysis" && state.channel != "commentary" && state.channel != "final") {
            state.header = Header::Unsupported;
            return state;
        }
        if (channelEnd == std::string_view::npos) {
            state.header = Header::Channel;
            return state;
        }
        if (remainder[channelEnd] != ' ' && remainder[channelEnd] != '\t') {
            state.header = Header::Unsupported;
            return state;
        }
        remainder.remove_prefix(channelEnd);
        state.recipientAfterChannel = true;
    }

    constexpr std::string_view recipientPrefix = " to=functions.";
    if (remainder.size() >= recipientPrefix.size() && remainder.compare(0, recipientPrefix.size(), recipientPrefix) == 0) {
        state.header = Header::ToolRecipient;
        remainder.remove_prefix(recipientPrefix.size());
        state.partialToolName = std::string(remainder);
        if (state.partialToolName.find_first_of(" \t\r\n<") != std::string::npos) {
            state.header = Header::Unsupported;
        }
        return state;
    }

    if (state.channel.empty() || state.recipientAfterChannel) {
        state.header = Header::Unsupported;
    }
    return state;
}

}  // namespace ovms
