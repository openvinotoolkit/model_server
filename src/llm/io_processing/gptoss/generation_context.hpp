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
#pragma once

#include <string>

#include "../model_generation_context.hpp"

namespace ovms {

struct GptOssGenerationContext {
    enum class Header {
        None,
        Assistant,
        Channel,
        ToolRecipient,
        Unsupported
    };

    Header header = Header::None;
    std::string channel;
    std::string partialToolName;
    bool recipientAfterChannel = false;
};

class GptOssGenerationContextAdapter {
public:
    /**
     * - Reads only context.renderedPrompt; the generic reasoning state is unchanged.
     * - Inspects the last Harmony message for an unfinished assistant header;
     *   absent headers, non-assistant messages, and headers with a body return None.
     * - Classifies the header as Assistant, Channel, or ToolRecipient, extracting
     *   the channel, partial function name, and whether the recipient follows the channel.
     * - Marks unrecognized header fields Unsupported; consumers decide how the extracted
     *   prompt facts affect their own startup state.
     * - Returns owned state without retaining the prompt view or replaying prompt text.
     */
    static GptOssGenerationContext adapt(const ModelGenerationContext& context);
};

}  // namespace ovms
