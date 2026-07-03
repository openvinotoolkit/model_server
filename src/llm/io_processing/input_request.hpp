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
#include <variant>
#include <vector>

#include <openvino/genai/chat_history.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovms {

// Discriminated union between chat-based and raw-prompt requests.
// ChatHistory: /v3/chat/completions and /v3/responses paths.
//   Image content arrays are preserved as JsonContainer until ImageDecodingProcessor runs.
// std::string: /v3/completions path — raw prompt as received.
using InputPayload = std::variant<ov::genai::ChatHistory, std::string>;

// Per-request data passed through the InputProcessor chain.
// parseRequest() calls extractInputRequest() which fills input + generationConfig in one step;
// InputProcessor fills the remaining output fields.
struct InputRequest {
    InputPayload input;                            // set in parseRequest()
    ov::genai::GenerationConfig generationConfig;  // set in parseRequest()
    // add_generation_prompt is folded into ChatHistory's extra_context (chat_template_kwargs).

    std::string promptText;               // written by ChatTemplateProcessor / RawPromptExtractor
    ov::Tensor inputIds;                  // written by TokenizationProcessor (all paths)
    std::vector<ov::Tensor> inputImages;  // written by ImageDecodingProcessor
    std::vector<ov::Tensor> inputVideos;
    std::vector<ov::Tensor> inputAudios;
};

}  // namespace ovms
