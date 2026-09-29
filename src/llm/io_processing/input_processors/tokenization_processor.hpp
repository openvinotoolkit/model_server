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

#include <openvino/genai/tokenizer.hpp>

#include "../base_input_processor.hpp"

namespace ovms {

// Encodes req.promptText into req.inputIds using the servable tokenizer.
// Active when: all paths (LM chat, LM completions, VLM chat). For VLM the resulting
// inputIds are used for max-length checks and prompt token usage statistics only;
// the VLM pipeline tokenizes internally and does not receive inputIds.
// addSpecialTokens: false for chat path (template already added them), true for completions.
class TokenizationProcessor : public BaseInputProcessor {
public:
    TokenizationProcessor(ov::genai::Tokenizer& tokenizer, bool addSpecialTokens);
    absl::Status process(InputRequest& req) override;

private:
    ov::genai::Tokenizer& tokenizer;  // non-owning; lifetime tied to InputProcessorContext
    bool addSpecialTokens;
};

}  // namespace ovms
