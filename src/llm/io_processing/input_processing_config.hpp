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

namespace ovms {

// Deployment-level configuration for InputProcessor, populated once at servable init.
struct InputProcessingConfig {
    // True for VLM servables. Enables ImageDecodingProcessor; TokenizationProcessor
    // still runs (to populate inputIds for usage statistics and max-length checks)
    // but inputIds is not passed to the VLM pipeline for inference.
    bool isVLM = false;
    // True when the GenAI built-in tokenizer.apply_chat_template() should be used
    // even on Python-enabled builds (i.e. ChatTemplateMode::MINJA).
    // False (default) uses PyJinjaTemplateProcessor when PYTHON_DISABLE==0.
    bool useMinja = false;
};

}  // namespace ovms
