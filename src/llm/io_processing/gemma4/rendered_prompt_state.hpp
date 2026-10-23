//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************
#pragma once

#include <string>
#include <vector>

#include <openvino/genai/generation_config.hpp>

namespace ovms {

enum class Gemma4RenderedPromptState {
    NEW_TURN,
    OPEN_THOUGHT,
    CLOSED_THOUGHT,
};

Gemma4RenderedPromptState classifyRenderedPromptState(
    const std::string& renderedPrompt,
    const std::vector<std::string>& thoughtStartTags,
    const std::string& thoughtEndTag);

Gemma4RenderedPromptState classifyGemma4RenderedPromptState(
    const std::string& renderedPrompt);

// Pure token-JSON adapter, also used by focused probes without loading a runtime.
bool adaptGemma4TokenGrammarForRenderedPrompt(std::string& grammar, const std::string& renderedPrompt);

// Returns true only when the config was changed.
bool adaptGemma4ToolGrammarForRenderedPrompt(
    ov::genai::GenerationConfig& config,
    const std::string& renderedPrompt);

}  // namespace ovms
