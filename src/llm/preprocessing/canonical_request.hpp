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

#include <functional>
#include <optional>
#include <string>
#include <variant>

#include <openvino/genai/json_container.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovms {

// Forward declarations
using ImageHistory = std::vector<std::pair<size_t, ov::Tensor>>;

enum class RendererType {
    CPP_TOKENIZER,
    PY_JINJA,
};

// For C++ renderer path (tokenizer.apply_chat_template)
struct CppPath {
    std::reference_wrapper<const ov::genai::ChatHistory> chatHistory;
    std::reference_wrapper<const ImageHistory> imageHistory;
    std::optional<ov::genai::JsonContainer> tools;
    std::optional<ov::genai::JsonContainer> chatTemplateKwargs;
    std::optional<std::string> rawPrompt;
    bool addGenerationPrompt = true;
};

// For Python Jinja renderer path
struct PyPath {
    std::reference_wrapper<const std::string> processedJson;
};

// Single variant type: either C++ data or Python data, never both
using CanonicalRequest = std::variant<CppPath, PyPath>;

}  // namespace ovms
