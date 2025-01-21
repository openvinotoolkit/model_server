//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include <openvino/openvino.hpp>
// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "src/python/utils.hpp"

namespace ovms {

class TextProcessor {
public:
    std::string bosToken = "";
    std::string eosToken = "";
    std::unique_ptr<PyObjectWrapper<py::object>> chatTemplate = nullptr;

    static bool applyChatTemplate(TextProcessor& textProcessor, std::string modelsPath, const std::string& requestBody, std::string& output);
};

template <typename T>
static std::string packPromptTokens(T* input, size_t size) {
    std::stringstream ss;
    ss << "prompt_token_ids: [";
    for (size_t i = 0; i < size; i++) {
        if (i == 0)
            ss << input[i];
        else
            ss << ", " << input[i];
    }
    ss << "]";
    return ss.str();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static std::string getPromptTokensString(const ov::Tensor& tensor) {
    size_t size = tensor.get_size();
    switch (tensor.get_element_type()) {
    case ov::element::i32:
        return packPromptTokens(tensor.data<int>(), size);
    case ov::element::i16:
        return packPromptTokens(tensor.data<int16_t>(), size);
    case ov::element::i64:
        return packPromptTokens(tensor.data<int64_t>(), size);
    case ov::element::f32:
        return packPromptTokens(tensor.data<float>(), size);
    case ov::element::f64:
        return packPromptTokens(tensor.data<double>(), size);
    default: {
        std::stringstream ss;
        ss << "Could not pack input tokens for element type: " << tensor.get_element_type();
        return ss.str();
    }
    }
}
#pragma GCC diagnostic pop

}  // namespace ovms
