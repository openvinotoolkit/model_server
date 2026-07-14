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

#include <openvino/genai/tokenizer.hpp>

#include "chat_template/caps.hpp"
#include "input_processing_config.hpp"
#if (PYTHON_DISABLE == 0)
#include "../py_jinja_template_processor.hpp"
#endif

namespace ovms {

// Holds the per-deployment resources needed by InputProcessor.
// Created once during servable initialization; reused across requests.
struct InputProcessorContext {
    InputProcessingConfig config;
    ChatTemplateCaps chatTemplateCaps;
    ov::genai::Tokenizer tokenizer;
#if (PYTHON_DISABLE == 0)
    PyJinjaTemplateProcessor* templateProcessor = nullptr;
#endif
};

}  // namespace ovms
