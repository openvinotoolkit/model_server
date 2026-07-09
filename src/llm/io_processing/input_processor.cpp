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

#include "input_processor.hpp"

#include <chrono>
#include <typeinfo>
#include <variant>

#include "../../config.hpp"
#include "../../logging.hpp"
#include "input_processors/chat_template_processor.hpp"
#include "input_processors/empty_content_array_normalization_processor.hpp"
#include "input_processors/image_decoding_processor.hpp"
#include "input_processors/chat_template_adapter.hpp"
#include "input_processors/raw_prompt_extractor.hpp"
#include "input_processors/text_content_normalization_processor.hpp"
#include "input_processors/tokenization_processor.hpp"

namespace ovms {

InputProcessor::InputProcessor(InputProcessorContext& context,
    const InputRequest& req) {
    const bool isChatPath = std::holds_alternative<ov::genai::ChatHistory>(req.input);
    // Chat template already adds special tokens; completions path needs them added by the tokenizer.
    const bool addSpecialTokens = !isChatPath;

    if (isChatPath) {
        // Normalize empty content arrays to null before any content-aware processor runs.
        processors.emplace_back(std::make_unique<EmptyContentArrayNormalizationProcessor>());

        // Flatten text-only content arrays for both LM and VLM. Arrays that contain
        // images (or other modalities) are left untouched for ImageDecodingProcessor.
        processors.emplace_back(std::make_unique<TextContentNormalizationProcessor>());

        if (context.config.isVLM) {
            const auto& settings = Config::instance().getServerSettings();
            processors.emplace_back(std::make_unique<ImageDecodingProcessor>(
                settings.allowedLocalMediaPath,
                settings.allowedMediaDomains));
        }

        if (context.chatTemplateCaps.needsWorkarounds()) {
            processors.emplace_back(std::make_unique<ChatTemplateAdapter>(context.chatTemplateCaps));
        }

#if (PYTHON_DISABLE == 0)
        // Select Jinja path (runtime-prepared first, then optional in-process fallback).
        if (!context.config.useMinja) {
            processors.emplace_back(std::make_unique<ChatTemplateProcessor>(
                context.tokenizer,
                context.preparedRuntimeChatTemplate,
                context.templateProcessor));
        } else {
            processors.emplace_back(std::make_unique<ChatTemplateProcessor>(context.tokenizer));
        }
#else
        if (!context.config.useMinja) {
            processors.emplace_back(std::make_unique<ChatTemplateProcessor>(
                context.tokenizer,
                context.preparedRuntimeChatTemplate));
        } else {
            processors.emplace_back(std::make_unique<ChatTemplateProcessor>(context.tokenizer));
        }
#endif
    } else {
        processors.emplace_back(std::make_unique<RawPromptExtractor>());
    }

    // TokenizationProcessor runs for all paths:
    // - LM: provides inputIds for inference.
    // - VLM: provides inputIds for usage statistics and max-length checks
    //   (the VLM pipeline tokenizes internally; inputIds is not passed to it).
    if (!context.config.isVLM || isChatPath) {
        processors.emplace_back(std::make_unique<TokenizationProcessor>(
            context.tokenizer, addSpecialTokens));
    }
}

absl::Status InputProcessor::process(InputRequest& req) {
    for (auto& processor : processors) {
        if (llm_calculator_logger->should_log(spdlog::level::trace)) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "InputProcessor: executing {}", typeid(*processor).name());
        }
        if (llm_calculator_logger->should_log(spdlog::level::debug)) {
            const auto start = std::chrono::steady_clock::now();
            auto status = processor->process(req);
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start);
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "InputProcessor: {} took {} us",
                typeid(*processor).name(), elapsed.count());
            if (!status.ok()) {
                return status;
            }
        } else {
            auto status = processor->process(req);
            if (!status.ok()) {
                return status;
            }
        }
    }
    return absl::OkStatus();
}

}  // namespace ovms
