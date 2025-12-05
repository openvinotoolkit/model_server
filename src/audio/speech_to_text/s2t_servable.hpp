//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/speech_to_text/s2t_calculator.pb.h"

namespace ovms {

struct SttExecutionContext {
    std::shared_ptr<ov::genai::TextStreamer> textStreamer;
    bool sendLoopbackSignal = false;
    std::string lastStreamerCallbackOutput;
};

struct SttServable {
    std::filesystem::path parsedModelsPath;
    std::shared_ptr<ov::genai::WhisperPipeline> sttPipeline;
    std::mutex sttPipelineMutex;

    SttServable(const std::string& modelDir, const std::string& targetDevice, const std::string& graphPath) {
        auto fsModelsPath = std::filesystem::path(modelDir);
        if (fsModelsPath.is_relative()) {
            parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
        } else {
            parsedModelsPath = fsModelsPath.string();
        }
        sttPipeline = std::make_shared<ov::genai::WhisperPipeline>(parsedModelsPath.string(), targetDevice);
    }

    void createStreamer(std::shared_ptr<SttExecutionContext>& executionContext){
        executionContext->lastStreamerCallbackOutput = "";  // initialize with empty string
        auto callback = [& lastStreamerCallbackOutput = executionContext->lastStreamerCallbackOutput](std::string text) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
            lastStreamerCallbackOutput = text;
            return ov::genai::StreamingStatus::RUNNING;
        };
        ov::AnyMap streamerConfig;
        streamerConfig.insert(ov::genai::skip_special_tokens(false));
        executionContext->textStreamer = std::make_shared<ov::genai::TextStreamer>(getProperties()->tokenizer, callback, streamerConfig);
    }
};

using SttServableMap = std::unordered_map<std::string, std::shared_ptr<SttServable>>;
}  // namespace ovms
