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

#include <memory>
#include <mutex>
#include <string>

namespace ov::genai {
class ASRPipeline;
}  // namespace ov::genai

namespace ovms {

/// Shared access to the ASR pipeline of a served speech-to-text graph.
/// Holding the handle keeps the pipeline alive even if the graph is reloaded or retired meanwhile.
struct SttPipelineHandle {
    std::shared_ptr<ov::genai::ASRPipeline> pipeline;
    /// Serializes ASRPipeline::generate() calls with the other users of the served pipeline.
    std::shared_ptr<std::mutex> pipelineMutex;
    std::string modelsPath;
};

}  // namespace ovms
