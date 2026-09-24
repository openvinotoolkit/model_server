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

#include <optional>
#include <string>
#include <vector>

#include "../base_input_processor.hpp"

namespace ovms {

// Decodes video_url content entries from ChatHistory messages into video tensors
// and rewrites each video_url part into a text part carrying an
// <ov_genai_video_N> tag. Runs after ImageDecodingProcessor and before
// TextContentNormalizationProcessor, which then flattens the message content
// (text + image tags + video tags) into a single string.
// Each video_url entry carries a "url" array of already-decoded frame references
// (base64 data URIs, HTTP/HTTPS URLs or local file paths); mp4 decoding is done
// on the client side.
// Active when: config.isVLM && input is ChatHistory variant.
class VideoFramesProcessor : public BaseInputProcessor {
public:
    VideoFramesProcessor(std::optional<std::string> allowedLocalMediaPath,
        std::optional<std::vector<std::string>> allowedMediaDomains);
    absl::Status process(InputRequest& req) override;

private:
    std::optional<std::string> allowedLocalMediaPath;
    std::optional<std::vector<std::string>> allowedMediaDomains;
};

}  // namespace ovms
