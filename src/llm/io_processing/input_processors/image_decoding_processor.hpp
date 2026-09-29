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

// Decodes image_url content entries from ChatHistory messages into tensors and
// injects <ov_genai_image_N> tags into message content.
// Active when: config.isVLM && input is ChatHistory variant.
class ImageDecodingProcessor : public BaseInputProcessor {
public:
    ImageDecodingProcessor(std::optional<std::string> allowedLocalMediaPath,
        std::optional<std::vector<std::string>> allowedMediaDomains);
    absl::Status process(InputRequest& req) override;

private:
    std::optional<std::string> allowedLocalMediaPath;
    std::optional<std::vector<std::string>> allowedMediaDomains;
};

}  // namespace ovms
