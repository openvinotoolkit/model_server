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

#include "chat_template_adapter.hpp"

#include <variant>

#include "../chat_template_adapter.hpp"

namespace ovms {

ChatTemplateAdapter::ChatTemplateAdapter(const ChatTemplateCaps& caps) :
    caps(caps) {}

absl::Status ChatTemplateAdapter::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::OkStatus();
    }
    auto& chatHistory = std::get<ov::genai::ChatHistory>(req.input);
    chat_template_adapter::applyToHistory(caps, chatHistory);
    return absl::OkStatus();
}

}  // namespace ovms
