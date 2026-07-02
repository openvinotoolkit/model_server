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

#include "../../chat_template_caps.hpp"
#include "../base_input_processor.hpp"

namespace ovms {

// Applies input workarounds to ChatHistory based on detected ChatTemplateCaps.
// Runs before ChatTemplateProcessor so the template receives corrected input.
class InputWorkaroundsProcessor : public BaseInputProcessor {
public:
    explicit InputWorkaroundsProcessor(const ChatTemplateCaps& caps);

    absl::Status process(InputRequest& req) override;

private:
    const ChatTemplateCaps& caps;
};

}  // namespace ovms
