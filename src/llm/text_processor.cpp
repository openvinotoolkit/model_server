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
#include "text_processor.hpp"

#include <string>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

namespace ovms {

bool TextProcessor::applyChatTemplate(TextProcessor& textProcessor, std::string modelsPath, const std::string& requestBody, std::string& output) {
    if (textProcessor.chatTemplate == nullptr) {
        output = "Error: Chat template not loaded correctly, so it cannot be applied";
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("request_body"_a = requestBody, "chat_template"_a = textProcessor.chatTemplate->getObject(),
            "bos_token"_a = textProcessor.bosToken, "eos_token"_a = textProcessor.eosToken);
        py::exec(R"(
            output = ""
            error = ""
            try:
                messages = json.loads(request_body)["messages"]
                output = chat_template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True)
            except Exception as e:
                error = str(e)            
        )",
            py::globals(), locals);

        std::string result = locals["output"].cast<std::string>();
        std::string error = locals["error"].cast<std::string>();

        if (error != "") {
            output = std::move(error);
            return false;
        }

        output = std::move(result);
        return true;
    } catch (const pybind11::error_already_set& e) {
        LOG(INFO) << "Error occurred when applying chat template: " << e.what();
        output = "Unexpected error occurred when applying chat template";
    } catch (...) {
        LOG(INFO) << "Unexpected error occurred when applying chat template";
        output = "Unexpected error occurred when applying chat template";
    }
    return false;
}

}  // namespace ovms
