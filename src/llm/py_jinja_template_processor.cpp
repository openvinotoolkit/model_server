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
#include "py_jinja_template_processor.hpp"

#include <string>
#include <utility>

#include "src/port/rapidjson_document.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456)
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

bool PyJinjaTemplateProcessor::applyChatTemplate(PyJinjaTemplateProcessor& templateProcessor, std::string modelsPath, const std::string& requestBody, std::string& output) {
    if (templateProcessor.chatTemplate == nullptr) {
        output = "Error: Chat template not loaded correctly, so it cannot be applied";
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("request_body"_a = requestBody, "chat_template"_a = templateProcessor.chatTemplate->getObject(),
            "tool_chat_template"_a = templateProcessor.toolTemplate->getObject(), "models_path"_a = modelsPath,
            "bos_token"_a = templateProcessor.bosToken, "eos_token"_a = templateProcessor.eosToken);
        py::exec(R"(
            output = ""
            error = ""
            try:
                request_json = json.loads(request_body)
                messages = request_json["messages"]

                chat_template_kwargs = request_json.get("chat_template_kwargs", None)
                if chat_template_kwargs is None:
                    chat_template_kwargs = {}
                elif not isinstance(chat_template_kwargs, dict):
                    raise Exception("chat_template_kwargs must be an object")

                tools = request_json["tools"] if "tools" in request_json else None
                if tools is None:
                    output = chat_template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
                else:
                    output = tool_chat_template.render(messages=messages, tools=tools, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
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

static py::object rapidJsonValueToPyObject(const rapidjson::Value& value) {
    if (value.IsNull()) return py::none();
    if (value.IsBool()) return py::bool_(value.GetBool());
    if (value.IsInt()) return py::int_(value.GetInt());
    if (value.IsUint()) return py::int_(value.GetUint());
    if (value.IsInt64()) return py::int_(value.GetInt64());
    if (value.IsUint64()) return py::int_(value.GetUint64());
    if (value.IsDouble()) return py::float_(value.GetDouble());
    if (value.IsString()) return py::str(value.GetString());
    if (value.IsArray()) {
        py::list list;
        for (const auto& item : value.GetArray()) {
            list.append(rapidJsonValueToPyObject(item));
        }
        return list;
    }
    if (value.IsObject()) {
        py::dict dict;
        for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
            dict[py::str(it->name.GetString())] = rapidJsonValueToPyObject(it->value);
        }
        return dict;
    }
    return py::none();
}

bool PyJinjaTemplateProcessor::applyChatTemplate(PyJinjaTemplateProcessor& templateProcessor,
    ov::genai::ChatHistory& messages,
    const rapidjson::Value* tools,
    const rapidjson::Value* chatTemplateKwargs,
    std::string& output) {
    if (templateProcessor.chatTemplate == nullptr) {
        output = "Error: Chat template not loaded correctly, so it cannot be applied";
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        // Convert ChatHistory to Python list[dict] by extracting known fields
        py::list pyMessages;
        for (size_t i = 0; i < messages.size(); ++i) {
            py::dict pyMsg;
            auto role = messages[i]["role"].as_string();
            if (role.has_value()) {
                pyMsg[py::str("role")] = py::str(role.value());
            }
            auto content = messages[i]["content"].as_string();
            if (content.has_value()) {
                pyMsg[py::str("content")] = py::str(content.value());
            }
            pyMessages.append(pyMsg);
        }

        py::object pyTools = py::none();
        if (tools != nullptr && !tools->IsNull()) {
            pyTools = rapidJsonValueToPyObject(*tools);
        }

        py::dict pyKwargs;
        if (chatTemplateKwargs != nullptr && chatTemplateKwargs->IsObject()) {
            for (auto it = chatTemplateKwargs->MemberBegin(); it != chatTemplateKwargs->MemberEnd(); ++it) {
                pyKwargs[py::str(it->name.GetString())] = rapidJsonValueToPyObject(it->value);
            }
        }

        auto locals = py::dict(
            "messages"_a = pyMessages,
            "chat_template"_a = templateProcessor.chatTemplate->getObject(),
            "tool_chat_template"_a = templateProcessor.toolTemplate->getObject(),
            "bos_token"_a = templateProcessor.bosToken,
            "eos_token"_a = templateProcessor.eosToken,
            "tools"_a = pyTools,
            "chat_template_kwargs"_a = pyKwargs);
        py::exec(R"(
            output = ""
            error = ""
            try:
                if chat_template_kwargs is None:
                    chat_template_kwargs = {}
                elif not isinstance(chat_template_kwargs, dict):
                    raise Exception("chat_template_kwargs must be an object")

                if tools is None:
                    output = chat_template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
                else:
                    output = tool_chat_template.render(messages=messages, tools=tools, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
            except Exception as e:
                error = str(e)
        )",
            py::globals(), locals);

        std::string result = locals["output"].cast<std::string>();
        std::string error = locals["error"].cast<std::string>();

        if (!error.empty()) {
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
