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

#include "../module.hpp"
#include "pythoninterpretermodule.hpp"

#include <string>

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

namespace py = pybind11;
using namespace py::literals;

#if defined(_WIN32)
#define PYTHON_RUNTIME_EXPORT __declspec(dllexport)
#else
#define PYTHON_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

extern "C" PYTHON_RUNTIME_EXPORT ovms::Module* OVMS_createPythonInterpreterModule() {
    return new ovms::PythonInterpreterModule();
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_validatePythonEnvironment(const char** errorMessage) {
    static thread_local std::string lastError;
    if (errorMessage != nullptr) {
        *errorMessage = nullptr;
    }

    bool ownsInterpreter = false;
    try {
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
            ownsInterpreter = true;
        }
        {
            py::gil_scoped_acquire acquire;
            // Validate that OVMS Python bindings are importable and executable.
            py::module_::import("pyovms");
        }
        if (ownsInterpreter) {
            py::finalize_interpreter();
        }
        return true;
    } catch (const py::error_already_set& e) {
        lastError = e.what();
    } catch (const std::exception& e) {
        lastError = e.what();
    } catch (...) {
        lastError = "Unknown python runtime validation error";
    }

    if (ownsInterpreter && Py_IsInitialized()) {
        py::finalize_interpreter();
    }
    if (errorMessage != nullptr) {
        *errorMessage = lastError.c_str();
    }
    return false;
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_applyChatTemplateRuntime(
    const char* modelsPath,
    const char* requestBody,
    const char* chatTemplate,
    const char* bosToken,
    const char* eosToken,
    const char** output) {
    static thread_local std::string lastOutput;
    if (output != nullptr) {
        *output = nullptr;
    }

    if (modelsPath == nullptr || requestBody == nullptr || chatTemplate == nullptr || bosToken == nullptr || eosToken == nullptr) {
        lastOutput = "Invalid input passed to OVMS_applyChatTemplateRuntime";
        if (output != nullptr) {
            *output = lastOutput.c_str();
        }
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict(
            "models_path"_a = std::string(modelsPath),
            "request_body"_a = std::string(requestBody),
            "chat_template"_a = std::string(chatTemplate),
            "bos_token"_a = std::string(bosToken),
            "eos_token"_a = std::string(eosToken));

        py::exec(R"(
            import json
            from pathlib import Path
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from jinja2.ext import Extension

            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)

            def strftime_now(format):
                import datetime as _dt
                return _dt.datetime.now().strftime(format)

            class AssistantTracker(Extension):
                tags = {"generation"}

                def __init__(self, environment: ImmutableSandboxedEnvironment):
                    super().__init__(environment)
                    environment.extend(activate_tracker=self.activate_tracker)
                    self._rendered_blocks = None
                    self._generation_indices = None

                def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
                    lineno = next(parser.stream).lineno
                    body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                    return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

                @jinja2.pass_eval_context
                def _generation_support(self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro) -> str:
                    rv = caller()
                    if self.is_active():
                        start_index = len("".join(self._rendered_blocks))
                        end_index = start_index + len(rv)
                        self._generation_indices.append((start_index, end_index))
                    return rv

                def is_active(self) -> bool:
                    return self._rendered_blocks or self._generation_indices

                def activate_tracker(self, rendered_blocks: list[int], generation_indices: list[int]):
                    class _TrackerGuard:
                        def __enter__(_self):
                            if self.is_active():
                                raise ValueError("AssistantTracker should not be reused before closed")
                            self._rendered_blocks = rendered_blocks
                            self._generation_indices = generation_indices

                        def __exit__(_self, _exc_type, _exc, _tb):
                            self._rendered_blocks = None
                            self._generation_indices = None

                    return _TrackerGuard()

            output = ""
            error = ""
            tool_chat_template = None

            try:
                templates_directory = models_path
                template_loader = jinja2.FileSystemLoader(searchpath=templates_directory)
                jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols], loader=template_loader)
                jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                jinja_env.globals["raise_exception"] = raise_exception
                jinja_env.globals["strftime_now"] = strftime_now
                jinja_env.filters["from_json"] = json.loads
                jinja_env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)

                tokenizer_config_file = Path(templates_directory + "/tokenizer_config.json")
                if tokenizer_config_file.is_file():
                    with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chat_template_from_tokenizer_config = data.get("chat_template", None)
                    if isinstance(chat_template_from_tokenizer_config, list):
                        for template_entry in chat_template_from_tokenizer_config:
                            if isinstance(template_entry, dict) and template_entry.get("name") == "tool_use":
                                tool_chat_template = template_entry.get("template")

                additional_templates_dir = Path(templates_directory + "/additional_chat_templates")
                tool_use_template_file = additional_templates_dir / "tool_use.jinja"
                if tool_use_template_file.is_file():
                    with open(tool_use_template_file, "r", encoding="utf-8") as f:
                        tool_chat_template = f.read()

                chat_template_jinja_file = Path(templates_directory + "/chat_template.jinja")
                if chat_template_jinja_file.is_file():
                    with open(chat_template_jinja_file, "r", encoding="utf-8") as f:
                        chat_template = f.read()

                template = jinja_env.from_string(chat_template)
                tool_template = jinja_env.from_string(tool_chat_template) if tool_chat_template is not None else template

                request_json = json.loads(request_body)
                messages = request_json["messages"]

                chat_template_kwargs = request_json.get("chat_template_kwargs", None)
                if chat_template_kwargs is None:
                    chat_template_kwargs = {}
                elif not isinstance(chat_template_kwargs, dict):
                    raise Exception("chat_template_kwargs must be an object")

                tools = request_json["tools"] if "tools" in request_json else None
                if tools is None:
                    output = template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
                else:
                    output = tool_template.render(messages=messages, tools=tools, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True, **chat_template_kwargs)
            except Exception as e:
                error = str(e)
        )",
            py::globals(), locals);

        std::string error = locals["error"].cast<std::string>();
        if (!error.empty()) {
            lastOutput = error;
            if (output != nullptr) {
                *output = lastOutput.c_str();
            }
            return false;
        }

        lastOutput = locals["output"].cast<std::string>();
        if (output != nullptr) {
            *output = lastOutput.c_str();
        }
        return true;
    } catch (const py::error_already_set& e) {
        lastOutput = e.what();
    } catch (const std::exception& e) {
        lastOutput = e.what();
    } catch (...) {
        lastOutput = "Unexpected runtime error in OVMS_applyChatTemplateRuntime";
    }

    if (output != nullptr) {
        *output = lastOutput.c_str();
    }
    return false;
}

#undef PYTHON_RUNTIME_EXPORT
