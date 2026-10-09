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
#include "python_jinja_test_utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

#include "src/python/utils.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace ovms::test {
namespace {
const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";
}  // namespace

void loadPyJinjaTemplateProcessor(PyJinjaTemplateProcessor& templateProcessor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& templatesDirectory) {
    if (tokenizer == ov::genai::Tokenizer()) {
        SPDLOG_ERROR("Tokenizer is not initialized. Cannot load test Jinja template processor.");
        return;
    }
    std::string chatTemplate = tokenizer.get_original_chat_template();
    const std::string bosToken = tokenizer.get_bos_token();
    const std::string eosToken = tokenizer.get_eos_token();
    if (chatTemplate.empty()) {
        SPDLOG_ERROR("Chat template was not found in model files.");
        return;
    }

    templateProcessor.bosToken = bosToken;
    templateProcessor.eosToken = eosToken;

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("chat_template"_a = chatTemplate,
            "templates_directory"_a = templatesDirectory);
        py::exec(R"(
            global json
            import json
            from pathlib import Path
            global contextmanager
            from contextlib import contextmanager
            global jinja2
            import jinja2
            global ImmutableSandboxedEnvironment
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            global Extension
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

                @contextmanager
                def activate_tracker(self, rendered_blocks: list[int], generation_indices: list[int]):
                    try:
                        if self.is_active():
                            raise ValueError("AssistantTracker should not be reused before closed")
                        self._rendered_blocks = rendered_blocks
                        self._generation_indices = generation_indices
                        yield
                    finally:
                        self._rendered_blocks = None
                        self._generation_indices = None

            tool_chat_template = None
            template = None
            tool_template = None
            templates_directory = templates_directory
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
            if tool_chat_template is not None:
                tool_template = jinja_env.from_string(tool_chat_template)
            else:
                tool_template = template
        )",
            py::globals(), locals);

        templateProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
        templateProcessor.toolTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["tool_template"]);
    } catch (const py::error_already_set& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Test Jinja template loading failed: {}", e.what());
    } catch (const std::exception& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Test Jinja template loading failed: {}", e.what());
    } catch (...) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Test Jinja template loading failed with an unexpected error");
    }
}

}  // namespace ovms::test
