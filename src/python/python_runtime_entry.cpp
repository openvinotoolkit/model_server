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
#include "python_runtime_env.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../logging.hpp"

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

namespace {

constexpr const char* PY_RUNTIME_INIT_ERROR_PREFIX = "OVMS_PY_RUNTIME_INIT_ERROR: ";

struct PreparedChatTemplateRuntime {
    std::string bosToken;
    std::string eosToken;
    PyObject* chatTemplate = nullptr;
    PyObject* toolTemplate = nullptr;
};

thread_local std::string lastRuntimeOutput;

void setRuntimeOutput(const std::string& outputText, const char** output) {
    lastRuntimeOutput = outputText;
    if (output != nullptr) {
        *output = lastRuntimeOutput.c_str();
    }
}

bool isInterpreterInitialized() {
    return Py_IsInitialized();
}

void setInterpreterNotInitializedError(const char** output, const char* context) {
    setRuntimeOutput(std::string(PY_RUNTIME_INIT_ERROR_PREFIX) +
                         "Python interpreter is not initialized for " + context,
        output);
}

[[maybe_unused]] bool hasOperationalPythonExecutable(std::string& details) {
#ifdef _WIN32
    const std::vector<std::string> candidates = {"python.exe", "python3.exe"};
#else
    const std::vector<std::string> candidates = {"python3", "python"};
#endif

    for (const auto& candidate : candidates) {
        if (ovms::existsExecutableInPath(candidate)) {
            return true;
        }
    }

    details = "No operational Python executable found in PATH (expected python/python3 or python.exe/python3.exe).";
    return false;
}

// Validates Python-related environment paths needed by runtime checks.
// - If PYTHONHOME is set, it must point to an existing directory.
// - If PYTHONPATH is set, at least one non-empty entry must exist on disk.
//   Non-existing entries are tolerated as long as at least one valid entry exists.
bool validateEnvPaths(std::string& details) {
    if (const char* pythonHome = std::getenv("PYTHONHOME"); pythonHome != nullptr && pythonHome[0] != '\0') {
        std::error_code ec;
        if (!std::filesystem::exists(std::filesystem::path(pythonHome), ec)) {
            details = std::string("PYTHONHOME points to non-existing path: ") + pythonHome;
            return false;
        }
    }

    if (const char* pythonPath = std::getenv("PYTHONPATH"); pythonPath != nullptr && pythonPath[0] != '\0') {
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        std::string pathValue(pythonPath);
        size_t start = 0;
        bool hadNonEmptyEntry = false;
        bool hasExistingEntry = false;
        while (start <= pathValue.size()) {
            size_t end = pathValue.find(separator, start);
            std::string directory = (end == std::string::npos) ? pathValue.substr(start) : pathValue.substr(start, end - start);
            if (!directory.empty()) {
                hadNonEmptyEntry = true;
                std::error_code ec;
                if (std::filesystem::exists(std::filesystem::path(directory), ec) && !ec) {
                    hasExistingEntry = true;
                }
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }

        if (hadNonEmptyEntry && !hasExistingEntry) {
            details = "PYTHONPATH does not contain any existing directory";
            return false;
        }
    }

    return true;
}

}  // namespace

extern "C" PYTHON_RUNTIME_EXPORT ovms::Module* OVMS_createPythonInterpreterModule() {
    ovms::initialize_named_loggers_from_default();
    return new ovms::PythonInterpreterModule();
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_validatePythonEnvironment(const char** errorMessage) {
    ovms::initialize_named_loggers_from_default();
    static thread_local std::string lastError;
    if (errorMessage != nullptr) {
        *errorMessage = nullptr;
    }

    const char* skipGlobalPyEnv = std::getenv("OVMS_TEST_SKIP_GLOBAL_PY_ENV");
    if (skipGlobalPyEnv != nullptr && std::string(skipGlobalPyEnv) == "1" && Py_IsInitialized()) {
        return true;
    }

    if (!validateEnvPaths(lastError)) {
        if (errorMessage != nullptr) {
            *errorMessage = lastError.c_str();
        }
        return false;
    }

    // PATH-based executable presence check disabled by request.
    // Keep environment path validation and embedded interpreter/import checks.
    // if (!hasOperationalPythonExecutable(lastError)) {
    //     if (errorMessage != nullptr) {
    //         *errorMessage = lastError.c_str();
    //     }
    //     return false;
    // }

    bool ownsInterpreter = false;
    bool success = false;
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
        success = true;
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

    if (success) {
        return true;
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
    if (output != nullptr) {
        *output = nullptr;
    }

    if (modelsPath == nullptr || requestBody == nullptr || chatTemplate == nullptr || bosToken == nullptr || eosToken == nullptr) {
        setRuntimeOutput("Invalid input passed to OVMS_applyChatTemplateRuntime", output);
        return false;
    }

    if (!isInterpreterInitialized()) {
        setInterpreterNotInitializedError(output, "chat template application");
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

                def __init__(self, environment):
                    super().__init__(environment)
                    environment.extend(activate_tracker=self.activate_tracker)
                    self._rendered_blocks = None
                    self._generation_indices = None

                def parse(self, parser):
                    lineno = next(parser.stream).lineno
                    body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                    return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

                @jinja2.pass_eval_context
                def _generation_support(self, context, caller):
                    rv = caller()
                    if self.is_active():
                        start_index = len("".join(self._rendered_blocks))
                        end_index = start_index + len(rv)
                        self._generation_indices.append((start_index, end_index))
                    return rv

                def is_active(self):
                    return self._rendered_blocks or self._generation_indices

                def activate_tracker(self, rendered_blocks, generation_indices):
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
            locals, locals);

        std::string error = locals["error"].cast<std::string>();
        if (!error.empty()) {
            setRuntimeOutput(error, output);
            return false;
        }

        setRuntimeOutput(locals["output"].cast<std::string>(), output);
        return true;
    } catch (const py::error_already_set& e) {
        setRuntimeOutput(e.what(), output);
    } catch (const std::exception& e) {
        setRuntimeOutput(e.what(), output);
    } catch (...) {
        setRuntimeOutput("Unexpected runtime error in OVMS_applyChatTemplateRuntime", output);
    }

    return false;
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_createPreparedChatTemplateRuntime(
    const char* modelsPath,
    const char* chatTemplate,
    const char* bosToken,
    const char* eosToken,
    void** preparedHandle,
    const char** output) {
    if (output != nullptr) {
        *output = nullptr;
    }
    if (preparedHandle != nullptr) {
        *preparedHandle = nullptr;
    }

    if (modelsPath == nullptr || chatTemplate == nullptr || bosToken == nullptr || eosToken == nullptr || preparedHandle == nullptr) {
        setRuntimeOutput("Invalid input passed to OVMS_createPreparedChatTemplateRuntime", output);
        return false;
    }

    if (!isInterpreterInitialized()) {
        setInterpreterNotInitializedError(output, "chat template preparation");
        return false;
    }

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict(
            "models_path"_a = std::string(modelsPath),
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

                def __init__(self, environment):
                    super().__init__(environment)
                    environment.extend(activate_tracker=self.activate_tracker)
                    self._rendered_blocks = None
                    self._generation_indices = None

                def parse(self, parser):
                    lineno = next(parser.stream).lineno
                    body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                    return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

                @jinja2.pass_eval_context
                def _generation_support(self, context, caller):
                    rv = caller()
                    if self.is_active():
                        start_index = len("".join(self._rendered_blocks))
                        end_index = start_index + len(rv)
                        self._generation_indices.append((start_index, end_index))
                    return rv

                def is_active(self):
                    return self._rendered_blocks or self._generation_indices

                def activate_tracker(self, rendered_blocks, generation_indices):
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

            prepared_chat_template = None
            prepared_tool_template = None
            error = ""

            try:
                template_loader = jinja2.FileSystemLoader(searchpath=models_path)
                jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols], loader=template_loader)
                jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                jinja_env.globals["raise_exception"] = raise_exception
                jinja_env.globals["strftime_now"] = strftime_now
                jinja_env.filters["from_json"] = json.loads
                jinja_env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)

                tool_chat_template = None

                tokenizer_config_file = Path(models_path) / "tokenizer_config.json"
                if tokenizer_config_file.is_file():
                    with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chat_template_from_tokenizer_config = data.get("chat_template", None)
                    if isinstance(chat_template_from_tokenizer_config, list):
                        for template_entry in chat_template_from_tokenizer_config:
                            if isinstance(template_entry, dict) and template_entry.get("name") == "tool_use":
                                tool_chat_template = template_entry.get("template")

                additional_templates_dir = Path(models_path) / "additional_chat_templates"
                tool_use_template_file = additional_templates_dir / "tool_use.jinja"
                if tool_use_template_file.is_file():
                    with open(tool_use_template_file, "r", encoding="utf-8") as f:
                        tool_chat_template = f.read()

                chat_template_jinja_file = Path(models_path) / "chat_template.jinja"
                if chat_template_jinja_file.is_file():
                    with open(chat_template_jinja_file, "r", encoding="utf-8") as f:
                        chat_template = f.read()

                prepared_chat_template = jinja_env.from_string(chat_template)
                prepared_tool_template = jinja_env.from_string(tool_chat_template) if tool_chat_template is not None else prepared_chat_template
            except Exception as e:
                error = str(e)
        )",
            locals, locals);

        std::string error = locals["error"].cast<std::string>();
        if (!error.empty()) {
            setRuntimeOutput(error, output);
            return false;
        }

        auto chatTemplateObject = locals["prepared_chat_template"].cast<py::object>();
        auto toolTemplateObject = locals["prepared_tool_template"].cast<py::object>();
        auto prepared = std::make_unique<PreparedChatTemplateRuntime>();
        prepared->bosToken = bosToken;
        prepared->eosToken = eosToken;
        prepared->chatTemplate = chatTemplateObject.release().ptr();
        prepared->toolTemplate = toolTemplateObject.release().ptr();
        *preparedHandle = prepared.release();
        return true;
    } catch (const py::error_already_set& e) {
        setRuntimeOutput(e.what(), output);
    } catch (const std::exception& e) {
        setRuntimeOutput(e.what(), output);
    } catch (...) {
        setRuntimeOutput("Unexpected runtime error in OVMS_createPreparedChatTemplateRuntime", output);
    }

    return false;
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_applyPreparedChatTemplateRuntime(
    void* preparedHandle,
    const char* requestBody,
    const char** output) {
    if (output != nullptr) {
        *output = nullptr;
    }

    if (preparedHandle == nullptr || requestBody == nullptr) {
        setRuntimeOutput("Invalid input passed to OVMS_applyPreparedChatTemplateRuntime", output);
        return false;
    }

    if (!isInterpreterInitialized()) {
        setInterpreterNotInitializedError(output, "prepared chat template application");
        return false;
    }

    auto* prepared = reinterpret_cast<PreparedChatTemplateRuntime*>(preparedHandle);

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict(
            "request_body"_a = std::string(requestBody),
            "chat_template"_a = py::reinterpret_borrow<py::object>(prepared->chatTemplate),
            "tool_chat_template"_a = py::reinterpret_borrow<py::object>(prepared->toolTemplate),
            "bos_token"_a = prepared->bosToken,
            "eos_token"_a = prepared->eosToken);

        py::exec(R"(
            import json
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
            locals, locals);

        std::string error = locals["error"].cast<std::string>();
        if (!error.empty()) {
            setRuntimeOutput(error, output);
            return false;
        }

        setRuntimeOutput(locals["output"].cast<std::string>(), output);
        return true;
    } catch (const py::error_already_set& e) {
        setRuntimeOutput(e.what(), output);
    } catch (const std::exception& e) {
        setRuntimeOutput(e.what(), output);
    } catch (...) {
        setRuntimeOutput("Unexpected runtime error in OVMS_applyPreparedChatTemplateRuntime", output);
    }

    return false;
}

extern "C" PYTHON_RUNTIME_EXPORT void OVMS_destroyPreparedChatTemplateRuntime(void* preparedHandle) {
    if (preparedHandle == nullptr) {
        return;
    }

    auto* prepared = reinterpret_cast<PreparedChatTemplateRuntime*>(preparedHandle);
    if (!Py_IsInitialized()) {
        delete prepared;
        return;
    }

    py::gil_scoped_acquire acquire;
    if (prepared->chatTemplate != nullptr) {
        py::reinterpret_steal<py::object>(prepared->chatTemplate);
        prepared->chatTemplate = nullptr;
    }
    if (prepared->toolTemplate != nullptr) {
        py::reinterpret_steal<py::object>(prepared->toolTemplate);
        prepared->toolTemplate = nullptr;
    }
    delete prepared;
}

#undef PYTHON_RUNTIME_EXPORT
