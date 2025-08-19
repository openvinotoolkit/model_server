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
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include <fstream>

#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../status.hpp"
#include "../filesystem.hpp"
#include "../stringutils.hpp"
#include "language_model/continuous_batching/servable.hpp"
#include "language_model/continuous_batching/servable_initializer.hpp"
#include "language_model/legacy/servable_initializer.hpp"
#include "servable.hpp"
#include "servable_initializer.hpp"
#include "visual_language_model/continuous_batching/servable.hpp"
#include "visual_language_model/legacy/servable_initializer.hpp"

namespace ovms {

static const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";
static const std::string DEFAULT_CHAT_TEMPLATE = R"({% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }})";

void GenAiServableInitializer::loadChatTemplate(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory) {
#if (PYTHON_DISABLE == 0)
    loadPyTemplateProcessor(properties, chatTemplateDirectory);
#else
    loadDefaultTemplateProcessorIfNeeded(properties);
#endif
}

static bool checkIfGGUFModel(const std::string& chatTemplatePath) {
    if (!std::filesystem::exists(chatTemplatePath))
        return false;

    if (std::filesystem::is_regular_file(chatTemplatePath) && (chatTemplatePath.find(".gguf") != std::string::npos)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Chat template path is a GGUF file: {}", chatTemplatePath);
        return true;
    }
    if (std::filesystem::is_directory(chatTemplatePath)) {
        for (const auto& entry : std::filesystem::directory_iterator(chatTemplatePath)) {
            if (entry.is_regular_file() && entry.path().filename().string().find(".gguf") != std::string::npos) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Chat template directory contains GGUF file: {}", entry.path().filename().string());
                return true;
            }
        }
    }
    return false;
}

static std::pair<std::optional<std::string>, std::optional<std::string>> getBosAndEosTokenFromTokenizerVocab(const std::shared_ptr<ov::genai::Tokenizer>& tokenizer) {
    auto vocab = tokenizer->get_vocab();
    SPDLOG_TRACE("Tokenizer vocab size: {}", vocab.size());
    auto bosTokenId = tokenizer->get_bos_token_id();
    auto eosTokenId = tokenizer->get_eos_token_id();
    std::optional<std::string> bosToken;
    std::optional<std::string> eosToken;
    // since tokenizer get_bos_token does not work for gguf we will search in map by value
    for (const auto& [token, id] : vocab) {
        // SPDLOG_TRACE("Tokenizer vocab token: {}, id: {}", token, id);
        if (id == bosTokenId) {
            bosToken = token;
        }
        if (id == eosTokenId) {
            eosToken = token;
        }
    }
    return std::make_pair(bosToken, eosToken);
}

#if (PYTHON_DISABLE == 0)
void GenAiServableInitializer::loadPyTemplateProcessor(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory) {
    py::gil_scoped_acquire acquire;
    try {
        bool isGGUFModel = checkIfGGUFModel(chatTemplateDirectory);
        // how to define locals as py::dict to pass into py:exec?
        // we need to pass tokenizer template and bos/eos tokens to python code
        // if we have GGUF model, we will use them to create a template object
        std::string tokenizerTemplate = properties->tokenizer.get_chat_template();
        std::string tokenizerBosToken = properties->tokenizer.get_bos_token();
        std::string tokenizerEosToken = properties->tokenizer.get_eos_token();
        // time measure following if statement
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        // Workaround for CVS-172426
        if (isGGUFModel && (tokenizerBosToken.empty() || tokenizerEosToken.empty())) {
            // if tokenizer bos/eos tokens are empty, we will try to get them from tokenizer vocab
            std::pair<std::optional<std::string>, std::optional<std::string>> tokens;
            tokens = getBosAndEosTokenFromTokenizerVocab(std::make_shared<ov::genai::Tokenizer>(properties->tokenizer));
            SPDLOG_TRACE("Tokens bos: {}, eos: {}", tokens.first.has_value() ? tokens.first.value() : "None", tokens.second.has_value() ? tokens.second.value() : "None");
            if (tokens.first.has_value()) {
                tokenizerBosToken = tokens.first.value();
            }
            if (tokens.second.has_value()) {
                tokenizerEosToken = tokens.second.value();
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        SPDLOG_TRACE("Time to get bos/eos tokens from tokenizer: {} ms", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        auto& tokenizer = properties->tokenizer;
        std::string bosToken = tokenizer.decode(std::vector<int64_t>({properties->tokenizer.get_bos_token_id()}), {ov::genai::skip_special_tokens(false)});
        std::string eosToken = tokenizer.decode(std::vector<int64_t>({properties->tokenizer.get_eos_token_id()}), {ov::genai::skip_special_tokens(false)});
        SPDLOG_TRACE("Tokenizer bos token: {}, eos token: {}, bos token id: {}, eos token id: {}, decoded bos: {} decoded eos:{} isGGUF:{}", tokenizerBosToken, tokenizerEosToken, properties->tokenizer.get_bos_token_id(), properties->tokenizer.get_eos_token_id(), bosToken, eosToken, isGGUFModel);
        auto locals = py::dict("tokenizer_template"_a = tokenizerTemplate,
            "tokenizer_bos_token"_a = tokenizerBosToken,
            "tokenizer_eos_token"_a = tokenizerEosToken,
            "templates_directory"_a = chatTemplateDirectory,
            "is_gguf_model"_a = isGGUFModel);
        py::exec(R"(
            # Following the logic from:
            # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/tokenization_utils_base.py#L1837
            global json
            import json
            from pathlib import Path
            global datetime
            import datetime

            global contextmanager
            from contextlib import contextmanager

            global jinja2
            import jinja2
            global ImmutableSandboxedEnvironment
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from jinja2.ext import Extension
            print("Loading chat template from directory:", templates_directory)
            gguf_bos_token = None
            gguf_eos_token = None
            gguf_template = None
            try:
               from gguf_parser import GGUFParser
               # find gguf file in templates_directory and open it with GGUFParser
               ggufFile = None
               # if templates_directory is actually a file, use it directly
               if Path(templates_directory).is_file() and templates_directory.endswith(".gguf"):
                   ggufFile = Path(templates_directory)
               else:
                   # otherwise, look for gguf files in the directory
                   print("Searching for GGUF files in:", templates_directory)
               for file in Path(templates_directory).glob("*.gguf"):
                   ggufFile = file
                   break
               print(f"GGUF file found: {ggufFile}")
               parser = GGUFParser(ggufFile)
               parser.parse()
               metadata = parser.metadata
               gguf_template = metadata.get('tokenizer.chat_template', None)
               gguf_bos_token = metadata.get('tokenizer.ggml.tokens')[metadata.get('tokenizer.ggml.bos_token_id')]
               gguf_eos_token = metadata.get('tokenizer.ggml.tokens')[metadata.get('tokenizer.ggml.eos_token_id')]
               # if gguf_bos/eos_token are None then set them to "NoNe" string # FIXME for debug
               gguf_bos_token = "NoNe" if gguf_bos_token is None else gguf_bos_token
               gguf_eos_token = "NoNe" if gguf_eos_token is None else gguf_eos_token
               # print template, bos_token and eos_token
               print("GGUF format version:", metadata.get("version"))
               print(f"Context length:{metadata.get('context_length')}")
               print("GGUF file successfully loaded:", ggufFile)
            except Exception as e:
                print(f"Caught exception while trying to load GGUF file: {e}")
            finally:
                print("GGUF file loading finished.")


            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)
            
            # Appears in some of mistral chat templates
            def strftime_now(format):
                return datetime.datetime.now().strftime(format)
            
            # Following the logic from:
            # https://github.com/huggingface/transformers/blob/7188e2e28c6d663284634732564143b820a03f8b/src/transformers/utils/chat_template_utils.py#L398
            class AssistantTracker(Extension):
                # This extension is used to track the indices of assistant-generated tokens in the rendered chat
                tags = {"generation"}

                def __init__(self, environment: ImmutableSandboxedEnvironment):
                    # The class is only initiated by jinja.
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
                        # Only track generation indices if the tracker is active
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


            # Default chat template accepts only single message and outputs only it's 'content'
            # effectively turning it into a regular prompt. 
            default_chat_template = "{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"

            bos_token = ""
            eos_token = ""
            chat_template = default_chat_template
            set_default_chat_template = False
            tool_chat_template = None

            template = None
            tool_template = None

            # Try to read template from template.jinja file
            jinja_file = Path(templates_directory + "/chat_template.jinja")
            jinja_file_legacy = Path(templates_directory + "/template.jinja")
            template_loader = jinja2.FileSystemLoader(searchpath=templates_directory)
            jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols], loader=template_loader)
            jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
            jinja_env.globals["raise_exception"] = raise_exception
            jinja_env.globals["strftime_now"] = strftime_now
            if jinja_file.is_file():
                template = jinja_env.get_template("chat_template.jinja")
            elif jinja_file_legacy.is_file():
                template = jinja_env.get_template("template.jinja")

            # Try to read data from tokenizer_config.json
            tokenizer_config_file = Path(templates_directory + "/tokenizer_config.json")
            if tokenizer_config_file.is_file():
                f = open(templates_directory + "/tokenizer_config.json", "r", encoding="utf-8")
                data = json.load(f)
                bos_token = data.get("bos_token", "")
                bos_token = "" if bos_token is None else bos_token  # Null token conversion to empty string.
                eos_token = data.get("eos_token", "")
                eos_token = "" if eos_token is None else eos_token  # Null token conversion to empty string.

                chat_template = data.get("chat_template", default_chat_template)
                if isinstance(chat_template, list):
                    for template_entry in chat_template:
                        if isinstance(template_entry, dict):
                            if template_entry.get("name") == "default":
                                chat_template = template_entry.get("template")
                            elif template_entry.get("name") == "tool_use":
                                tool_chat_template = template_entry.get("template")
            if template is None:
                if is_gguf_model and (chat_template == default_chat_template):
                    # in this case we want to get chat template from tokenizer passed to script
                    template = jinja_env.from_string(tokenizer_template)
                    bos_token = tokenizer_bos_token
                    eos_token = tokenizer_eos_token
                else:
                    template = jinja_env.from_string(chat_template)
            if tool_chat_template is not None:
                tool_template = jinja_env.from_string(tool_chat_template)
            else:
                tool_template = template
            print("Setting everything")
            # print comparison between tokenizer eos/bos/chat_template and gguf ones. Fist bos, then eos, then chat_template
            print(f"Tokenizer bos_token: {tokenizer_bos_token}, GGUF bos_token: {gguf_bos_token}")
            print(f"Tokenizer eos_token: {tokenizer_eos_token}, GGUF eos_token: {gguf_eos_token}")
            print(f"Tokenizer chat_template: \n{tokenizer_template}\n, GGUF chat_template: \n{gguf_template}")
        )",
            py::globals(), locals);
        // need to check if chat template directory contains gguf file
        std::string pythonChatTemplate = locals["chat_template"].cast<std::string>();
        SPDLOG_TRACE("Python chat template: {}", pythonChatTemplate);
        properties->templateProcessor.bosToken = locals["bos_token"].cast<std::string>();
        properties->templateProcessor.eosToken = locals["eos_token"].cast<std::string>();
        properties->templateProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
        properties->templateProcessor.toolTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["tool_template"]);
        SPDLOG_ERROR("ER");
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::cast_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::key_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (...) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with an unexpected error");
    }
    SPDLOG_ERROR("ER");
}
#else
void GenAiServableInitializer::loadDefaultTemplateProcessorIfNeeded(std::shared_ptr<GenAiServableProperties> properties) {
    const std::string modelChatTemplate = properties->tokenizer.get_chat_template();
    if (modelChatTemplate.empty()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Could not load model chat template. Using default template.");
        properties->tokenizer.set_chat_template(DEFAULT_CHAT_TEMPLATE);
    }
}
#endif

Status parseModelsPath(std::string& outPath, std::string modelsPath, std::string graphPath) {
    auto fsModelsPath = std::filesystem::path(modelsPath);
    if (fsModelsPath.is_relative()) {
        outPath = (std::filesystem::path(graphPath) / fsModelsPath).string();
    } else {
        outPath = fsModelsPath.string();
    }

    if (outPath.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is empty. ", outPath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::exists(outPath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} does not exist. ", outPath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    // if is directory or file with .gguf extension then it is ok
    // FIXME gguf format with parts
    if (std::filesystem::is_directory(outPath) || endsWith(outPath, ".gguf")) {
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is not a directory nor GGUF file ", outPath);
    return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
}

std::optional<uint32_t> parseMaxModelLength(std::string& modelsPath) {
    std::string configPath = FileSystem::appendSlash(modelsPath) + "config.json";
    std::optional<uint32_t> maxModelLength;
    if (std::filesystem::exists(configPath.c_str())) {
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) {
            return maxModelLength;
        }
        rapidjson::Document modelConfig;
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::ParseResult parseResult = modelConfig.ParseStream(isw);
        if (parseResult.Code()) {
            return maxModelLength;
        }
        std::vector<std::string> maxLengthFields = {"max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"};
        for (auto field : maxLengthFields) {
            if (modelConfig.HasMember(field.c_str()) && modelConfig[field.c_str()].IsUint()) {
                maxModelLength = modelConfig[field.c_str()].GetUint();
                break;
            }
        }
    }
    return maxModelLength;
}

Status determinePipelineType(PipelineType& pipelineType, const mediapipe::LLMCalculatorOptions& nodeOptions, const std::string& graphPath) {
    // Assuming that models_path is always set
    std::string parsedModelsPath;
    auto status = parseModelsPath(parsedModelsPath, nodeOptions.models_path(), graphPath);
    if (status != StatusCode::OK) {
        return status;
    }

    std::filesystem::path parsedModelsPathFs(parsedModelsPath);
    // Existence of embeddings models indicates we are dealing with VLM pipeline
    bool isVLM = (std::filesystem::exists(parsedModelsPathFs / "openvino_text_embeddings_model.xml") &&
                  std::filesystem::exists(parsedModelsPathFs / "openvino_vision_embeddings_model.bin"));

    // If pipeline type is not explicitly defined by the user, we need to determine it based on the content of the models directory and configuration
    if (nodeOptions.pipeline_type() == mediapipe::LLMCalculatorOptions::AUTO) {
        if (nodeOptions.device() == "NPU") {
            if (isVLM) {
                pipelineType = PipelineType::VLM;
            } else {
                pipelineType = PipelineType::LM;
            }
        } else {
            if (isVLM) {
                pipelineType = PipelineType::VLM_CB;
            } else {
                pipelineType = PipelineType::LM_CB;
            }
        }
    } else {
        switch (nodeOptions.pipeline_type()) {
        case mediapipe::LLMCalculatorOptions::LM:
            pipelineType = PipelineType::LM;
            break;
        case mediapipe::LLMCalculatorOptions::VLM:
            pipelineType = PipelineType::VLM;
            break;
        case mediapipe::LLMCalculatorOptions::LM_CB:
            pipelineType = PipelineType::LM_CB;
            break;
        case mediapipe::LLMCalculatorOptions::VLM_CB:
            pipelineType = PipelineType::VLM_CB;
            break;
        default:
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node options do not contain any recognized pipeline configuration.");
            return StatusCode::INTERNAL_ERROR;
        }

        if (isVLM && (pipelineType != PipelineType::VLM && pipelineType != PipelineType::VLM_CB)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Models directory content indicates VLM pipeline, but pipeline type is set to non-VLM type.");
            return StatusCode::INTERNAL_ERROR;
        }

        if (!isVLM && (pipelineType == PipelineType::VLM || pipelineType == PipelineType::VLM_CB)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Models directory content indicates non-VLM pipeline, but pipeline type is set to VLM type.");
            return StatusCode::INTERNAL_ERROR;
        }
    }
    return StatusCode::OK;
}

Status initializeGenAiServable(std::shared_ptr<GenAiServable>& servable, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    Status status;
    if (nodeOptions.has_models_path()) {
        // need to initialize pipelineType with some value to avoid compiler warning, determinePipelineType will set it properly
        PipelineType pipelineType{PipelineType::LM_CB};
        status = determinePipelineType(pipelineType, nodeOptions, graphPath);
        if (status != StatusCode::OK) {
            return status;
        }
        if (pipelineType == PipelineType::LM_CB) {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Initializing Language Model Continuous Batching servable");
            ContinuousBatchingServableInitializer cbServableInitializer;
            servable = std::make_shared<ContinuousBatchingServable>();
            status = cbServableInitializer.initialize(servable, nodeOptions, graphPath);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
                return status;
            }
        } else if (pipelineType == PipelineType::VLM_CB) {
            // VLM uses CB engine, so initialization part is shared (both servables share the same properties),
            // therefore we can use CB servable initializer to initialize VLM servable
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Initializing Visual Language Model Continuous Batching servable");
            ContinuousBatchingServableInitializer cbServableInitializer;
            servable = std::make_shared<VisualLanguageModelServable>();
            status = cbServableInitializer.initialize(servable, nodeOptions, graphPath);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
                return status;
            }
        } else if (pipelineType == PipelineType::LM) {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Initializing Language Model Legacy servable");
            LegacyServableInitializer legacyServableInitializer;
            status = legacyServableInitializer.initialize(servable, nodeOptions, graphPath);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
                return status;
            }
        } else if (pipelineType == PipelineType::VLM) {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Initializing Visual Language Model Legacy servable");
            VisualLanguageModelLegacyServableInitializer legacyServableInitializer;
            status = legacyServableInitializer.initialize(servable, nodeOptions, graphPath);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
                return status;
            }
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node options do not contain any recognized pipeline configuration.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node requires models_path to be set.");
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}
}  // namespace ovms
