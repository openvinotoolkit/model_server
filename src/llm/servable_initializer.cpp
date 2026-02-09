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
#include "../json_parser.hpp"
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

void GenAiServableInitializer::loadChatTemplate(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory) {
#if (PYTHON_DISABLE == 0)
    ExtraGenerationInfo extraGenInfo = readExtraGenerationInfo(properties, chatTemplateDirectory);
    loadPyTemplateProcessor(properties, extraGenInfo);
#else
    if (properties->tokenizer.get_chat_template().empty()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, CHAT_TEMPLATE_WARNING_MESSAGE);
    }
#endif
    // In non-python build, GenAI handles chat template loading
}

#if (PYTHON_DISABLE == 0)
// Helper function for case-insensitive comparison of file extensions
static bool hasGGUFExtension(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    if (ext.size() != 5)  // ".gguf" is 5 characters
        return false;
    // Compare case-insensitively
    return std::equal(ext.begin(), ext.end(), ".gguf",
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

static bool checkIfGGUFModel(const std::string& modelDirectoryPath) {
    if (!std::filesystem::exists(modelDirectoryPath))
        return false;

    std::filesystem::path modelPath(modelDirectoryPath);
    if (std::filesystem::is_regular_file(modelPath) && hasGGUFExtension(modelPath)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model path is a GGUF file: {}", modelDirectoryPath);
        return true;
    }
    if (std::filesystem::is_directory(modelPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(modelPath)) {
            if (entry.is_regular_file() && hasGGUFExtension(entry.path())) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model path is a directory that contains GGUF file: {}", entry.path().filename().string());
                return true;
            }
        }
    }
    return false;
}

ExtraGenerationInfo GenAiServableInitializer::readExtraGenerationInfo(std::shared_ptr<GenAiServableProperties> properties, const std::string& chatTemplateDirectory) {
    ExtraGenerationInfo extraGenInfo;
    bool isGgufModel = checkIfGGUFModel(chatTemplateDirectory);
    // we need to pass tokenizer template and bos/eos tokens to python code
    // if we have GGUF model, we will use them to create a template object
    std::string tokenizerTemplate;
    std::string tokenizerBosToken;
    std::string tokenizerEosToken;
    if (isGgufModel) {
        tokenizerTemplate = properties->tokenizer.get_chat_template();
        tokenizerBosToken = properties->tokenizer.get_bos_token();
        tokenizerEosToken = properties->tokenizer.get_eos_token();

        SPDLOG_TRACE("Tokenizer bos token: {}, eos token: {}, bos token id: {}, eos token id: {} isGGUF:{} chat_template from tokenizer: \n{}",
            tokenizerBosToken, tokenizerEosToken, properties->tokenizer.get_bos_token_id(), properties->tokenizer.get_eos_token_id(), isGgufModel, tokenizerTemplate);

        extraGenInfo.bosTokenFromTokenizer = tokenizerBosToken;
        extraGenInfo.bosTokenIdFromTokenizer = properties->tokenizer.get_bos_token_id();
        extraGenInfo.eosTokenFromTokenizer = tokenizerEosToken;
        extraGenInfo.eosTokenIdFromTokenizer = properties->tokenizer.get_eos_token_id();
        extraGenInfo.chatTemplateFromTokenizer = tokenizerTemplate;
    }

    extraGenInfo.chatTemplateDirectory = chatTemplateDirectory;
    extraGenInfo.isGgufModel = isGgufModel;

    return extraGenInfo;
}

void GenAiServableInitializer::loadPyTemplateProcessor(std::shared_ptr<GenAiServableProperties> properties, const ExtraGenerationInfo& extraGenInfo) {
    // At this point tokenizer cannot be uninitialized as we need to access its methods for prepare for chat template processing
    if (properties->tokenizer == ov::genai::Tokenizer()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Tokenizer is not initialized. Cannot load chat template processor.");
        return;
    }
    std::string chatTemplate = properties->tokenizer.get_original_chat_template();
    std::string bosToken = properties->tokenizer.get_bos_token();
    std::string eosToken = properties->tokenizer.get_eos_token();
    if (bosToken.empty()) {
        SPDLOG_ERROR("BOS token was not found in model files.");
        return;
    }
    if (eosToken.empty()) {
        SPDLOG_ERROR("EOS token was not found in model files.");
        return;
    }
    if (chatTemplate.empty()) {
        SPDLOG_ERROR("Chat template was not found in model files.");
        return;
    }

    properties->templateProcessor.bosToken = bosToken;
    properties->templateProcessor.eosToken = eosToken;

    py::gil_scoped_acquire acquire;
    try {
        auto locals = py::dict("chat_template"_a = chatTemplate,
            "templates_directory"_a = extraGenInfo.chatTemplateDirectory);
        py::exec(R"(
            # Following the logic from:
            # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/tokenization_utils_base.py#L1837
            global json
            import json
            from pathlib import Path

            global contextmanager
            from contextlib import contextmanager

            global jinja2
            import jinja2
            global ImmutableSandboxedEnvironment
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from jinja2.ext import Extension

            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)

            # Appears in some of mistral chat templates and gpt-oss chat templates
            def strftime_now(format):
                import datetime as _dt
                return _dt.datetime.now().strftime(format)

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

            
            # Optional dedicated tool chat template (might not be present)
            tool_chat_template = None

            # Variables needed to be set at the end of this script execution
            template = None
            tool_template = None

            # Load Jinja2 environment
            template_loader = jinja2.FileSystemLoader(searchpath=templates_directory)
            jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols], loader=template_loader)
            jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
            jinja_env.globals["raise_exception"] = raise_exception
            jinja_env.globals["strftime_now"] = strftime_now
            jinja_env.filters["from_json"] = json.loads

            # Try to read data from tokenizer_config.json to get additional tool chat template if present
            tokenizer_config_file = Path(templates_directory + "/tokenizer_config.json")
            if tokenizer_config_file.is_file():
                with open(templates_directory + "/tokenizer_config.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                chat_template_from_tokenizer_config = data.get("chat_template", None)
                if isinstance(chat_template_from_tokenizer_config, list):
                    for template_entry in chat_template_from_tokenizer_config:
                        if isinstance(template_entry, dict):
                            if template_entry.get("name") == "tool_use":
                                tool_chat_template = template_entry.get("template")
            
            # Try read tool_use.jinja template file from additional_chat_templates directory if exists
            additional_templates_dir = Path(templates_directory + "/additional_chat_templates")
            tool_use_template_file = additional_templates_dir / "tool_use.jinja"
            if tool_use_template_file.is_file():
                with open(tool_use_template_file, "r", encoding="utf-8") as f:
                    tool_chat_template = f.read()
            
            # Temporary override of GenAI value as we want jinja file to have priority over tokenizer RT info
            chat_template_jinja_file = Path(templates_directory + "/chat_template.jinja")
            if chat_template_jinja_file.is_file():
                with open(chat_template_jinja_file, "r", encoding="utf-8") as f:
                    chat_template = f.read()
                print("\n[INFO] Reading chat template from chat_template.jinja file in model directory.")
            
            # Load templates from strings
            template = jinja_env.from_string(chat_template)
            if tool_chat_template is not None:
                tool_template = jinja_env.from_string(tool_chat_template)
            else:
                tool_template = template
        )",
            py::globals(), locals);

        properties->templateProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
        properties->templateProcessor.toolTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["tool_template"]);

        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Loaded Python Jinja template processor. Bos token: {}, Eos token: {}, Chat template: \n{}",
            bosToken, eosToken, locals["chat_template"].cast<std::string>());
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::cast_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const pybind11::key_error& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (const std::exception& e) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
    } catch (...) {
        SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
        SPDLOG_DEBUG("Chat template loading failed with an unexpected error");
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
    if (std::filesystem::is_directory(outPath) || (std::filesystem::path(outPath).extension() == ".gguf")) {
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is not a directory nor GGUF file ", outPath);
    return StatusCode::LLM_NODE_PATH_DOES_NOT_EXIST_AND_NOT_GGUFFILE;
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
            // Create VLMProcessor for vision encoding and embedding preparation
            try {
                auto vlmProperties = std::static_pointer_cast<VisualLanguageModelServableProperties>(servable->getProperties());
                ov::AnyMap pluginConfig;
                JsonParser::parsePluginConfig(nodeOptions.plugin_config(), pluginConfig);
                vlmProperties->processor = std::make_shared<ov::genai::VLMProcessor>(vlmProperties->modelsPath, vlmProperties->device, pluginConfig);
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Error during VLMProcessor initialization: {}", e.what());
                return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
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
    servable->determineDecodingMethod();
    return StatusCode::OK;
}
}  // namespace ovms
