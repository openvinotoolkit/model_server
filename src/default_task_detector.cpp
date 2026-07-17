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
#include "default_task_detector.hpp"

#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "port/rapidjson_error.hpp"
#include "port/rapidjson_istreamwrapper.hpp"
#include "stringutils.hpp"

namespace ovms {

// ─── Static sets ─────────────────────────────────────────────────────────────

// Architectures whose task cannot be determined from config.json alone.
// When modules.json is absent, the model identifier is inspected for keywords.
static const std::unordered_set<std::string> ambiguousArchitectures = {
    "Qwen3ForCausalLM",
};

// Architectures that end with "Model" but are NOT embedding models.
// Used to prevent EmbeddingsDetector's generic suffix rule from claiming them.
static const std::unordered_set<std::string> knownNonEmbeddingModelArchitectures = {
    "CLIPTextModel",
    "InternVLChatModel",
    "UNet2DConditionModel",
};

// ─── Static helpers ───────────────────────────────────────────────────────────

// Iterates the architectures array in a config.json document.
// Returns true if pred matches any architecture string; false if field is
// missing, null, or not an array.
template <typename Pred>
static bool anyArchitecture(const rapidjson::Document& doc, Pred&& pred) {
    if (!doc.IsObject() || !doc.HasMember("architectures"))
        return false;
    const auto& archs = doc["architectures"];
    if (!archs.IsArray())
        return false;
    for (const auto& a : archs.GetArray()) {
        if (!a.IsString())
            continue;
        if (pred(std::string(a.GetString())))
            return true;
    }
    return false;
}

// Iterates module entries in a modules.json document (a JSON array).
// Returns true if pred matches any "type" field value.
template <typename Pred>
static bool anyModuleType(const rapidjson::Document& doc, Pred&& pred) {
    if (!doc.IsArray())
        return false;
    for (const auto& mod : doc.GetArray()) {
        if (!mod.IsObject() || !mod.HasMember("type"))
            continue;
        const auto& type = mod["type"];
        if (!type.IsString())
            continue;
        if (pred(std::string(type.GetString())))
            return true;
    }
    return false;
}

// ─── ModelCatalogContext ──────────────────────────────────────────────────────

ModelCatalogContext::ModelCatalogContext(std::filesystem::path modelBasePath, std::string modelIdentifier) :
    basePath(std::move(modelBasePath)),
    identifier(std::move(modelIdentifier)) {}

void ModelCatalogContext::addContent(std::string filename, std::string content) {
    preloadedContent.emplace(std::move(filename), std::move(content));
}

const std::string& ModelCatalogContext::modelIdentifier() const {
    return identifier;
}

const rapidjson::Document* ModelCatalogContext::json(const std::string& filename) const {
    // Return cached result if this filename was already attempted
    auto cacheIt = jsonCache.find(filename);
    if (cacheIt != jsonCache.end()) {
        return cacheIt->second.get();
    }

    std::unique_ptr<rapidjson::Document> doc;

    // Check in-memory preloaded content first (e.g. HF-downloaded files)
    auto contentIt = preloadedContent.find(filename);
    if (contentIt != preloadedContent.end()) {
        doc = std::make_unique<rapidjson::Document>();
        doc->Parse(contentIt->second.c_str());
        if (doc->HasParseError()) {
            doc = nullptr;
        }
    } else if (!basePath.empty()) {
        // Try reading from the model directory on disk
        const auto filePath = basePath / filename;
        std::ifstream file(filePath);
        if (file.is_open()) {
            doc = std::make_unique<rapidjson::Document>();
            rapidjson::IStreamWrapper wrapper(file);
            doc->ParseStream(wrapper);
            if (doc->HasParseError()) {
                doc = nullptr;
            }
        }
    }

    // Cache result — nullptr means "tried and not available"
    auto inserted = jsonCache.emplace(filename, std::move(doc));
    return inserted.first->second.get();
}

// ─── Speech2TextDetector ──────────────────────────────────────────────────────

std::string Speech2TextDetector::getName() const { return "speech2text"; }

bool Speech2TextDetector::scan(const ModelCatalogContext& ctx) const {
    const auto* config = ctx.json("config.json");
    if (!config)
        return false;
    return anyArchitecture(*config, [](const std::string& arch) {
        return arch == "WhisperForConditionalGeneration" ||
               arch == "Qwen3ASRForConditionalGeneration" ||
               startsWith(arch, "SeamlessM4T");
    });
}

// ─── Text2SpeechDetector ─────────────────────────────────────────────────────

std::string Text2SpeechDetector::getName() const { return "text2speech"; }

bool Text2SpeechDetector::scan(const ModelCatalogContext& ctx) const {
    const auto* config = ctx.json("config.json");
    if (!config || !config->IsObject() || !config->HasMember("architectures"))
        return false;

    const auto& archs = (*config)["architectures"];

    // Special case: null architectures with n_mels field (e.g. Kokoro TTS)
    if (archs.IsNull()) {
        return config->HasMember("n_mels");
    }

    if (!archs.IsArray())
        return false;
    for (const auto& a : archs.GetArray()) {
        if (!a.IsString())
            continue;
        const std::string arch = a.GetString();
        if (arch == "ParlerTTSForConditionalGeneration" ||
            arch == "SpeechT5ForTextToSpeech" ||
            endsWith(arch, "ForTextToSpeech"))
            return true;
    }
    return false;
}

// ─── RerankDetector ──────────────────────────────────────────────────────────

std::string RerankDetector::getName() const { return "rerank"; }

bool RerankDetector::scan(const ModelCatalogContext& ctx) const {
    // Layer 1: modules.json (authoritative — sentence-transformers cross-encoder)
    if (const auto* modules = ctx.json("modules.json")) {
        if (anyModuleType(*modules, [](const std::string& type) {
                return type.find("LogitScore") != std::string::npos ||
                       type.find("CrossEncoder") != std::string::npos;
            }))
            return true;
    }

    const auto* config = ctx.json("config.json");
    if (!config)
        return false;

    // Layer 2: architecture suffix (covers ForSequenceClassification, Reward Models)
    if (anyArchitecture(*config, [](const std::string& arch) {
            return endsWith(arch, "ForSequenceClassification");
        }))
        return true;

    // Layer 3: ambiguous architecture + model name keyword fallback.
    // Only applied when modules.json is absent — if modules.json is present it is
    // authoritative and name heuristics must not override it.
    // (TEMPORARY — until export tooling adds modules.json to exported models)
    if (!ctx.json("modules.json")) {
        const std::string normalId = toLower(ctx.modelIdentifier());
        return anyArchitecture(*config, [&normalId](const std::string& arch) {
            return ambiguousArchitectures.count(arch) > 0 &&
                   normalId.find("rerank") != std::string::npos;
        });
    }
    return false;
}

// ─── ImageGenerationDetector ─────────────────────────────────────────────────

std::string ImageGenerationDetector::getName() const { return "image_generation"; }

bool ImageGenerationDetector::scan(const ModelCatalogContext& ctx) const {
    // config.json architectures
    if (const auto* config = ctx.json("config.json")) {
        if (anyArchitecture(*config, [](const std::string& arch) {
                return arch == "CLIPTextModel" ||
                       arch == "UNet2DConditionModel" ||
                       arch == "AutoencoderKL" ||
                       endsWith(arch, "Transformer2DModel");
            }))
            return true;
    }

    // model_index.json _class_name (Diffusers pipeline format)
    if (const auto* idx = ctx.json("model_index.json")) {
        if (idx->IsObject() && idx->HasMember("_class_name") && (*idx)["_class_name"].IsString()) {
            const std::string className = (*idx)["_class_name"].GetString();
            if (className.find("StableDiffusion") != std::string::npos ||
                className.find("Flux") != std::string::npos)
                return true;
        }
    }
    return false;
}

// ─── EmbeddingsDetector ──────────────────────────────────────────────────────

std::string EmbeddingsDetector::getName() const { return "embeddings"; }

bool EmbeddingsDetector::scan(const ModelCatalogContext& ctx) const {
    // Layer 1: modules.json (authoritative — sentence-transformers bi-encoder)
    if (const auto* modules = ctx.json("modules.json")) {
        if (anyModuleType(*modules, [](const std::string& type) {
                return type.find("Pooling") != std::string::npos;
            }))
            return true;
    }

    const auto* config = ctx.json("config.json");
    if (!config)
        return false;

    // Layer 2: known embedding architectures (exact + suffix)
    if (anyArchitecture(*config, [](const std::string& arch) {
            // Exact known embedding model classes
            if (arch == "BertModel" ||
                arch == "JinaBertModel" ||
                arch == "MPNetModel" ||
                arch == "Qwen2Model" ||
                arch == "RobertaModel" ||
                arch == "T5EncoderModel" ||
                arch == "XLMRobertaModel")
                return true;
            // Suffix catch-all for novel embedding models.
            // Exclude image-generation architectures: those ending with Transformer2DModel
            // (e.g. FluxTransformer2DModel) are always image generation, not embeddings.
            // Also exclude architectures explicitly known to belong to other tasks.
            if (knownNonEmbeddingModelArchitectures.count(arch))
                return false;
            if (ambiguousArchitectures.count(arch))
                return false;
            if (endsWith(arch, "Transformer2DModel"))
                return false;
            return endsWith(arch, "EncoderModel") || endsWith(arch, "Model");
        }))
        return true;

    // Layer 3: ambiguous architecture + model name keyword fallback.
    // Only applied when modules.json is absent — if modules.json is present it is
    // authoritative and name heuristics must not override it.
    // (TEMPORARY — until OVMS export tooling adds modules.json to exported models)
    if (!ctx.json("modules.json")) {
        const std::string normalId = toLower(ctx.modelIdentifier());
        return anyArchitecture(*config, [&normalId](const std::string& arch) {
            return ambiguousArchitectures.count(arch) > 0 &&
                   normalId.find("embed") != std::string::npos;
        });
    }
    return false;
}

// ─── TextGenerationDetector ──────────────────────────────────────────────────

std::string TextGenerationDetector::getName() const { return "text_generation"; }

bool TextGenerationDetector::scan(const ModelCatalogContext& ctx) const {
    const auto* config = ctx.json("config.json");
    if (!config)
        return false;
    // Note: if we reach this detector, all higher-priority detectors have
    // already returned false, so ForCausalLM / ForConditionalGeneration
    // architectures are safe to claim here (including Qwen3ForCausalLM when
    // no modules.json and no name keyword matched earlier).
    return anyArchitecture(*config, [](const std::string& arch) {
        return arch == "InternVLChatModel" ||
               endsWith(arch, "ForCausalLM") ||
               endsWith(arch, "ForConditionalGeneration");
    });
}

// ─── DefaultTaskDetector ─────────────────────────────────────────────────────

DefaultTaskDetector::DefaultTaskDetector() {
    // Priority order is critical for correctness:
    // - Speech2Text / Text2Speech before TextGen (share ForConditionalGeneration suffix)
    // - ImageGen before Embeddings (CLIPTextModel / UNet2DConditionModel end with "Model")
    // - Embeddings before TextGen (general "Model" suffix check)
    // - TextGen last (catch-all for ForCausalLM / ForConditionalGeneration)
    detectors.push_back(std::make_unique<Speech2TextDetector>());
    detectors.push_back(std::make_unique<Text2SpeechDetector>());
    detectors.push_back(std::make_unique<RerankDetector>());
    detectors.push_back(std::make_unique<ImageGenerationDetector>());
    detectors.push_back(std::make_unique<EmbeddingsDetector>());
    detectors.push_back(std::make_unique<TextGenerationDetector>());
}

std::string DefaultTaskDetector::detect(const ModelCatalogContext& ctx) const {
    for (const auto& detector : detectors) {
        if (detector->scan(ctx))
            return detector->getName();
    }
    return "";
}

}  // namespace ovms
