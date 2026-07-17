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

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "port/rapidjson_document.hpp"

namespace ovms {

// Provides lazy-loaded, cached access to JSON files in a model catalog directory.
// Each file is read from disk (or pre-loaded memory) and parsed at most once.
// Pass by const& to all BaseTaskDetector::scan() implementations.
class ModelCatalogContext {
    std::filesystem::path basePath;
    std::string identifier;
    // In-memory file content (e.g. HF-downloaded config). Queried before filesystem.
    std::unordered_map<std::string, std::string> preloadedContent;
    // JSON parse cache. nullptr value = file tried but not available / parse error.
    mutable std::unordered_map<std::string, std::unique_ptr<rapidjson::Document>> jsonCache;

public:
    // For local model paths. modelIdentifier is typically the full path string,
    // used for keyword-based disambiguation.
    ModelCatalogContext(std::filesystem::path modelBasePath, std::string modelIdentifier);

    // Inject pre-downloaded file content (e.g. a config.json fetched from HuggingFace).
    // Must be called before the first json() call for that filename.
    void addContent(std::string filename, std::string content);

    // Model identifier used for keyword-based disambiguation (e.g. "Qwen3-Embedding-0.6B").
    const std::string& modelIdentifier() const;

    // Returns a pointer to the parsed JSON document for the given filename.
    // Loads and parses on first call; returns nullptr if absent or unparsable.
    const rapidjson::Document* json(const std::string& filename) const;
};

// Abstract interface for a per-task detector.
class BaseTaskDetector {
public:
    virtual bool scan(const ModelCatalogContext& ctx) const = 0;
    virtual std::string getName() const = 0;
    virtual ~BaseTaskDetector() = default;
};

class Speech2TextDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

class Text2SpeechDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

class RerankDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

class ImageGenerationDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

class EmbeddingsDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

class TextGenerationDetector final : public BaseTaskDetector {
public:
    bool scan(const ModelCatalogContext& ctx) const override;
    std::string getName() const override;
};

// Iterates registered detectors in priority order and returns the first match.
// Ordering guarantees that specific detectors (e.g. Speech2Text) precede generic
// catch-all detectors (e.g. TextGeneration), making scan() implementations
// mutually exclusive by design without cross-detector coordination.
class DefaultTaskDetector {
    std::vector<std::unique_ptr<BaseTaskDetector>> detectors;

public:
    DefaultTaskDetector();
    // Returns the detected task name, or empty string if no detector matched.
    std::string detect(const ModelCatalogContext& ctx) const;
};

}  // namespace ovms
