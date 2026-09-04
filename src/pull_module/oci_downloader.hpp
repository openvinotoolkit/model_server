#pragma once
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
#include <optional>
#include <string>

#include "model_downloader.hpp"
#include "../capi_frontend/server_settings.hpp"

namespace ovms {
class Status;

// Downloads a CNCF ModelPack (https://github.com/modelpack/model-spec) image
// by delegating to the `llmman` CLI (https://github.com/llmmanorg/llmman).
//
// llmman already implements the whole OCI side of this: registry auth, the
// ModelPack media types, resumable blob download, a content-addressed local
// store and extraction. `llmman resolve <reference>` pulls the image if it is
// not present locally and prints a single line of JSON describing where the
// model ended up:
//
//     {"reference":"ghcr.io/org/model:tag","path":"/abs/path","format":"safetensors"}
//
// This class turns that into the two things the rest of the pull flow needs:
// the models_path to write into graph.pbtxt (getModelPath()) and, for GGUF
// payloads, the file name to append to it (getGgufFilename()).
class OciDownloader : public IModelDownloader {
public:
    OciDownloader(const ExportSettings& exportSettings, const GraphExportType& task, const std::string& inSourceModel,
        const std::string& inDownloadPath, bool inOverwrite,
        const std::string& llmmanBinary = "");
    Status downloadModel() override;

    // Only valid after downloadModel() returned OK.
    // Absolute path to write into graph.pbtxt as models_path, or "./" when the
    // model was converted into the graph directory itself.
    const std::string& getModelPath() const { return this->modelPath; }
    const std::optional<std::string>& getGgufFilename() const { return this->ggufFilename; }

    // Name of the llmman executable to invoke: $LLMMAN_BIN when set, "llmman"
    // otherwise. Resolved through PATH by exec_cmd().
    static std::string resolveLlmmanBinary();

protected:
    ExportSettings exportSettings;
    const GraphExportType task;
    const std::string llmmanBinary;
    std::string modelPath;
    std::optional<std::string> ggufFilename;

    std::string getVersionCmd() const;
    std::string getResolveCmd() const;
    Status checkLlmmanIsPresent();
    // Extracts "path" and "format" from llmman's stdout. Diagnostics that
    // llmman writes to stderr are interleaved into the same buffer by
    // exec_cmd(), so the last line that parses as a JSON object wins.
    static Status parseResolveOutput(const std::string& output, std::string& outPath, std::string& outFormat);
    // True when the directory already holds an OpenVINO IR model, i.e. it can
    // be served without an optimum-cli conversion.
    static bool containsOpenVinoIr(const std::string& directory);
    // Converts a HuggingFace-format checkout that llmman resolved into
    // OpenVINO IR inside the graph directory, reusing OptimumDownloader.
    Status convertToOpenVinoIr(const std::string& resolvedPath);
};
}  // namespace ovms
