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
//
// Stand-in for the `llmman` CLI (https://github.com/llmmanorg/llmman) used by
// the OciDownloader tests, so they never touch a container registry. What it
// prints is driven entirely by the environment:
//
//   LLMMAN_MOCK_FAIL=1       `resolve` exits non-zero, as it would for an
//                            unauthorized or nonexistent reference
//   LLMMAN_MOCK_NOISE=1      emit a progress line before the JSON, the way the
//                            real binary writes diagnostics to stderr (which
//                            exec_cmd() merges into the same buffer)
//   LLMMAN_MOCK_OUTPUT=<s>   print <s> verbatim instead of the JSON document
//   LLMMAN_MOCK_PATH=<p>     value of the JSON "path" member
//   LLMMAN_MOCK_FORMAT=<f>   value of the JSON "format" member, default
//                            "safetensors"
#include <cstdlib>
#include <iostream>
#include <string>

static const char* envOrDefault(const char* name, const char* defaultValue) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : defaultValue;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: llmman <command>" << std::endl;
        return 2;
    }

    const std::string command = argv[1];
    if (command == "--version") {
        std::cout << "llmman 0.0.0-mock" << std::endl;
        return 0;
    }
    if (command != "resolve") {
        std::cout << "unknown command: " << command << std::endl;
        return 2;
    }

    const std::string reference = (argc > 2) ? argv[2] : "";
    if (std::getenv("LLMMAN_MOCK_FAIL") != nullptr) {
        std::cout << "Error: failed to pull " << reference << std::endl;
        return 1;
    }
    if (std::getenv("LLMMAN_MOCK_NOISE") != nullptr) {
        std::cout << "[llmman] pulling " << reference << std::endl;
    }

    const char* verbatim = std::getenv("LLMMAN_MOCK_OUTPUT");
    if (verbatim != nullptr) {
        std::cout << verbatim << std::endl;
        return 0;
    }

    std::cout << "{\"reference\":\"" << reference
              << "\",\"path\":\"" << envOrDefault("LLMMAN_MOCK_PATH", "")
              << "\",\"format\":\"" << envOrDefault("LLMMAN_MOCK_FORMAT", "safetensors")
              << "\"}" << std::endl;
    return 0;
}
