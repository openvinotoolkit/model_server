//****************************************************************************
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

#include <string>

// Small path helpers for the draft-model subdirectory used by the
// speculative-decoding text-generation graph.
//
// This header is intentionally dependency-light: only <string> and (in the
// .cpp) the internal filesystem helper. It exists as a separate translation
// unit / Bazel target (`//src/graph_export:graph_export_paths`) so that
// callers which only need these two utilities — most notably
// `src/pull_module/hf_pull_model_module.cpp` — can depend on the tiny
// `graph_export_paths` target instead of the heavy `graph_export` target.
//
// The `graph_export` target transitively pulls MediaPipe framework headers
// (`@mediapipe//mediapipe/framework:calculator_graph`, etc.) plus the OVMS
// module/server-settings/schema/version graph, so linking it into the HF pull
// module would drag MediaPipe into a code path that is expected to stay on
// the "core `ovms`" side of the runtime-split boundary (see
// python_runtime_separation_architecture.md, "Build and Linkage Changes").
//
// Keep this header free of MediaPipe / server-settings / module includes.
namespace ovms {

std::string getDraftModelDirectoryName(std::string draftModel);
std::string getDraftModelDirectoryPath(const std::string& directoryPath, const std::string& draftModel);

}  // namespace ovms
