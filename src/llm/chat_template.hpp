//*****************************************************************************
// Copyright 2024 Intel Corporation
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

// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

namespace ovms {
class TextProcessor;

bool applyChatTemplate(TextProcessor& textProcessor, std::string modelsPath, std::string& requestBody, std::string& output);

}  // namespace ovms
