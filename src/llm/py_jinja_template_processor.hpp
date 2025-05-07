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
#include <memory>
#include <sstream>
#include <string>

#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

#include "src/python/utils.hpp"

namespace ovms {

class PyJinjaTemplateProcessor {
public:
    std::string bosToken = "";
    std::string eosToken = "";
    std::unique_ptr<PyObjectWrapper<py::object>> chatTemplate = nullptr;

    static bool applyChatTemplate(PyJinjaTemplateProcessor& templateProcessor, std::string modelsPath, const std::string& requestBody, std::string& output);
};
}  // namespace ovms
