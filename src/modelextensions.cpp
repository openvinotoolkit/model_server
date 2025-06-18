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
#include "modelextensions.hpp"

namespace ovms {
const std::array<const char*, 2> OV_MODEL_FILES_EXTENSIONS{".xml", ".bin"};
const std::array<const char*, 1> ONNX_MODEL_FILES_EXTENSIONS{".onnx"};
const std::array<const char*, 2> PADDLE_MODEL_FILES_EXTENSIONS{".pdmodel", ".pdiparams"};
const std::array<const char*, 1> TF_MODEL_FILES_EXTENSIONS{".pb"};
const std::array<const char*, 1> TFLITE_MODEL_FILES_EXTENSIONS{".tflite"};
}  // namespace ovms
