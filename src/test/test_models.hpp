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
#pragma once

#include <string>

#include "platform_utils.hpp"

const std::string dummy_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/dummy", false);
const std::string dummy_fp64_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/dummy_fp64", false);
const std::string sum_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/add_two_inputs_model", false);
const std::string increment_1x3x4x5_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/increment_1x3x4x5", false);
const std::string passthrough_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/passthrough", false);
const std::string passthrough_string_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/passthrough_string", false);
const std::string dummy_saved_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/dummy_saved_model", false);
const std::string dummy_tflite_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/dummy_tflite", false);
const std::string scalar_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/scalar", false);
const std::string no_name_output_model_location = getGenericFullPathForSrcTest(std::filesystem::current_path().u8string() + "/src/test/no_name_output", false);

constexpr const char* DUMMY_MODEL_INPUT_NAME = "b";
constexpr const char* DUMMY_MODEL_OUTPUT_NAME = "a";
constexpr const int DUMMY_MODEL_INPUT_SIZE = 10;
constexpr const int DUMMY_MODEL_OUTPUT_SIZE = 10;
constexpr const float DUMMY_ADDITION_VALUE = 1.0;
const ovms::signed_shape_t DUMMY_MODEL_SHAPE{1, 10};
// FIXME const ovms::Shape DUMMY_MODEL_SHAPE_META{1, 10};

constexpr const char* DUMMY_FP64_MODEL_INPUT_NAME = "input:0";
constexpr const char* DUMMY_FP64_MODEL_OUTPUT_NAME = "output:0";

constexpr const char* SUM_MODEL_INPUT_NAME_1 = "input1";
constexpr const char* SUM_MODEL_INPUT_NAME_2 = "input2";
constexpr const char* SUM_MODEL_OUTPUT_NAME = "sum";
constexpr const int SUM_MODEL_INPUT_SIZE = 10;
constexpr const int SUM_MODEL_OUTPUT_SIZE = 10;

constexpr const char* INCREMENT_1x3x4x5_MODEL_INPUT_NAME = "input";
constexpr const char* INCREMENT_1x3x4x5_MODEL_OUTPUT_NAME = "output";
constexpr const float INCREMENT_1x3x4x5_ADDITION_VALUE = 1.0;

constexpr const char* PASSTHROUGH_MODEL_INPUT_NAME = "input";
constexpr const char* PASSTHROUGH_MODEL_OUTPUT_NAME = "copy:0";

constexpr const char* PASSTHROUGH_STRING_MODEL_INPUT_NAME = "my_name";
constexpr const char* PASSTHROUGH_STRING_MODEL_OUTPUT_NAME = "my_name";

constexpr const char* SCALAR_MODEL_INPUT_NAME = "model_scalar_input";
constexpr const char* SCALAR_MODEL_OUTPUT_NAME = "model_scalar_output";

const std::string UNUSED_SERVABLE_NAME = "UNUSED_SERVABLE_NAME";
constexpr const ovms::model_version_t UNUSED_MODEL_VERSION = 42;  // Answer to the Ultimate Question of Life
