#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("//:common_settings.bzl", "ovms_cc_library", "PYBIND_DEPS", "COPTS_PYTHON")

ovms_cc_library(
    name = "llm_engine",
    srcs = [],
    deps =  ["@llm_engine//:llm_engine"],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

ovms_cc_library(
    name = "llmcalculator",
    srcs = ["http_llm_calculator.cc"],
    deps = [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/kfserving_api:kfserving_api_cpp",
        "//src:libovmsprofiler",
        ":genai_servables",
        "//src:httppayload"
    ]+ select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON,
    alwayslink = 1, # needed, so the calculator can be registered by MediaPipe
)

mediapipe_proto_library(
    name = "llmcalculator_proto", # llmcalculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["llm_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

ovms_cc_library(
    name = "openai_completions_api_handler",
    hdrs = ["apis/openai_completions.hpp", "apis/openai_json_response.hpp"],
    srcs = ["apis/openai_completions.cpp"],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "@mediapipe//mediapipe/framework:calculator_framework", # required for absl status
        "//src:libovmsprofiler",
        "//third_party:curl",
        "//src:image_conversion",
        "//src:libovmsfilesystem",
        "@stb//:image",
        ":openai_request",
        ":output_parsers",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

ovms_cc_library(
    name = "openai_request",
    hdrs = ["apis/openai_request.hpp"],
    srcs = [],
    deps = [
        "//src/port:rapidjson_document",
        ":apis_tool_schema_wrapper",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

ovms_cc_library(
    name = "apis_tool_schema_wrapper",
    hdrs = ["apis/tool_schema_wrapper.hpp"],
    deps = [
        "//src/port:rapidjson_document",
    ],
    visibility = ["//visibility:public"],
)
ovms_cc_library(
    name = "partial_json_builder",
    hdrs = ["io_processing/partial_json_builder.hpp"],
    srcs = ["io_processing/partial_json_builder.cpp"],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "io_processing_utils",
    hdrs = ["io_processing/utils.hpp"],
    srcs = ["io_processing/utils.cpp"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "io_processing_base_output_parser",
    hdrs = ["io_processing/base_output_parser.hpp"],
    srcs = ["io_processing/base_output_parser.cpp"],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        ":io_processing_utils",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
)
ovms_cc_library(
    name = "io_processing_qwen3coder_tool_parser",
    hdrs = ["io_processing/qwen3coder/qwen3coder_tool_parser.hpp"],
    srcs = ["io_processing/qwen3coder/qwen3coder_tool_parser.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src/utils:rapidjson_utils",
        ":io_processing_utils",
        ":io_processing_base_output_parser",
        ":apis_tool_schema_wrapper",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
)
ovms_cc_library( # TODO split further so we don't have to recompile everything when changing one parser ...
    name = "output_parsers",
    hdrs = [
            "io_processing/hermes3/tool_parser.hpp",
            "io_processing/llama3/tool_parser.hpp",
            "io_processing/phi4/tool_parser.hpp",
            "io_processing/mistral/tool_parser.hpp",
            "io_processing/qwen3/reasoning_parser.hpp",
            "io_processing/gptoss/reasoning_parser.hpp",
            "io_processing/gptoss/tool_parser.hpp",
            "io_processing/gptoss/harmony.hpp",
            "io_processing/output_parser.hpp",
    ],
    srcs = [
            "io_processing/hermes3/tool_parser.cpp",
            "io_processing/llama3/tool_parser.cpp",
            "io_processing/phi4/tool_parser.cpp",
            "io_processing/mistral/tool_parser.cpp",
            "io_processing/qwen3/reasoning_parser.cpp",
            "io_processing/gptoss/reasoning_parser.cpp",
            "io_processing/gptoss/tool_parser.cpp",
            "io_processing/gptoss/harmony.cpp",
            "io_processing/output_parser.cpp",
    ],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src:libovmslogging",
        "//src:libovmsstring_utils",
        ":partial_json_builder",
        ":io_processing_base_output_parser",
        ":io_processing_qwen3coder_tool_parser",
        ":io_processing_utils",
        ":apis_tool_schema_wrapper",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "generation_config_builders",
    hdrs = ["io_processing/base_generation_config_builder.hpp",
            "io_processing/phi4/generation_config_builder.hpp",
            "io_processing/llama3/generation_config_builder.hpp",
            "io_processing/hermes3/generation_config_builder.hpp",
            "io_processing/generation_config_builder.hpp"],
    srcs = ["io_processing/base_generation_config_builder.cpp",
            "io_processing/phi4/generation_config_builder.cpp",
            "io_processing/llama3/generation_config_builder.cpp",
            "io_processing/hermes3/generation_config_builder.cpp"],
    deps = [
        ":openai_request",
        "//src:libovmslogging",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

ovms_cc_library(
    name = "genai_servables",
    hdrs = ["servable.hpp", 
            "servable_initializer.hpp", 
            "language_model/continuous_batching/servable.hpp",
            "language_model/continuous_batching/llm_executor.hpp",
            "language_model/continuous_batching/servable_initializer.hpp",
            "visual_language_model/continuous_batching/servable.hpp",
            "language_model/legacy/servable.hpp",
            "language_model/legacy/servable_initializer.hpp",
            "language_model/legacy/legacy_executor.hpp",
            "visual_language_model/legacy/servable.hpp",
            "visual_language_model/legacy/servable_initializer.hpp",
            "visual_language_model/legacy/legacy_executor.hpp",
            "text_utils.hpp"],
    srcs = ["servable.cpp",
            "servable_initializer.cpp",
            "language_model/continuous_batching/servable.cpp",
            "language_model/continuous_batching/servable_initializer.cpp",
            "visual_language_model/continuous_batching/servable.cpp",
            "language_model/legacy/servable.cpp",
            "language_model/legacy/servable_initializer.cpp",
            "language_model/legacy/legacy_executor.cpp",
            "visual_language_model/legacy/servable.cpp",
            "visual_language_model/legacy/servable_initializer.cpp",
            "visual_language_model/legacy/legacy_executor.cpp"],
    deps = [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src:libmodelconfigjsonparser",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovmsmediapipe_utils",
        "//src:libovmsprofiler",
        "//src:libovmsstring_utils",
        "//src:libovmsfilesystem",
        "llmcalculator_cc_proto",
        ":openai_completions_api_handler",
        ":generation_config_builders",
        "//src:httppayload",
        "//src:libhttpclientconnection",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }) + select({
        "//:disable_python": [],
        "//:not_disable_python" : [":py_jinja_template_processor"],
    }),
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

ovms_cc_library(
    name = "py_jinja_template_processor",
    hdrs = ["py_jinja_template_processor.hpp"],
    srcs = ["py_jinja_template_processor.cpp"],
    deps = ["@mediapipe//mediapipe/framework:calculator_framework",
            "//third_party:openvino",
            "//src:libovmslogging",
            "//src/python:utils",
    ] + PYBIND_DEPS,
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON
)

