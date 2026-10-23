#
# Copyright (c) 2026 Intel Corporation
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
load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS", "COPTS_TESTS")

cc_test(
    name = "gemma4_parser_contract_test",
    env_inherit = ["GEMMA4_TOKENIZER_PATH", "PATH"],
    srcs = [
        "gemma4_parser_contract_test.cpp",
        "gemma4_reasoning_semantic_refit_test.cpp",
        "gemma4_recovery_contract_test.cpp",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src:test_platform_utils",
        "//src/llm:output_parsers",
        "//src/llm:text_streamer",
        "//third_party:genai",
        "//third_party:openvino",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)
