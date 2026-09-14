load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS", "COPTS_TESTS")

cc_test(
    name = "gemma4_generation_contract_test",
    env_inherit = ["PATH"],
    srcs = ["gemma4_generation_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src:test_platform_utils",
        "//src/llm:generation_config_builders",
        "//third_party:genai",
        "//third_party:openvino",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)

cc_test(
    name = "gemma4_prompt_state_generation_contract_test",
    env_inherit = ["PATH"],
    srcs = ["gemma4_prompt_state_generation_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src/llm:generation_config_builders",
        "//src/llm:io_processing_input_processors",
        "//third_party:genai",
        "//third_party:openvino",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)

cc_test(
    name = "openai_parallel_tool_calls_contract_test",
    env_inherit = ["PATH"],
    srcs = ["openai_parallel_tool_calls_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src:test_platform_utils",
        "//src/llm:openai_completions_api_handler",
        "//src/llm:openai_responses_handler",
        "//third_party:genai",
        "//third_party:openvino",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)
