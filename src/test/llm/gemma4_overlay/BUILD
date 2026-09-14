load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS", "COPTS_TESTS")

cc_test(
    name = "gemma4_chat_template_overlay_contract_test",
    srcs = ["gemma4_chat_template_overlay_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src/llm:chat_template_analyzer",
        "//src/llm:io_processing_input_processors",
        "//third_party:genai",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)

cc_test(
    name = "gemma4_google_jinja_contract_test",
    srcs = ["gemma4_google_jinja_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src/llm:chat_template_analyzer",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)
