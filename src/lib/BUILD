load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS")

cc_binary(
    name = "ovms_llm_calculators",
    srcs = ["//src/llm:llm_calculators_plugin_src"],
    deps = [
        "//src/llm:llm_calculators_plugin_api",
    ],
    linkopts = COMMON_STATIC_LIBS_LINKOPTS + select({
        "//src:windows": ["/FORCE:UNRESOLVED"],
        "//conditions:default": [
            "-Wl,--allow-shlib-undefined",
            "-Wl,-rpath,/ovms/lib:/opt/intel/openvino/runtime/lib/intel64",
        ],
    }),
    linkshared = True,
    local_defines = COMMON_LOCAL_DEFINES,
    visibility = ["//visibility:public"],
)