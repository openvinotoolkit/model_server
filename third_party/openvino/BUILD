package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "openvino",
    srcs = [
        "bin/intel64/Release/lib/libinference_engine.so",
        "bin/intel64/Release/lib/libinference_engine_legacy.so",
        "bin/intel64/Release/lib/libinference_engine_c_api.so"
    ],
    hdrs = glob([
        "inference-engine/include/**/*.h",
        "inference-engine/include/**/*.hpp"
    ]),
    strip_include_prefix = "inference-engine/include",
)
