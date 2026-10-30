load("//:common_settings.bzl", "LINUX_COMMON_STATIC_LIBS_COPTS")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mpi_openvino_video",
    srcs = ["mpi_openvino_video.cpp"],
    copts = select({
        "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
        "//src:windows": [],
    }),
    deps = [
        "//src/mediapipe_gst_calculators:gst_openvino_inference_calculator_cc_proto",
        "//src/mediapipe_gst_calculators:gst_video_openvino_inference_calculator",
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    linkopts = select({
        "//src:windows": ["d3d11.lib", "dxgi.lib"],
        "//conditions:default": ["-lOpenCL", "-lva", "-lva-drm"],
    }),
)