load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "detection_color_by_id_calculator",
    srcs = ["detection_color_by_id_calculator.cc"],
    deps=[
        ":detection_color_by_id_calculator_cc_proto",
        "@mediapipe//mediapipe/util:render_data_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/formats:location_data_cc_proto",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/util:color_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "detection_color_by_id_calculator_proto",
    srcs = ["detection_color_by_id_calculator.proto"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)