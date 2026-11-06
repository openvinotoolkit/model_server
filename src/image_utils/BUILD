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

load("//:common_settings.bzl", "ovms_cc_library")

cc_binary(
    name = "image_decode_benchmark",
    srcs = ["image_decode_benchmark.cpp"],
    deps = [
        "@stb//:image",
        "//third_party:opencv",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "kserve_image_conversion",
    hdrs = ["kserve_image_conversion.hpp"],
    srcs = ["kserve_image_conversion.cpp", "opencv_decoder_api.hpp"],
    deps = [
        ":image_conversion",
        "@stb//:image",
        "//third_party:openvino",
        "//src:libovmslogging",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libovms_opencv_image_decoder.so",
    srcs = ["opencv_decoder.cpp", "opencv_decoder_api.hpp"],
    deps = [
        "//third_party:openvino",
        "//third_party:opencv",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "ovms_opencv_image_decoder.dll",
    srcs = ["opencv_decoder.cpp", "opencv_decoder_api.hpp"],
    deps = [
        "//third_party:openvino",
        "//third_party:opencv",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "opencv_image_decoder",
    srcs = select({
        "//src:windows": [":ovms_opencv_image_decoder.dll"],
        "//conditions:default": [":libovms_opencv_image_decoder.so"],
    }),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "kserve_image_conversion_test",
    linkstatic = True,
    srcs = ["//src:test/kserve_image_conversion_test.cpp"],
    deps = [
        ":kserve_image_conversion",
        "//src:libovmsstatus",
        "//third_party:openvino",
        "@stb//:image",
        "@com_google_googletest//:gtest_main",
    ],
    data = ["//src:test/images/gray16.png"],
)

cc_binary(
    name = "libno_opencv_audit.so",
    visibility = ["//src:__pkg__"],
    srcs = ["//src:test/no_opencv_audit.c"],
    testonly = True,
    linkshared = True,
    target_compatible_with = select({
        "//src:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)

sh_test(
    name = "no_opencv_runtime_test",
    srcs = ["//src:test/no_opencv_runtime_test.sh"],
    data = [
        ":libno_opencv_audit.so",
        ":kserve_image_conversion_test",
        ":libovms_opencv_image_decoder.so",
        "//src:ovms",
        "//src:test/no_opencv_server_test.py",
        "//src:test/dummy/1/dummy.xml",
        "//src:test/dummy/1/dummy.bin",
        "//src:test/images/rgb.jpg",
    ],
    target_compatible_with = select({
        "//src:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)

ovms_cc_library(
    name = "image_conversion",
    hdrs = [
        "image_conversion.hpp",
    ],
    srcs = [
        "image_conversion.cpp",
    ],
    deps = [
        "@stb//:image",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "//third_party:openvino",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "decoded_image_size",
    hdrs = ["decoded_image_size.hpp"],
    srcs = ["decoded_image_size.cpp"],
    deps = [
        "@stb//:image",
        ":image_conversion",
    ],
    visibility = ["//visibility:public"],
)
