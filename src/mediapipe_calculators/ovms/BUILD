#
# Copyright (c) 2023-2026 Intel Corporation
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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ovtensor_tensor_conversion_calculators",
    srcs = ["ovtensor_tensor_conversion_calculators.cc"],
    deps = [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:tensor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ovtensor_tensor_conversion_calculators_test",
    srcs = ["ovtensor_tensor_conversion_calculators_test.cc"],
    deps = [
        ":ovtensor_tensor_conversion_calculators",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/formats:tensor",
    ],
)
