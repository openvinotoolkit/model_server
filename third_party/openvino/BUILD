#
# Copyright (c) 2020 Intel Corporation
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
