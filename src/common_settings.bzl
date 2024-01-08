#
# Copyright (c) 2024 Intel Corporation
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
COMMON_STATIC_LIBS_COPTS = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-Wno-sign-compare",
    "-fvisibility=hidden", # Needed for pybind targets
    "-Werror",
]

COMMON_STATIC_LIBS_LINKOPTS = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
]

COMMON_FUZZER_COPTS = [
    "-fsanitize=address",
    "-fprofile-generate",
    "-ftest-coverage",
]

COMMON_FUZZER_LINKOPTS = [
    "-fprofile-generate",
    "-fsanitize=address",
    "-fsanitize-coverage=trace-pc",
    "-static-libasan",
]
