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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_OPUS_VERSION = "1.5.2"


def opus_repository():
    http_archive(
        name = "libopus",
        sha256 = "65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1",
        strip_prefix = "opus-" + _OPUS_VERSION,
        urls = ["https://github.com/xiph/opus/releases/download/v" + _OPUS_VERSION + "/opus-" + _OPUS_VERSION + ".tar.gz"],
        build_file = "@ovms//third_party/opus:BUILD",
    )
