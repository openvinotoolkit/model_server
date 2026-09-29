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

# Keep this in sync with @mediapipe//WORKSPACE - mediapipe's own glog repos are
# never fetched since its WORKSPACE file is not evaluated by Bazel, so any
# glog repo mediapipe's BUILD files depend on must be mirrored here.
def glog_dependencies():
    # 2020-08-21 SHA 3a0d4d22c5ae0b9a2216988411cfa6bf860cc372
    http_archive(
        name = "com_github_glog_glog",
        strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
        sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
        urls = [
            "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
        ],
    )

    # Non-gflags variant required by mediapipe//third_party:BUILD glog select()
    http_archive(
        name = "com_github_glog_glog_no_gflags",
        strip_prefix = "glog-0.6.0",
        sha256 = "8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6",
        build_file = "@mediapipe//third_party:glog_no_gflags.BUILD",
        urls = [
            "https://github.com/google/glog/archive/v0.6.0.tar.gz",
        ],
        patches = [
            "@mediapipe//third_party:com_github_glog_glog.diff",
        ],
        patch_args = [
            "-p1",
        ],
    )
