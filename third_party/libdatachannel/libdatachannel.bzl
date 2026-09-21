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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_LIBDATACHANNEL_TAG = "v0.24.5"  # https://github.com/paullouisageneau/libdatachannel/releases/tag/v0.24.5

def libdatachannel_repository():
    # Submodules provide usrsctp (data channels), libjuice (ICE) and libsrtp (media transport).
    new_git_repository(
        name = "libdatachannel",
        remote = "https://github.com/paullouisageneau/libdatachannel",
        tag = _LIBDATACHANNEL_TAG,
        init_submodules = True,
        recursive_init_submodules = True,
        build_file = "@ovms//third_party/libdatachannel:BUILD",
    )
