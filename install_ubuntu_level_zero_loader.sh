#!/bin/bash
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
set -euo pipefail

case "${BASE_OS:?BASE_OS must be set}" in
    ubuntu22)
        loader_url="https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/level-zero_1.24.2+u22.04_amd64.deb"
        ;;
    ubuntu24)
        loader_url="https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20261110T100000Z/pool/main/l/level-zero-loader/libze1_1.34.0-1~24.04~ppa1_amd64.deb"
        ;;
    ubuntu26)
        loader_url="https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20261110T100000Z/pool/main/l/level-zero-loader/libze1_1.34.0-1~26.04~ppa1_amd64.deb"
        ;;
    *)
        printf 'Unsupported BASE_OS: %s\n' "$BASE_OS" >&2
        exit 1
        ;;
esac

loader_package=$(mktemp /tmp/level-zero-loader.XXXXXX.deb)
trap 'rm -f "$loader_package"' EXIT
curl -fL "$loader_url" -o "$loader_package"
case "$BASE_OS" in
    ubuntu22) loader_sha256=7c304e93835d96025c90f6a3d8f2ce5edf142c24da9f4871113a4f0225fef22e ;;
    ubuntu24) loader_sha256=4ea0d892f5e2b82db3c15c9e9a97327b6323189972b75fe4fc8b029a3aa3d26d ;;
    ubuntu26) loader_sha256=d2e3a395ff0b40add337330dcd9e86857963bc173c84a532b5c1cda0dbe7a184 ;;
esac
printf '%s  %s\n' "$loader_sha256" "$loader_package" | sha256sum --check --status -
dpkg -i "$loader_package"