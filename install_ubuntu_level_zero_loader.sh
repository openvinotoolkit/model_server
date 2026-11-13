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
        loader_url="https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb"
        ;;
    ubuntu26)
        loader_url="https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~26.04~ppa1_amd64.deb"
        ;;
    *)
        printf 'Unsupported BASE_OS: %s\n' "$BASE_OS" >&2
        exit 1
        ;;
esac

loader_package=$(mktemp /tmp/level-zero-loader.XXXXXX.deb)
trap 'rm -f "$loader_package"' EXIT
curl -fL "$loader_url" -o "$loader_package"
dpkg -i "$loader_package"