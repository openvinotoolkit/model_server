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

base_os=${BASE_OS:?BASE_OS must be set}
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT

read_pin() {
    awk -v key="${1}_${base_os^^}" '$1 == key && $2 == "?=" { print $3; exit }' \
        /usr/local/share/ovms-versions.mk
}

download_verified() {
    local url checksum
    url=$(read_pin "${1}_URL")
    checksum=$(read_pin "${1}_SHA256")
    if [[ -z $url || ! $checksum =~ ^[a-f0-9]{64}$ ]]; then
        printf 'Missing URL or invalid SHA-256 for %s on %s\n' "$1" "$base_os" >&2
        exit 1
    fi
    curl --retry 5 --retry-delay 2 -fL "$url" -o "$2"
    printf '%s  %s\n' "$checksum" "$2" | sha256sum --check -
}

download_verified LEVEL_ZERO_LOADER "$work_dir/loader.deb"
if [[ ${NPU:-0} == 1 ]]; then
    download_verified NPU_DRIVER "$work_dir/npu.tar.gz"
    mkdir "$work_dir/npu"
    tar -xf "$work_dir/npu.tar.gz" -C "$work_dir/npu"
    apt-get update
    apt-get install -y --no-install-recommends libtbb12
    dpkg -i "$work_dir/loader.deb" "$work_dir/npu/"*.deb
    apt-get clean
    rm -rf /var/lib/apt/lists/*
else
    dpkg -i "$work_dir/loader.deb"
fi