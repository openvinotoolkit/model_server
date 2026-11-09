#!/bin/bash -x
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
# Script should be used only as a part of Dockerfiles
set -e

declare -a downloaded_debs=()

download_verified_deb() {
    local url checksum filename package
    url=$1
    checksum=$2
    filename=${url##*/}
    package="/tmp/gpu_deps/$filename"
    curl -fL -o "$package" "$url"
    if ! printf '%s  %s\n' "$checksum" "$package" | sha256sum --check --status -; then
        echo "Error: SHA-256 checksum mismatch for $filename." >&2
        exit 1
    fi
    downloaded_debs+=("$package")
}

# Check if INSTALL_DRIVER_VERSION is set
# In case it is not set we reach default switch condition and leave apt in abnormal state
if [ -z "$INSTALL_DRIVER_VERSION" ]; then
    echo "Error: INSTALL_DRIVER_VERSION cannot be empty."
    exit 1
fi

apt-get update && apt-get install -y libnuma1 ocl-icd-libopencl1 --no-install-recommends && rm -rf /var/lib/apt/lists/*
mkdir -p /tmp/gpu_deps

case "$INSTALL_DRIVER_VERSION" in
    "24.52.32224")
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb f653c4f76d532a5a4cc94212a1e4daa6f444c51a5b8b88d217bfd84731931295
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb 9a93bca10c4baafc9601142542784298cb40b2b5dbe62ed50352fc75ab90b9a4
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb dc9a2db05b066bdad2a9208af6f59bc8bb5ffddb1b0fc298d21797c3e034b64a
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb 764fcef49713bd9c25b7b005cf0400dd50fc2dd3142a88ceb7af8fc4cc748c91
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb 2db57b56851bb3cf554611b8807888a0e58779162e757c1f848f64061e5857e0
        dpkg -i "${downloaded_debs[@]}"
        ;;
    "25.48.36300")
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libze-intel-gpu1_25.48.36300.8-0_amd64.deb 23e0987c9a0fedcb4ac1fe1de43954df4e4d827b3f7d0d9801ae21cdb2e0ee17
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-opencl-icd_25.48.36300.8-0_amd64.deb a23952b5cd05cba80b195d7668bedfcc9a51040f3339b88a78c9fdf6c9bb7486
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libigdgmm12_22.8.2_amd64.deb fa66d2debfb5674bccdf26795d8903019aa5ef65a9c179059b72c8e964d69f58
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-ocloc_25.48.36300.8-0_amd64.deb 15a4b977d375581b3bc14ae67f76a8c3b379d1f0f7ba9ff813aeb8d207f3feb6
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-core-2_2.24.8+20344_amd64.deb a701b03553f42111e6790cfd1bf9bdf7e7c2106b2489083892be338b51688ed7
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-opencl-2_2.24.8+20344_amd64.deb 670d3c8d0073b90ef5cc9f10d42b4bc86c08985aa5f7617b81006d7598d543c7
        dpkg -i "${downloaded_debs[@]}"
        ;;
    "26.31.39395")
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-core-2_2.40.13+22418_amd64.deb ebd795e9fddf303a9b24b7f04545d8ddd9ad1f85b3d0cb1166476fab24da6d44
        download_verified_deb https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-opencl-2_2.40.13+22418_amd64.deb 4f990874efc11c3f6091a663b08aef576c4af592dcd8f12e116f8c2fc92d34d9
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-ocloc_26.31.39395.13-0_amd64.deb 12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-opencl-icd_26.31.39395.13-0_amd64.deb 5a9c9e8fdca8a2f9e22754b1a4618c7babf21d7c3ab3503c680005007c7a8c44
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libigdgmm12_22.10.0_amd64.deb 6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54
        download_verified_deb https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libze-intel-gpu1_26.31.39395.13-0_amd64.deb 1722943f81b576b9bb8d61016464208f48ce533dc3bf24ad39605293115cc289
        dpkg -i "${downloaded_debs[@]}"
        ;;
    *)
        dpkg -P intel-gmmlib intel-igc-core intel-igc-opencl intel-level-zero-gpu intel-ocloc intel-opencl intel-opencl-icd && \
        apt-get update && apt-get -y --no-install-recommends install dpkg-dev && rm -rf /var/lib/apt/lists/* && \
        cd "/drivers/${INSTALL_DRIVER_VERSION}" && \
        dpkg-scanpackages . > Packages && \
        cd - && \
        echo "deb [trusted=yes arch=amd64] file:/drivers/${INSTALL_DRIVER_VERSION} ./" > "/etc/apt/sources.list.d/intel-graphics-${INSTALL_DRIVER_VERSION}.list" && \
        apt-get update && \
        apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free libmfx1
        ;;
esac

apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
