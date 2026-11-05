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
# Check if INSTALL_DRIVER_VERSION is set
if [ -z "$INSTALL_DRIVER_VERSION" ]; then
    echo "Error: INSTALL_DRIVER_VERSION cannot be empty."
    exit 1
fi
if [ -z "$DNF_TOOL" ]; then
    echo "Error: DNF_TOOL cannot be empty."
    exit 1
fi

declare -a downloaded_rpms=()

download_verified_rpm() {
    local url checksum filename package
    url=$1
    checksum=$2
    filename=${url##*/}
    package="/tmp/gpu_deps/$filename"
    wget -q -O "$package" "$url"
    if ! printf '%s  %s\n' "$checksum" "$package" | sha256sum --check --status -; then
        echo "Error: SHA-256 checksum mismatch for $filename." >&2
        exit 1
    fi
    downloaded_rpms+=("$package")
}

mkdir -p /tmp/gpu_deps

case $INSTALL_DRIVER_VERSION in \
"25.18.33578") \
        $DNF_TOOL install --nodocs -y libedit libnl3; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/i/intel-gmmlib-22.7.2-i1146.el9_7.x86_64.rpm a865b962da64cc1b651931472a381f9b29b401fab941d2a117a26d2386e9f8a8 ; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/i/intel-igc-core-2.11.43-1146.el9_7.x86_64.rpm 571ee49824c080fd38b2e23317de205473a0713fe88e573eba12490f1dc68138 ; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/i/intel-igc-opencl-2.11.43-1146.el9_7.x86_64.rpm ccf42c97a7dc2839179531b0901b2449e0ba95b5dd379432e16185c41e455373 ; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/i/intel-opencl-25.18.33578.77-1146.el9_7.x86_64.rpm 376ff448aba69dd86998873fb44e1ba3f29806fa070994bb0c2888c7758c0136 ; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/i/intel-level-zero-gpu-1.6.33578.77-1146.el9_7.x86_64.rpm 1f9a540640dffdb805a6384ea7f44896f1f1545495eec269d60451384aafcdde ; \
        download_verified_rpm https://repositories.intel.com/gpu/rhel/9.7/pool/l/level-zero-1.24.0.0-1146.el9_7.x86_64.rpm 8a540c80f8f70a3c40b9c79c1206644187dc18366942a6537be6b655be93f072 ; \
;; \
        *) \
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}." ; \
        exit 1 ; \
esac ; \
download_verified_rpm https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.13-4.el9.x86_64.rpm 0a589ca7000927ee350d1d54493ec159efc80a4fa957eecf37650c65991ee520 && \
rpm -ivh "${downloaded_rpms[@]}" && rm -rf /tmp/gpu_deps && \
echo "Installed opencl driver version:" ;\
echo `rpm -qa | grep intel-opencl` ; \

