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

# Check if INSTALL_DRIVER_VERSION is set
# In case it is not set we reach default switch condition and leave apt in abnormal state
if [ -z "$INSTALL_DRIVER_VERSION" ]; then
    echo "Error: INSTALL_DRIVER_VERSION cannot be empty."
    exit 1
fi

apt-get update && apt-get install -y libnuma1 ocl-icd-libopencl1 --no-install-recommends && rm -rf /var/lib/apt/lists/* && \
case $INSTALL_DRIVER_VERSION in \
"23.13.26032") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.05.25593.11/libigdgmm12_22.3.0_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13700.14/intel-igc-core_1.0.13700.14_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13700.14/intel-igc-opencl_1.0.13700.14_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.13.26032.30/intel-opencl-icd_23.13.26032.30_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.13.26032.30/libigdgmm12_22.3.0_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"23.22.26516") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.22.26516.18/intel-level-zero-gpu_1.3.26516.18_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14062.11/intel-igc-core_1.0.14062.11_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14062.11/intel-igc-opencl_1.0.14062.11_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.22.26516.18/intel-opencl-icd_23.22.26516.18_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/23.22.26516.18/libigdgmm12_22.3.0_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"24.26.30049") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-level-zero-gpu_1.3.30049.6_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-opencl-icd_24.26.30049.6_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/libigdgmm12_22.3.20_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-core_1.0.17193.4_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-opencl_1.0.17193.4_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"24.39.31294") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-level-zero-gpu_1.6.31294.12_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-opencl-icd_24.39.31294.12_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/libigdgmm12_22.5.2_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-core_1.0.17791.9_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-opencl_1.0.17791.9_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
*) \
        dpkg -P intel-gmmlib intel-igc-core intel-igc-opencl intel-level-zero-gpu intel-ocloc intel-opencl intel-opencl-icd && \
        apt-get update && apt-get -y --no-install-recommends install dpkg-dev && rm -rf /var/lib/apt/lists/* && \
        cd /drivers/${INSTALL_DRIVER_VERSION} && \
            dpkg-scanpackages .  > Packages && \
            cd - ; \
        echo "deb [trusted=yes arch=amd64] file:/drivers/${INSTALL_DRIVER_VERSION} ./" > /etc/apt/sources.list.d/intel-graphics-${INSTALL_DRIVER_VERSION}.list ; \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            intel-opencl-icd \
            intel-level-zero-gpu level-zero \
            intel-media-va-driver-non-free libmfx1 && \
            rm -rf /var/lib/apt/lists/* ; \
esac
apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
