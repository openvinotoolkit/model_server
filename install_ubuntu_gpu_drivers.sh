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
"21.48.21782") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-gmmlib_21.3.3_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-core_1.0.9441_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-opencl_1.0.9441_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-opencl-icd_21.48.21782_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-level-zero-gpu_1.2.21782_amd64.deb ; \
        dpkg -i intel*.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"22.10.22597") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/22.10.22597/intel-gmmlib_22.0.2_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.10409/intel-igc-core_1.0.10409_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.10409/intel-igc-opencl_1.0.10409_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/22.10.22597/intel-opencl-icd_22.10.22597_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/22.10.22597/intel-level-zero-gpu_1.3.22597_amd64.deb ; \
        dpkg -i intel*.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"22.35.24055") \
        apt-get update && apt-get install -y --no-install-recommends gpg gpg-agent && \
        curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
        echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        intel-opencl-icd=22.35.24055+i815~u20.04 \
        intel-level-zero-gpu=1.3.24055+i815~u20.04 \
        level-zero=1.8.5+i815~u20.04 && \
        apt-get purge gpg gpg-agent --yes && apt-get --yes autoremove && \
        apt-get clean ; \
        rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* ; \
;; \
"22.43.24595") \
        apt-get update && apt-get install -y --no-install-recommends gpg gpg-agent && \
        curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
        echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        intel-opencl-icd=22.43.24595.35+i538~20.04 \
        intel-level-zero-gpu=1.3.24595.35+i538~20.04 \
        level-zero=1.8.8+i524~u20.04 && \
        apt-get purge gpg gpg-agent --yes && apt-get --yes autoremove && \
        apt-get clean ; \
        rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* ; \
;; \
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
