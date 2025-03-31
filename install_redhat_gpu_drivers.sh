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

case $INSTALL_DRIVER_VERSION in \
"22.28.23726") \
        $DNF_TOOL install --nodocs -y libedit ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-gmmlib-22.1.7-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-core-1.0.11485-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-opencl-1.0.11485-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-opencl-22.28.23726.1-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5-devel/intel-level-zero-gpu-1.3.23453-i392.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5-devel/level-zero-1.8.1-i392.el8.x86_64.rpm ; \
;; \
"22.43.24595") \
        $DNF_TOOL install --nodocs -y libedit ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-gmmlib-22.3.1-i529.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-igc-core-1.0.12504.6-i537.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-igc-opencl-1.0.12504.6-i537.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-opencl-22.43.24595.35-i538.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-level-zero-gpu-1.3.24595.35-i538.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/level-zero-1.8.8-i524.el8.x86_64.rpm ; \
;; \
"23.22.26516") \
        $DNF_TOOL install --nodocs -y libedit ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-gmmlib-22.3.7-i678.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-igc-core-1.0.14062.14-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-igc-opencl-1.0.14062.14-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-opencl-23.22.26516.25-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-level-zero-gpu-1.3.26516.25-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/l/level-zero-1.11.0-674.el8_6.x86_64.rpm ; \
;; \
"24.45.31740") \
        $DNF_TOOL install --nodocs -y libedit ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/i/intel-gmmlib-22.5.3-i1057.el9_4.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/i/intel-igc-core-2.1.14-1057.el9_4.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/i/intel-igc-opencl-2.1.14-1057.el9_4.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/i/intel-opencl-24.45.31740.15-1057.el9_4.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/i/intel-level-zero-gpu-1.6.31740.15-1057.el9_4.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.4/pool/l/level-zero-1.18.5.0-1055.el9_4.x86_64.rpm ; \
;; \
"24.52.32224") \
        $DNF_TOOL install --nodocs -y libedit ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/i/intel-opencl-24.52.32224.14-1077.el9_5.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/i/intel-gmmlib-22.5.5-i1077.el9_5.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/i/intel-igc-core-2.5.12-1077.el9_5.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/i/intel-igc-opencl-2.5.12-1077.el9_5.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/i/intel-level-zero-gpu-1.6.32224.14-1077.el9_5.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/9.5/pool/l/level-zero-1.19.2.0-1077.el9_5.x86_64.rpm ; \
;; \

        *) \
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}." ; \
        exit 1 ; \
esac ; \
rpm -ivh https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.13-4.el9.x86_64.rpm && \
echo "Installed opencl driver version:" ;\
echo `rpm -qa | grep intel-opencl` ; \

