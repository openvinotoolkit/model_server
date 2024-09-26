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
case $INSTALL_DRIVER_VERSION in \
"21.38.21026") \
        mkdir /tmp/gpu_deps ; \
        curl -L --output /tmp/gpu_deps/intel-igc-core-1.0.8708-1.el8.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02870435-intel-igc/intel-igc-core-1.0.8708-1.el8.x86_64.rpm ; \
        curl -L --output /tmp/gpu_deps/intel-opencl-21.38.21026-1.el8.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02871549-intel-opencl/intel-opencl-21.38.21026-1.el8.x86_64.rpm ; \
        curl -L --output /tmp/gpu_deps/intel-igc-opencl-devel-1.0.8708-1.el8.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02870435-intel-igc/intel-igc-opencl-devel-1.0.8708-1.el8.x86_64.rpm ; \
        curl -L --output /tmp/gpu_deps/intel-igc-opencl-1.0.8708-1.el8.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02870435-intel-igc/intel-igc-opencl-1.0.8708-1.el8.x86_64.rpm ; \
        curl -L --output /tmp/gpu_deps/intel-gmmlib-21.2.1-1.el7.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02320646-intel-gmmlib/intel-gmmlib-21.2.1-1.el8.x86_64.rpm ; \
        curl -L --output /tmp/gpu_deps/intel-gmmlib-devel-21.2.1-1.el8.x86_64.rpm https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/centos-stream-8-x86_64/02320646-intel-gmmlib/intel-gmmlib-devel-21.2.1-1.el8.x86_64.rpm ; \
        cd /tmp/gpu_deps && rpm -iv *.rpm && rm -Rf /tmp/gpu_deps ; \
;; \
"22.10.22597") \
        $DNF_TOOL install -y libedit ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-gmmlib-22.0.3-i699.3.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-core-1.0.10409-i699.3.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/level-zero-1.7.9-i699.3.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-opencl-1.0.10409-i699.3.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-ocloc-22.10.22597-i699.3.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-opencl-22.10.22597-i699.3.el8.x86_64.rpm ; \
;; \
"22.28.23726") \
        $DNF_TOOL install -y libedit ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-gmmlib-22.1.7-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-core-1.0.11485-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-igc-opencl-1.0.11485-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5/intel-opencl-22.28.23726.1-i419.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5-devel/intel-level-zero-gpu-1.3.23453-i392.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.5-devel/level-zero-1.8.1-i392.el8.x86_64.rpm ; \
;; \
"22.43.24595") \
        $DNF_TOOL install -y libedit ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-gmmlib-22.3.1-i529.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-igc-core-1.0.12504.6-i537.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-igc-opencl-1.0.12504.6-i537.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-opencl-22.43.24595.35-i538.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/intel-level-zero-gpu-1.3.24595.35-i538.el8.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/graphics/rhel/8.6/level-zero-1.8.8-i524.el8.x86_64.rpm ; \
;; \
"23.22.26516") \
        $DNF_TOOL install -y libedit ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-gmmlib-22.3.7-i678.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-igc-core-1.0.14062.14-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-igc-opencl-1.0.14062.14-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-opencl-23.22.26516.25-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/i/intel-level-zero-gpu-1.3.26516.25-682.el8_6.x86_64.rpm ; \
        rpm -ivh https://repositories.intel.com/gpu/rhel/8.6/pool/l/level-zero-1.11.0-674.el8_6.x86_64.rpm ; \
;; \

        *) \
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}." ; \
        exit 1 ; \
esac ; \
rpm -ivh http://vault.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm && \
echo "Installed opencl driver version:" ;\
echo `rpm -qa | grep intel-opencl` ; \

