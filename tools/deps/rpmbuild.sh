#!/bin/bash
#
# Copyright (c) 2020 Intel Corporation
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

set -e
set -x

yum update -y
yum groupinstall -y "Development Tools"
yum install -y ruby curl wget

wget https://dl.fedoraproject.org/pub/epel/7/SRPMS/Packages/o/opencl-headers-2.2-1.20180306gite986688.el7.src.rpm
wget https://dl.fedoraproject.org/pub/epel/7/SRPMS/Packages/o/ocl-icd-2.2.12-1.el7.src.rpm

mkdir -vp /pkg/{src,bin}
cp *.src.rpm /pkg/src

rpmbuild --rebuild opencl-headers*.src.rpm
rpm -i /root/rpmbuild/RPMS/noarch/opencl-headers-2.2-1*.rpm

rpmbuild --rebuild ocl-icd-*.src.rpm
rpm -i /root/rpmbuild/RPMS/x86_64/ocl-icd-*.rpm

cp -v /root/rpmbuild/RPMS/noarch/opencl-headers-2.2-1*.rpm /pkg/bin/
cp -v /root/rpmbuild/RPMS/x86_64/ocl-icd-2*.rpm /pkg/bin/

tar cvJf /ovms-rpmbuild-deps.tar.xz /pkg
