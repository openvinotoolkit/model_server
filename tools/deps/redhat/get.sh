#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
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

mkdir -vp /root/pkg/{src,bin,internal_src}

yum install -y xz pkg-config

cd /rpmbuild
tar -xf ovms-rpmbuild-deps.tar.xz
cp -v pkg/src/* /root/pkg/internal_src/
cp -v pkg/bin/* /root/pkg/bin/

cd /root/pkg/bin/
rpm -i *.rpm
cd ../

ls -lah ./src/
ls -lah ./internal_src/
ls -lah ./bin/

cd ..
tar cvzf rpms.tar.xz ./pkg
