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

BASE_URL="http://vault.centos.org/centos/7.8.2003/os/Source/SPackages"

mkdir -vp /root/pkg/{src,bin,internal_src}

cd /rpmbuild
tar -xf ovms-rpmbuild-deps.tar.xz
cp -v pkg/src/* /root/pkg/internal_src/
cp -v pkg/bin/* /root/pkg/bin/

cd /root

LIBS=" libusb libusbx numactl-libs boost-thread boost-system boost-filesystem boost-program-options boost-chrono boost-date-time boost-atomic "

yum install -y --downloadonly --downloaddir=./pkg/bin/ $LIBS
cd ./pkg/bin/
rpm -i *.rpm
cd ../
for lib in $LIBS ; do
	if echo "$lib" | grep -q "boost" ; then
		echo "Skipping boost :: $lib"
		continue
	fi
        VER=$(rpm -qa $lib | sed 's/x86_64/src.rpm/')
	if [ "$lib" == "numactl-libs" ] ; then
		echo "Fixing numactl-libs $VER "
		VER=$(echo "$VER" | sed 's/numactl-libs/numactl/' )
		echo "Fixed as: $VER"
	fi
        curl -o "./src/$VER" --fail -L $BASE_URL/$VER
done
BOOST_VER=$(rpm -qa boost-system | sed 's/x86_64/src.rpm/' | sed 's/boost-system/boost/')
curl -o "./internal_src/$BOOST_VER" --fail -L $BASE_URL/$BOOST_VER

ls -lah ./src/
ls -lah ./internal_src/
ls -lah ./bin/

cd ..
tar cvzf rpms.tar.xz ./pkg
