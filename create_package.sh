#!/bin/bash
# Copyright (c) 2023 Intel Corporation
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

# This script should be used inside the build image to create a binary package based on the compiled artefacts

env
mkdir -vp /ovms_release/bin
mkdir -vp /ovms_release/lib
mkdir -vp /ovms_release/lib/hddl/config
mkdir -vp /ovms_release/lib/custom_nodes

cp /ovms/src/custom_nodes/tokenizer/build/src/lib*tokenizer.so /ovms_release/lib/custom_nodes/
if [ -f /openvino_contrib/modules/custom_operations/user_ie_extensions/user_ie_extensions/libopenvino_tokenizers.so ]; then cp -v /openvino_contrib/modules/custom_operations/user_ie_extensions/user_ie_extensions/libopenvino_tokenizers.so /ovms_release/lib/ ; fi
if [ -f /openvino_contrib/modules/custom_operations/user_ie_extensions/_deps/fast_tokenizer-src/lib/libcore_tokenizers.so ]; then cp -v /openvino_contrib/modules/custom_operations/user_ie_extensions/_deps/fast_tokenizer-src/lib/libcore_tokenizers.so /ovms_release/lib/ ; fi

find /ovms/bazel-out/k8-*/bin -iname '*.so*' ! -type d ! -name "*params" ! -name "lib_node_*" ! -path "*test_python_binding*" ! -name "*libpython*" -exec cp -v {} /ovms_release/lib/ \;
mv /ovms_release/lib/libcustom_node* /ovms_release/lib/custom_nodes/
cd /ovms_release/lib/ ; rm -f libazurestorage.so.* ; ln -s libazurestorage.so libazurestorage.so.7 ;ln -s libazurestorage.so libazurestorage.so.7.5
cd /ovms_release/lib/ ; rm -f libcpprest.so.2.10 ; ln -s libcpprest.so libcpprest.so.2.10
rm -f /ovms_release/lib/libssl.so
rm -f /ovms_release/lib/libsampleloader*

# Remove coverage libaries
if [ -f /ovms_release/lib/libjava.so ] ; then cd /ovms_release/lib/ && \
    rm -rf  libatk-wrapper.so libattach.so libawt_headless.so libawt.so libawt_xawt.so libdt_socket.so libfreetype.so \
	libextnet.so libfontmanager.so libinstrument.so libj2gss.so libj2pcsc.so libj2pkcs11.so libjaas.so \
	libjavajpeg.so libjava.so libjawt.so libjdwp.so libjimage.so libjli.so libjsig.so libjsound.so libjvm.so \
	liblcms.so libmanagement_agent.so libmanagement_ext.so libmanagement.so libmlib_image.so libnet.so libnio.so \
	libprefs.so librmi.so libsaproc.so libsctp.so libsplashscreen.so libsunec.so libsystemconf.so libunpack.so libverify.so libzip.so ; \
fi

# Remove capi temp libraries
if [ -f /ovms_release/lib/libsrc_Slibovms_Ushared.so ] ; then \
    rm -rf  /ovms_release/lib/libsrc_Slibovms_Ushared.so \
	/ovms_release/lib/libprediction_service_proto.so-2.params \
	/ovms_release/lib/libovms_shared.so-2.params ; \
fi

if ! [[ $debug_bazel_flags == *"PYTHON_DISABLE=1"* ]]; then cp -r /opt/intel/openvino/python /ovms_release/lib/python ; fi
if ! [[ $debug_bazel_flags == *"PYTHON_DISABLE=1"* ]]; then mv /ovms_release/lib/pyovms.so /ovms_release/lib/python ; fi
if [ -f /opt/intel/openvino/runtime/lib/intel64/plugins.xml ]; then cp /opt/intel/openvino/runtime/lib/intel64/plugins.xml /ovms_release/lib/ ; fi
find /opt/intel/openvino/runtime/lib/intel64/ -iname '*.mvcmd*' -exec cp -v {} /ovms_release/lib/ \;
if [ -d /opt/intel/openvino/runtime/3rdparty ] ; then find /opt/intel/openvino/runtime/3rdparty/ -iname '*libtbb.so.*' -exec cp -vP {} /ovms_release/lib/ \;; fi
find /opt/opencv/lib/ -iname '*.so*' -exec cp -vP {} /ovms_release/lib/ \;
cp /opt/opencv/share/licenses/opencv4/* /ovms/release_files/thirdparty-licenses/
if [ "$ov_use_binary" == "1" ] ; then cp /opt/intel/openvino/docs/licensing/EULA.txt /ovms/release_files/thirdparty-licenses/openvino.LICENSE.txt; fi
if [ "$ov_use_binary" == "0" ] ; then cp /openvino/LICENSE /ovms/release_files/thirdparty-licenses/openvino.LICENSE.txt; fi
if [ "$BASE_OS" == "redhat" ] ; then cp -P /usr/lib64/libpugixml.so* /ovms_release/lib/ ; fi

if [ "$FUZZER_BUILD" == "0" ]; then find /ovms/bazel-bin/src -name 'ovms' -type f -exec cp -v {} /ovms_release/bin \; ; fi;
cd /ovms_release/bin
if [ "$FUZZER_BUILD" == "0" ]; then patchelf --remove-rpath ./ovms && patchelf --set-rpath '$ORIGIN/../lib/' ./ovms; fi;
find /ovms_release/lib/ -iname '*.so*' -exec patchelf --debug --remove-rpath  {}  \;
find /ovms_release/lib/ -iname '*.so*' -exec patchelf --debug --set-rpath '$ORIGIN/../lib' {} \;

find /opt/intel/openvino/runtime/lib/intel64/ -iname '*.so*' -exec cp -vP {} /ovms_release/lib/ \;

cd /ovms
cp -v /ovms/release_files/LICENSE /ovms_release/
cp -v /ovms/release_files/metadata.json /ovms_release/
cp -rv /ovms/release_files/thirdparty-licenses /ovms_release/
mkdir -vp /ovms_release/include && cp /ovms/src/ovms.h /ovms_release/include
ls -lahR /ovms_release/


mkdir -p /ovms_pkg/${BASE_OS}
cd /ovms_pkg/${BASE_OS}
tar czf ovms.tar.gz --transform 's/ovms_release/ovms/' /ovms_release/
sha256sum ovms.tar.gz > ovms.tar.gz.sha256 && \
tar cJf ovms.tar.xz --transform 's/ovms_release/ovms/' /ovms_release/
sha256sum ovms.tar.xz > ovms.tar.xz.sha256
cd /ovms_release
ls -l

