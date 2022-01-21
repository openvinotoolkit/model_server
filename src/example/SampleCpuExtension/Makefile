#
# Copyright (c) 2020-2021 Intel Corporation
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

all: $(eval SHELL:=/bin/bash)
	/usr/bin/g++ --version
	. /opt/intel/openvino/setupvars.sh && /usr/bin/g++ -std=gnu++11 -fPIC -shared ov_extension.cpp CustomReluOp.cpp \
		-I /opt/intel/openvino/runtime/include/ie \
		-I /opt/intel/openvino/runtime/include \
		-o libcustom_relu_cpu_extension.so
