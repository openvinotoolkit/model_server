#
# Copyright (c) 2022 Intel Corporation
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

.PHONY: setup_repository

default: setup_repository

BASE_OS?=ubuntu

setup_repository:
# Download face detection model
	mkdir -p workspace/face-detection-retail-0004/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o workspace/face-detection-retail-0004/1/face-detection-retail-0004.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin	-o workspace/face-detection-retail-0004/1/face-detection-retail-0004.bin
# Build face_blur custom node
	cd ../../../src/custom_nodes && \
	make BASE_OS=${BASE_OS} NODES=face_blur && \
	mkdir -p ../../demos/face_blur/python/workspace/lib/ && \
	mv lib/${BASE_OS}/libcustom_node_face_blur.so ../../demos/face_blur/python/workspace/lib/libcustom_node_face_blur.so
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace
