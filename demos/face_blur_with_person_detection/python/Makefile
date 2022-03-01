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

setup_repository:
# Download face detection model
	mkdir -p workspace/face-detection-retail-0004/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o workspace/face-detection-retail-0004/1/face-detection-retail-0004.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin	-o workspace/face-detection-retail-0004/1/face-detection-retail-0004.bin
# Download person detection model
	mkdir -p workspace/person-detection-retail-0013/1
	curl https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -o workspace/person-detection-retail-0013/1/person-detection-retail-0013.xml
	curl https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/person-detection-retail-0013/FP32/person-detection-retail-0013.bin -o workspace/person-detection-retail-0013/1/person-detection-retail-0013.bin
# Build face_blur custom node
	cd ../../../src/custom_nodes/face_blur && \
	make && \
	mv lib ../../../demos/face_blur/python/workspace/lib
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace
