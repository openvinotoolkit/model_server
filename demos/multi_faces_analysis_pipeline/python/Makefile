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
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o workspace/face-detection-retail-0004/1/face-detection-retail-0004.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin	-o workspace/face-detection-retail-0004/1/face-detection-retail-0004.bin
# Download age gender recognition model
	mkdir -p workspace/age-gender-recognition-retail-0013/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -o workspace/age-gender-recognition-retail-0013/1/age-gender-recognition-retail-0013.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin -o workspace/age-gender-recognition-retail-0013/1/age-gender-recognition-retail-0013.bin
# Download emotion recognition model
	mkdir -p workspace/emotion-recognition-retail-0003/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -o workspace/emotion-recognition-retail-0003/1/emotions-recognition-retail-0003.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -o workspace/emotion-recognition-retail-0003/1/emotions-recognition-retail-0003.bin
# Build custom node
	cd ../../../src/custom_nodes/model_zoo_intel_object_detection && \
	make && \
	mv lib ../../../demos/multi_face_analysis_pipeline/python/workspace/lib
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace