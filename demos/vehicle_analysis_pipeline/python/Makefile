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
# Download vehicle detection model
	mkdir -p workspace/vehicle-detection-0202/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o workspace/vehicle-detection-0202/1/vehicle-detection-0202.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin	-o workspace/vehicle-detection-0202/1/vehicle-detection-0202.bin
# Download vehicle attributes recognition model
	mkdir -p workspace/vehicle-attributes-recognition-barrier-0042/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.xml -o workspace/vehicle-attributes-recognition-barrier-0042/1/vehicle-attributes-recognition-barrier-0042.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.bin -o workspace/vehicle-attributes-recognition-barrier-0042/1/vehicle-attributes-recognition-barrier-0042.bin
# Build custom node
	cd ../../../src/custom_nodes/model_zoo_intel_object_detection && \
	make && \
	mv lib ../../../demos/vehicle_analysis_pipeline/python/workspace/lib
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace