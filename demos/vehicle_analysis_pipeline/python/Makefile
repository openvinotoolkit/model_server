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

BUILD_CUSTOM_NODE=false
BASE_OS=ubuntu

setup_repository:
# Download vehicle detection model
	mkdir -p workspace/vehicle-detection-0202/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o workspace/vehicle-detection-0202/1/vehicle-detection-0202.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin	-o workspace/vehicle-detection-0202/1/vehicle-detection-0202.bin
# Download vehicle attributes recognition model
	mkdir -p workspace/vehicle-attributes-recognition-barrier-0042/1
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.xml -o workspace/vehicle-attributes-recognition-barrier-0042/1/vehicle-attributes-recognition-barrier-0042.xml
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.bin -o workspace/vehicle-attributes-recognition-barrier-0042/1/vehicle-attributes-recognition-barrier-0042.bin
ifeq ($(BUILD_CUSTOM_NODE),true)
# Build custom node
	cd ../../../src/custom_nodes && \
	make BASE_OS=$(BASE_OS) NODES=model_zoo_intel_object_detection && \
	mkdir -p ../../demos/vehicle_analysis_pipeline/python/workspace/lib && \
	mv lib/$(BASE_OS)/libcustom_node_model_zoo_intel_object_detection.so ../../demos/vehicle_analysis_pipeline/python/workspace/lib/libcustom_node_model_zoo_intel_object_detection.so
endif
# Copy configuration file to workspace directory
	cp config.json workspace/.
ifeq ($(BUILD_CUSTOM_NODE),false)
	sed -i 's:\/workspace\/lib\/libcustom_node_model_zoo_intel_object_detection.so:\/ovms\/lib\/custom_nodes\/libcustom_node_model_zoo_intel_object_detection.so:g' workspace/config.json
endif
clean:
	@rm -rf workspace