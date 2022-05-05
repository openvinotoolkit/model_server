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

VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"

.PHONY: setup_repository

default: setup_repository

setup_repository: venv
	@. $(ACTIVATE);
	@pip3 install openvino-dev openvino-dev[tensorflow]
	@omz_downloader --name ssdlite_mobilenet_v2
	@omz_downloader --name efficientnet-b0
	# Additional parameter is required to ensure input precision is FP32
	@omz_converter --name ssdlite_mobilenet_v2 --add_mo_arg=--input=image_tensor{f32}
	@omz_converter --name efficientnet-b0
	@mkdir -p workspace/models/ssdlite_mobilenet_v2/1
	@mkdir -p workspace/models/efficientnet-b0/1
	@cp public/ssdlite_mobilenet_v2/FP32/* workspace/models/ssdlite_mobilenet_v2/1/ -r
	@cp public/efficientnet-b0/FP32/* workspace/models/efficientnet-b0/1/ -r
	# Build custom node
	cd ../../../src/custom_nodes/model_zoo_intel_object_detection && \
	make && \
	mv lib ../../../demos/multi_object_classification_pipeline/python/workspace/lib
	@cp config.json workspace/.

venv:$(ACTIVATE)
	@echo -n "Using venv "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip3 install --upgrade pip
	@touch $(ACTIVATE)

clean:
	@rm -rf $(VIRTUALENV_DIR) workspace public