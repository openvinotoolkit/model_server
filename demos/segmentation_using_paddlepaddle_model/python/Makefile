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

THIS_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

.PHONY: setup_repository

default: setup_repository

download:
	mkdir workspace
	docker run -u $(id -u):$(id -g) -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} -e no_proxy=${no_proxy} -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu20_dev:latest omz_downloader --name ocrnet-hrnet-w48-paddle --output_dir /models

setup_repository: download
	python3 workspace/models/public/ocrnet-hrnet-w48-paddle/export.py --config workspace/models/public/ocrnet-hrnet-w48-paddle/configs/ocrnet/ocrnet_hrnetw48_cityscapes_1024x512_160k.yml --model_path workspace/models/public/ocrnet-hrnet-w48-paddle/model.pdparams --save_dir workspace/output
	mkdir -p model/1
	mv workspace/output/model.pdiparams model/1/model.pdiparams
	mv workspace/output/model.pdmodel model/1/model.pdmodel

clean:
	@rm -rf workspace