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

.PHONY: client_preprocessing, server_preprocessing

default: prepare_repository

THIS_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

venv:
	virtualenv -p python3 .tf_env
	. .tf_env/bin/activate
	pip3 install tensorflow==2.3.1

download_models: venv
	mkdir -p workspace/models/argmax
	docker run -u $(id -u):$(id -g) -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} -e no_proxy=${no_proxy} -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu18_dev:2024.6.0 omz_downloader --name googlenet-v2-tf --output_dir /models
	docker run -u $(id -u):$(id -g) -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} -e no_proxy=${no_proxy} -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu18_dev:2024.6.0 omz_downloader --name resnet-50-tf --output_dir /models
	python3 ../../../tests/models/argmax_sum.py --input_size 1001 --export_dir $(THIS_DIR)/workspace/models/argmax

convert_models: download_models
	docker run -u $(id -u):$(id -g) -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu18_dev:2024.6.0 omz_converter --name googlenet-v2-tf --download_dir /models --output_dir /models --precisions FP32
	docker run -u $(id -u):$(id -g) -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu18_dev:2024.6.0 omz_converter --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32
	docker run -u $(id -u):$(id -g) -v $(THIS_DIR)/workspace/models:/models:rw openvino/ubuntu18_dev:2024.6.0 mo --input input1,input2 --input_shape '[1,1001],[1,1001]' --saved_model_dir /models/argmax/ --output_dir /models/argmax/1

prepare_repository: convert_models
	mkdir -p models/googlenet-v2-tf/1
	mkdir -p models/resnet-50-tf/1
	mkdir -p models/argmax/1
	cp workspace/models/public/resnet-50-tf/FP32/* models/resnet-50-tf/1/
	cp workspace/models/public/googlenet-v2-tf/FP32/* models/googlenet-v2-tf/1/
	cp workspace/models/argmax/1/* models/argmax/1/
	cp config.json models/
	tree models
