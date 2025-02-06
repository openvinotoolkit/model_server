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

default: client_preprocessing

client_preprocessing:
# Download ONNX ResNet50 model
	curl --fail -L --create-dirs https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o workspace/resnet50-onnx/1/resnet50-caffe2-v1-9.onnx

BASE_OS?=ubuntu

server_preprocessing:
# Download ONNX ResNet50 model
	curl --fail -L --create-dirs https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o workspace/resnet50-onnx/1/resnet50-caffe2-v1-9.onnx
# Build custom node
	cd ../../../src/custom_nodes && \
	make BASE_OS=${BASE_OS} NODES=image_transformation && \
	mkdir -p ../../demos/using_onnx_model/python/workspace/lib && \
	mv lib/${BASE_OS}/libcustom_node_image_transformation.so ../../demos/using_onnx_model/python/workspace/lib/libcustom_node_image_transformation.so
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace
