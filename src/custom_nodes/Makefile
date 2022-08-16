#
# Copyright (c) 2021 Intel Corporation
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

HEADER_FILE_PATH ?= ../custom_node_interface.h
OPENCV_BUILD_FLAGS ?= ../opencv_cmake_flags.txt

BASE_OS ?= ubuntu
OPENCV_BRANCH ?= 4.6.0

NODE_NAME=model_zoo_intel_object_detection

.PHONY: default build

default: docker_build

docker_build:
	@cp $(HEADER_FILE_PATH) ./custom_node_interface.h
	@cp $(OPENCV_BUILD_FLAGS) ./opencv_cmake_flags.txt
	@cp -r ../queue.hpp ./queue.hpp
	@docker build -f Dockerfile.$(BASE_OS) -t custom_node_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --build-arg OPENCV_BRANCH=$(OPENCV_BRANCH) --build-arg NODE_NAME=$(NODE_NAME) .
	@mkdir -p ./lib/$(BASE_OS)
	@docker cp $$(docker create --rm custom_node_build_image:latest):/custom_nodes/lib/libcustom_node_$(NODE_NAME).so ./lib/$(BASE_OS)/
	@rm ./queue.hpp
	@rm opencv_cmake_flags.txt
	@rm custom_node_interface.h
