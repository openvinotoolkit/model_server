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
OPENCV_BUILD_FLAGS ?= ../../third_party/opencv/opencv_cmake_flags.txt
OPENCV_INSTALL_SCRIPT ?= ../../third_party/opencv/install_opencv.sh

BASE_OS ?= ubuntu
BASE_IMAGE ?= ubuntu:20.04 

NODES ?= add_one east_ocr face_blur horizontal_ocr image_transformation model_zoo_intel_object_detection
NODE_TYPE ?= cpp

.PHONY: all

default: all

all:
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
endif
	@cp $(HEADER_FILE_PATH) .
	@cp $(OPENCV_BUILD_FLAGS) .
	@cp $(OPENCV_INSTALL_SCRIPT) .
	@cp -r ../queue.hpp ./queue.hpp
# Pass down --no-cache option to docker build for the first node, but not for the rest, the rest will re-build last layer anyway
	first_iteration=true ; for NODE_NAME in $(NODES); do \
		if [ "$$first_iteration" = true ]; then \
		  first_iteration=false ; \
		  docker_cache_param=$(NO_CACHE_OPTION) ; \
		else \
		  docker_cache_param= ; \
		fi ; \
		echo $$docker_cache_param ; \
		docker build $$docker_cache_param -f Dockerfile.$(BASE_OS) -t custom_node_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --build-arg BASE_IMAGE=$(BASE_IMAGE) --build-arg NODE_NAME=$$NODE_NAME --build-arg NODE_TYPE=$(NODE_TYPE) . || exit 1 ; \
		mkdir -p ./lib/$(BASE_OS) ; \
		docker cp $$(docker create --rm custom_node_build_image:latest):/custom_nodes/lib/libcustom_node_$$NODE_NAME.so ./lib/$(BASE_OS)/ || exit 1 ; \
		echo "Built $$NODE_NAME" ; \
	done || exit 1
	@rm ./queue.hpp
	@rm install_opencv.sh
	@rm opencv_cmake_flags.txt
	@rm custom_node_interface.h
