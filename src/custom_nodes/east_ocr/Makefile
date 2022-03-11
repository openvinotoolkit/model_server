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

HEADER_FILE_PATH ?= ../../custom_node_interface.h

BASE_OS ?= ubuntu

OPENCV_DOWNLOAD_FOLDER ?= http://s3.toolbox.iotg.sclab.intel.com/opencv-packages/opencv/release_2022.1/20220311_0602-4.5.5_090/

.PHONY: default build

default: docker_build

docker_build:
	@cp $(HEADER_FILE_PATH) ./custom_node_interface.h
	@mkdir common
	@cp -r ../common/utils.hpp ./common/utils.hpp
	@cp -r ../common/opencv_utils.hpp ./common/opencv_utils.hpp
	@docker build -f Dockerfile.$(BASE_OS) -t custom_node_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --build-arg OPENCV_DOWNLOAD_FOLDER=$(OPENCV_DOWNLOAD_FOLDER) .
	@rm -Rf lib
	@docker cp $$(docker create --rm custom_node_build_image:latest):/custom_nodes/lib/ .
	@rm -rf ./common
	@rm custom_node_interface.h
