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

.PHONY: default build

default: docker_build

docker_build:
	@cp $(HEADER_FILE_PATH) ./custom_node_interface.h
	@docker build -f Dockerfile.$(BASE_OS) -t custom_node_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
	@rm -Rf lib
	@docker cp $$(docker create --rm custom_node_build_image:latest):/custom_nodes/lib/ ./lib/
	@rm custom_node_interface.h
