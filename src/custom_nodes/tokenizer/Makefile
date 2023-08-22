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
COMMON_DIR ?= ../common

BASE_OS ?= ubuntu

.PHONY: all

default: all

all:
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
endif
	@cp $(HEADER_FILE_PATH) .
	@cp -r $(COMMON_DIR) ./
	echo "Building tokenizer"
	docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) -t tokenizer_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
	mkdir -p ./lib/$(BASE_OS)
	docker cp $$(docker create --rm tokenizer_build_image:latest):/custom_nodes/tokenizer/build/src/libdetokenizer.so ./lib/$(BASE_OS)/
	docker cp $$(docker create --rm tokenizer_build_image:latest):/custom_nodes/tokenizer/build/src/libtokenizer.so ./lib/$(BASE_OS)/
	echo "Built tokenizer"
	@rm -rf ./common
	@rm custom_node_interface.h
