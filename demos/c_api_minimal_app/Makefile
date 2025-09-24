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
.DEFAULT_GOAL := all_web

HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"

OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
OVMS_CPP_IMAGE_TAG ?= latest
BASE_OS ?= ubuntu24

ifeq ($(BASE_OS),ubuntu24)
  BASE_OS_TAG_UBUNTU ?= 24.04
  PACKAGE_URL ?="https://github.com/openvinotoolkit/model_server/releases/download/v2025.3/ovms_ubuntu24.tar.gz"
  BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG_UBUNTU)
  DIST_OS=ubuntu
endif
ifeq ($(BASE_OS),redhat)
  BASE_OS_TAG_REDHAT ?= 9.6
  PACKAGE_URL ="https://github.com/openvinotoolkit/model_server/releases/download/v2025.3/ovms_redhat.tar.gz"
  BASE_IMAGE ?= registry.access.redhat.com/ubi9/ubi:$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
endif

clean:
	rm -vrf capi/$(DIST_OS)
	mkdir -vp capi/$(DIST_OS)/demos && cd capi/$(DIST_OS)

from_docker:
	mkdir -p capi/$(DIST_OS)
	docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
			"tar -c -C /ovms_pkg/$(BASE_OS) ovms.tar.gz ; sleep 2" | tar -x -C capi/$(DIST_OS)/ && cd ../../
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG))

from_web:
	wget -P capi/$(DIST_OS) ${PACKAGE_URL}
	mv capi/$(DIST_OS)/*.tar.gz capi/$(DIST_OS)/ovms.tar.gz

build_image:
	cp -vR capi_files/* capi/$(DIST_OS)/
	cp -r ../../src/test/dummy capi/$(DIST_OS)/demos
	cp -r ../../src/main_capi.c* capi/$(DIST_OS)/demos
	cp -r ../../src/main_benchmark.cpp capi/$(DIST_OS)/demos
	sed -i s+/ovms/src/test/configs/config.json+/ovms/demos/config.json+ capi/$(DIST_OS)/demos/main_capi.c
	sed -i s+/ovms/src/test/configs/config_standard_dummy.json+/ovms/demos/config_standard_dummy.json+ capi/$(DIST_OS)/demos/main_capi.cpp
	cd capi/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(DIST_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi:$(OVMS_CPP_IMAGE_TAG)

all_web: clean from_web build_image
	
all_docker: clean from_docker build_image
