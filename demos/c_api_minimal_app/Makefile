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
BASE_OS_TAG_UBUNTU ?= 20.04
BASE_OS_TAG_REDHAT ?= 8.7

OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
OVMS_CPP_IMAGE_TAG ?= latest
BASE_OS ?= ubuntu
DIST_OS ?= $(BASE_OS)

ifeq ($(BASE_OS),ubuntu)
  ifeq ($(BASE_OS_TAG_UBUNTU),20.04)
	DLDT_PACKAGE_URL ?= http://s3.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_ubuntu20_2023.1.0.11298.1588a332173_x86_64.tgz
  else ifeq  ($(BASE_OS_TAG_UBUNTU),22.04)
	DLDT_PACKAGE_URL ?= http://s3.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_ubuntu22_2023.1.0.11298.1588a332173_x86_64.tgz
  endif
  BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG_UBUNTU)
endif
ifeq ($(BASE_OS),redhat)
  BASE_IMAGE ?= registry.access.redhat.com/ubi8/ubi:$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
  DLDT_PACKAGE_URL ?= http://s3.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_rhel8_2023.1.0.11298.1588a332173_x86_64.tgz
endif

clean:
	rm -vrf capi/$(DIST_OS)
	mkdir -vp capi/$(DIST_OS)/demos && cd capi/$(DIST_OS)

from_docker:
	docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
			"tar -c -C / ovms.tar.gz ; sleep 2" | tar -x -C capi/$(DIST_OS)/ && cd ../../
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG))

from_web:
	wget -P capi/$(DIST_OS) ${DLDT_PACKAGE_URL}
	mv capi/$(DIST_OS)/*.tar.gz capi/$(DIST_OS)/ovms.tar.gz

build_image:
	cp -vR capi_files/* capi/$(DIST_OS)/
	cp -r ../../src/test/dummy capi/$(DIST_OS)/demos
	cp -r ../../src/main_capi.c* capi/$(DIST_OS)/demos
	sed -i s+/ovms/src/test/c_api/config.json+/ovms/demos/config.json+ capi/$(DIST_OS)/demos/main_capi.c
	sed -i s+/ovms/src/test/c_api/config_standard_dummy.json+/ovms/demos/config_standard_dummy.json+ capi/$(DIST_OS)/demos/main_capi.cpp
	cd capi/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi:$(OVMS_CPP_IMAGE_TAG)

all_web: clean from_web build_image
	
all_docker: clean from_docker build_image
