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
  BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG_UBUNTU)
endif
ifeq ($(BASE_OS),redhat)
  BASE_IMAGE ?= registry.access.redhat.com/ubi8/ubi:$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
endif

all:
	rm -vrf capi/$(DIST_OS) && mkdir -vp capi/$(DIST_OS)/demos && cd capi/$(DIST_OS) && \
		docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
			"tar -c -C / ovms.tar* ; sleep 2" | tar -x
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG))
	cd capi/$(DIST_OS) && sha256sum --check ovms.tar.gz.sha256
	cd capi/$(DIST_OS) && sha256sum --check ovms.tar.xz.sha256
	cp -vR capi_files/* capi/$(DIST_OS)/
	cp -r ../../src/test/dummy capi/$(DIST_OS)/demos
	cp -r ../../src/main3.c* capi/$(DIST_OS)/demos
	sed -i s+/ovms/src/test/c_api/config.json+/ovms/demos/config.json+ capi/$(DIST_OS)/demos/main3.c
	sed -i s+/ovms/src/test/c_api/config_standard_dummy.json+/ovms/demos/config_standard_dummy.json+ capi/$(DIST_OS)/demos/main3.cpp
	cd capi/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi:$(OVMS_CPP_IMAGE_TAG)