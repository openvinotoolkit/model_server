#
# Copyright (c) 2020-2021 Intel Corporation
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
HEADER_FILE_PATH ?= ../../customloaderinterface.hpp
OUT_LIB = libsampleloader.so
BASE_OS ?= ubuntu22
LIB_INC = -Iinclude -I../../ -Irapidjson-1.1.0/include
CFLAGS = -fPIC -Wl,-rpath,. -fpic 
OPS = -O2 -U_FORTIFY_SOURCE -fstack-protector -fno-omit-frame-pointer -D_FORTIFY_SOURCE=1 -fno-strict-overflow
LDFLAGS = -ldl -lpthread

BASE_OS_TAG_UBUNTU ?= 22.04
BASE_OS_TAG_REDHAT ?= 8.10

ifeq ($(findstring ubuntu,$(BASE_OS)),ubuntu)
  BASE_OS_TAG=$(BASE_OS_TAG_UBUNTU)
  ifeq ($(BASE_OS),ubuntu22)
	BASE_OS_TAG=22.04
  endif
  ifeq ($(BASE_OS),ubuntu20)
	BASE_OS_TAG=20.04
  endif
  BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG)
  DIST_OS=ubuntu
endif
ifeq ($(BASE_OS),redhat)
  BASE_OS_TAG=$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
endif
BASE_IMAGE=$(DIST_OS):$(BASE_OS_TAG)

all:
	wget https://github.com/Tencent/rapidjson/archive/v1.1.0.zip
	unzip v1.1.0.zip
	g++ -c -g0 -fPIC $(OPS) $(LIB_INC) *.cpp
	g++ *.o -g0 $(CFLAGS) $(LDFLAGS) $(LIB_INC) -shared -o $(OUT_LIB) 
	$(RM) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r

clean:
	$(RM) $(OUT_LIB) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r

docker_build:
	cp $(HEADER_FILE_PATH) ./customloaderinterface.hpp
	docker build -f Dockerfile.$(DIST_OS) -t custom_loader_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg BASE_IMAGE=${BASE_IMAGE} .
	rm -Rf lib && mkdir ./lib
	docker cp $$(docker create --rm custom_loader_build_image:latest):/src/example/SampleCustomLoader/libsampleloader.so ./lib/
	rm customloaderinterface.hpp
