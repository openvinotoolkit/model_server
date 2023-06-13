#
# Copyright (c) 2020-2022 Intel Corporation
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

# workaround for docker clipping build step logs
BUILDKIT_STEP_LOG_MAX_SIZE=500000000
BUILDKIT_STEP_LOG_MAX_SPEED=10000000

VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"
STYLE_CHECK_OPTS := --extensions=hpp,cc,cpp,h \
	--output=vs7 \
	--recursive \
	--linelength=120 \
	--filter=-build/c++11,-runtime/references,-whitespace/braces,-whitespace/indent,-build/include_order,-runtime/indentation_namespace,-build/namespaces,-whitespace/line_length,-runtime/string,-readability/casting,-runtime/explicit,-readability/todo
STYLE_CHECK_DIRS := src demos/common/cpp/src demos/image_classification/cpp demos/benchmark/cpp
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"
JOBS ?= $(shell python3 -c 'import multiprocessing as mp; print(mp.cpu_count())')

# Image on which OVMS is compiled. If DIST_OS is not set, it's also used for a release image.
# Currently supported BASE_OS values are: ubuntu redhat
BASE_OS ?= ubuntu

# do not change this; change versions per OS a few lines below (BASE_OS_TAG_*)!
BASE_OS_TAG ?= latest

BASE_OS_TAG_UBUNTU ?= 20.04
BASE_OS_TAG_REDHAT ?= 8.7

INSTALL_RPMS_FROM_URL ?=

CHECK_COVERAGE ?=0
RUN_TESTS ?= 1
NVIDIA ?=0
BUILD_NGINX ?= 0
MEDIAPIPE_DISABLE ?= 0

# NOTE: when changing any value below, you'll need to adjust WORKSPACE file by hand:
#         - uncomment source build section, comment binary section
#         - adjust binary version path - version variable is not passed to WORKSPACE file!
OV_SOURCE_BRANCH ?= releases/2023/0
OV_CONTRIB_BRANCH ?= releases/2023/0

OV_SOURCE_ORG ?= openvinotoolkit
OV_CONTRIB_ORG ?= openvinotoolkit

SENTENCEPIECE ?= 1

OV_USE_BINARY ?= 1
APT_OV_PACKAGE ?= openvino-2022.1.0
# opt, dbg:
BAZEL_BUILD_TYPE ?= opt
CMAKE_BUILD_TYPE ?= Release
MINITRACE ?= OFF

DISABLE_MEDIAPIPE_PARAMS ?= ""
ifeq ($(MEDIAPIPE_DISABLE),1)
	DISABLE_MEDIAPIPE_PARAMS = " --define MEDIAPIPE_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 "
endif

ifeq ($(BAZEL_BUILD_TYPE),dbg)
  BAZEL_DEBUG_FLAGS=" --strip=never --copt=-g -c dbg "$(DISABLE_MEDIAPIPE_PARAMS)
else
  BAZEL_DEBUG_FLAGS=" --strip=never "$(DISABLE_MEDIAPIPE_PARAMS)
endif

ifeq ($(MINITRACE),ON)
  MINITRACE_FLAGS="--copt=-DMTR_ENABLED"
else
  MINITRACE_FLAGS=""
endif

# Option to Override release image.
# Release image OS *must have* glibc version >= glibc version on BASE_OS:
DIST_OS ?= $(BASE_OS)
DIST_OS_TAG ?= $(BASE_OS_TAG)

ifeq ($(BASE_OS),ubuntu)
  BASE_OS_TAG=$(BASE_OS_TAG_UBUNTU)
  ifeq ($(NVIDIA),1)
	BASE_IMAGE=docker.io/nvidia/cuda:11.8.0-runtime-ubuntu20.04
	BASE_IMAGE_RELEASE=$(BASE_IMAGE)
  else
	BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG_UBUNTU)
	BASE_IMAGE_RELEASE=$(BASE_IMAGE)
  endif
  ifeq ($(BASE_OS_TAG_UBUNTU),20.04)
	INSTALL_DRIVER_VERSION ?= "22.43.24595"
	DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64.tgz
  else ifeq  ($(BASE_OS_TAG_UBUNTU),22.04)
	INSTALL_DRIVER_VERSION ?= "23.13.26032"
	DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64.tgz
  endif
endif
ifeq ($(BASE_OS),redhat)
  BASE_OS_TAG=$(BASE_OS_TAG_REDHAT)
  ifeq ($(NVIDIA),1)
    BASE_IMAGE=docker.io/nvidia/cuda:11.8.0-runtime-ubi8
	BASE_IMAGE_RELEASE=$(BASE_IMAGE)
  else
    BASE_IMAGE ?= registry.access.redhat.com/ubi8/ubi:$(BASE_OS_TAG_REDHAT)
	BASE_IMAGE_RELEASE=registry.access.redhat.com/ubi8/ubi-minimal:$(BASE_OS_TAG_REDHAT)
  endif	
  DIST_OS=redhat
  INSTALL_DRIVER_VERSION ?= "22.43.24595"
  DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_rhel8_2023.0.0.10926.b4452d56304_x86_64.tgz
endif

OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
ifeq ($(BAZEL_BUILD_TYPE),dbg)
  OVMS_CPP_DOCKER_IMAGE:=$(OVMS_CPP_DOCKER_IMAGE)-dbg
endif

OVMS_CPP_IMAGE_TAG ?= latest
ifeq ($(NVIDIA),1)
  IMAGE_TAG_SUFFIX = -cuda
endif

PRODUCT_NAME = "OpenVINO Model Server"
PRODUCT_VERSION ?= "2023.0.0"

OVMS_CPP_CONTAINER_NAME ?= server-test$(shell date +%Y-%m-%d-%H.%M.%S)
OVMS_CPP_CONTAINER_PORT ?= 9178

TEST_PATH ?= tests/functional/

BUILD_CUSTOM_NODES ?= true

.PHONY: default docker_build \

default: docker_build

venv:$(ACTIVATE)
	@echo -n "Using venv "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip3 install --upgrade pip
	@. $(ACTIVATE); pip3 install -vUqq setuptools
	@. $(ACTIVATE); pip3 install -qq -r tests/requirements.txt --use-deprecated=legacy-resolver
	@touch $(ACTIVATE)

cppclean: venv
	@echo "Checking cppclean..."
	@. $(ACTIVATE); bash -c "./ci/cppclean.sh"

style: venv clang-format-check cpplint cppclean

hadolint:
	@echo "Checking SDL requirements..."
	@echo "Checking docker files..."
	@./tests/hadolint.sh

bandit:
	@echo "Checking python files..."
	@. $(ACTIVATE); bash -c "./ci/bandit.sh"

license-headers:
	@echo "Checking license headers in files..."
	@. $(ACTIVATE); bash -c "python3 ./ci/lib_search.py . > missing_headers.txt"
	@if ! grep -FRq "All files have headers" missing_headers.txt; then\
        echo "Files with missing headers";\
        cat missing_headers.txt;\
		exit 1;\
	fi
	@rm missing_headers.txt

sdl-check: venv hadolint bandit license-headers

	@echo "Checking forbidden functions in files..."
	@. $(ACTIVATE); bash -c "python3 ./ci/lib_search.py . functions > forbidden_functions.txt"
	@if ! grep -FRq "All files checked for forbidden functions" forbidden_functions.txt; then\
		error Run python3 ./ci/lib_search.py . functions - to see forbidden functions file list.;\
	fi
	@rm forbidden_functions.txt

cpplint: venv
	@echo "Style-checking codebase..."
	@. $(ACTIVATE); echo ${PWD}; cpplint ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}

clang-format: venv
	@echo "Formatting files with clang-format.."
	@. $(ACTIVATE); find ${STYLE_CHECK_DIRS} -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec clang-format-6.0 -style=file -i {} \;

clang-format-check: clang-format
	@echo "Checking if clang-format changes were committed ..."
	@git diff --exit-code || (echo "clang-format changes not commited. Commit those changes first"; exit 1)
	@git diff --exit-code --staged || (echo "clang-format changes not commited. Commit those changes first"; exit 1)

.PHONY: docker_build
docker_build: ovms_builder_image targz_package ovms_release_image
ovms_builder_image:
ifeq ($(CHECK_COVERAGE),1)
  ifeq ($(RUN_TESTS),0)
	@echo "Cannot test coverage without running tests. Use 'CHECK_COVERAGE=1 RUN_TESTS=1 make docker_build'"; exit 1 ;
  endif
endif
ifeq ($(NVIDIA),1)
  ifeq ($(OV_USE_BINARY),1)
	@echo "Building NVIDIA plugin requires OV built from source. To build NVIDIA plugin and OV from source make command should look like this 'NVIDIA=1 OV_USE_BINARY=0 make docker_build'"; exit 1 ;
  endif
  ifeq ($(BASE_OS),redhat)
	@echo "copying RH entitlements"
	@cp -ru /etc/pki/entitlement .
	@mkdir rhsm-ca
	@cp -u /etc/rhsm/ca/* rhsm-ca/
  endif
endif
ifeq ($(BASE_OS),redhat)
	@mkdir -p entitlement
	@mkdir -p rhsm-ca
endif
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
	@docker pull $(BASE_IMAGE)
  ifeq ($(BASE_OS),redhat)
	@docker pull registry.access.redhat.com/ubi8/ubi-minimal:$(BASE_OS_TAG_REDHAT)
    ifeq ($(NVIDIA),1)
	@docker pull docker.io/nvidia/cuda:11.8.0-runtime-ubi8
    endif
  endif
endif
ifeq ($(BUILD_CUSTOM_NODES),true)
	@echo "Building custom nodes"
	@cd src/custom_nodes && make NO_DOCKER_CACHE=$(NO_DOCKER_CACHE) BASE_OS=$(BASE_OS) BASE_IMAGE=$(BASE_IMAGE) 
	@cd src/custom_nodes/tokenizer && make NO_DOCKER_CACHE=$(NO_DOCKER_CACHE) BASE_OS=$(BASE_OS) BASE_IMAGE=$(BASE_IMAGE) 
endif
	@echo "Building docker image $(BASE_OS)"
	# Provide metadata information into image if defined
	@mkdir -p .workspace
	@bash -c '$(eval PROJECT_VER_PATCH:=`git rev-parse --short HEAD`)'
	@bash -c '$(eval PROJECT_NAME:=${PRODUCT_NAME})'
	@bash -c '$(eval PROJECT_VERSION:=${PRODUCT_VERSION}.${PROJECT_VER_PATCH})'
ifneq ($(OVMS_METADATA_FILE),)
	@cp $(OVMS_METADATA_FILE) .workspace/metadata.json
else
	@touch .workspace/metadata.json
endif
	@cat .workspace/metadata.json
	docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTPS_PROXY) --build-arg no_proxy=$(NO_PROXY) \
		--build-arg ovms_metadata_file=.workspace/metadata.json --build-arg ov_source_branch="$(OV_SOURCE_BRANCH)" --build-arg ov_source_org="$(OV_SOURCE_ORG)" \
		--build-arg ov_contrib_org="$(OV_CONTRIB_ORG)" \
		--build-arg ov_use_binary=$(OV_USE_BINARY) --build-arg sentencepiece=$(SENTENCEPIECE) --build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL) \
		--build-arg APT_OV_PACKAGE=$(APT_OV_PACKAGE) --build-arg CHECK_COVERAGE=$(CHECK_COVERAGE) --build-arg RUN_TESTS=$(RUN_TESTS)\
		--build-arg build_type=$(BAZEL_BUILD_TYPE) --build-arg debug_bazel_flags=$(BAZEL_DEBUG_FLAGS) \
		--build-arg CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		--build-arg minitrace_flags=$(MINITRACE_FLAGS) \
		--build-arg PROJECT_NAME=${PROJECT_NAME} \
		--build-arg PROJECT_VERSION=${PROJECT_VERSION} \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NVIDIA=$(NVIDIA) --build-arg ov_contrib_branch="$(OV_CONTRIB_BRANCH)" \
		-t $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--build-arg JOBS=$(JOBS)

targz_package: ovms_builder_image
	docker build $(NO_CACHE_OPTION) -f DockerfileMakePackage . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg ov_use_binary=$(OV_USE_BINARY) --build-arg sentencepiece=$(SENTENCEPIECE) --build-arg BASE_OS=$(BASE_OS) \
		--build-arg NVIDIA=$(NVIDIA) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) \
		--build-arg BUILD_IMAGE=$(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)

ovms_release_image: targz_package
	rm -vrf dist/$(DIST_OS) && mkdir -vp dist/$(DIST_OS) && cd dist/$(DIST_OS) && \
		docker run $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash -c \
			"tar -c -C / ovms.tar* ; sleep 2" | tar -x
	-docker rm -v $$(docker ps -a -q -f status=exited -f ancestor=$(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX))
	cd dist/$(DIST_OS) && sha256sum --check ovms.tar.gz.sha256
	cd dist/$(DIST_OS) && sha256sum --check ovms.tar.xz.sha256
	cp -vR release_files/* dist/$(DIST_OS)/
	cd dist/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg INSTALL_RPMS_FROM_URL="$(INSTALL_RPMS_FROM_URL)" \
		--build-arg GPU=0 \
		--build-arg BASE_IMAGE=$(BASE_IMAGE_RELEASE) \
		--build-arg NVIDIA=$(NVIDIA) \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)
	cd dist/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
    	--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
    	--build-arg no_proxy=$(NO_PROXY) \
    	--build-arg INSTALL_RPMS_FROM_URL="$(INSTALL_RPMS_FROM_URL)" \
		--build-arg INSTALL_DRIVER_VERSION="$(INSTALL_DRIVER_VERSION)" \
    	--build-arg GPU=1 \
		--build-arg BASE_IMAGE=$(BASE_IMAGE_RELEASE) \
		--build-arg NVIDIA=$(NVIDIA) \
    	-t $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-gpu$(IMAGE_TAG_SUFFIX)
ifeq ($(BUILD_NGINX), 1)
	cd extras/nginx-mtls-auth && \
	http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) no_proxy=$(NO_PROXY) ./build.sh "$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(BASE_OS)" && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-nginx-mtls$(IMAGE_TAG_SUFFIX)
endif

# Ci build expects index.html in genhtml directory
get_coverage:
	@echo "Copying coverage report from build image to genhtml if exist..."
	@docker create -ti --name $(OVMS_CPP_CONTAINER_NAME) $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG) bash
	@docker cp $(OVMS_CPP_CONTAINER_NAME):/ovms/genhtml/ .  || true
	@docker rm -f $(OVMS_CPP_CONTAINER_NAME) || true
	@if [ -d genhtml/src ]; then $(MAKE) check_coverage; \
	else echo "ERROR: genhtml/src was not generated during build"; \
	fi
check_coverage:
	@echo "Checking if coverage is above threshold..."
	@docker run $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG) ./check_coverage.bat | grep success
	
test_checksec:
	@echo "Running checksec on libovms_shared library..."
	@docker rm -f $(OVMS_CPP_CONTAINER_NAME) || true
	@docker create -ti --name $(OVMS_CPP_CONTAINER_NAME) $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash
	@docker cp $(OVMS_CPP_CONTAINER_NAME):/ovms_release/lib/libovms_shared.so /tmp
	@docker cp $(OVMS_CPP_CONTAINER_NAME):/ovms_release/bin/ovms /tmp
	@docker rm -f $(OVMS_CPP_CONTAINER_NAME) || true
	@checksec --file=/tmp/libovms_shared.so --format=csv > checksec.txt
	@if ! grep -FRq "Full RELRO,Canary found,NX enabled,DSO,No RPATH,RUNPATH,Symbols,Yes" checksec.txt; then\
 		echo "ERROR: OVMS shared library security settings changed. Run checksec on ovms shared library and fix issues." && exit 1;\
	fi
	@echo "Running checksec on ovms binary..."
	@checksec --file=/tmp/ovms --format=csv > checksec.txt
	@if ! grep -FRq "Full RELRO,Canary found,NX enabled,PIE enabled,No RPATH,RUNPATH,Symbols,Yes" checksec.txt; then\
 		echo "ERROR: OVMS binary security settings changed. Run checksec on ovms binary and fix issues." && exit 1;\
	fi
	@rm -f checksec.txt
	@rm -f /tmp/ovms
	@rm -f /tmp/libovms_shared.so
	@echo "Checksec check success."

test_perf: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINER_NAME) \
		-v $(HOME)/resnet50-binary:/models/resnet50-binary \
		-p $(OVMS_CPP_CONTAINER_PORT):$(OVMS_CPP_CONTAINER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet-binary --model_path /models/resnet50-binary --port $(OVMS_CPP_CONTAINER_PORT); sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py \
	  --grpc_port $(OVMS_CPP_CONTAINER_PORT) \
		--images_numpy_path tests/performance/imgs.npy \
		--labels_numpy_path tests/performance/labels.npy \
		--iteration 1000 \
		--batchsize 1 \
		--report_every 100 \
		--input_name 0 \
		--output_name 1463 \
		--model_name resnet-binary
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME)

test_perf_dummy_model: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME) || true
	@echo "Starting docker image"
	@docker run -d --name $(OVMS_CPP_CONTAINER_NAME) \
		-v $(PWD)/src/test/dummy/1:/dummy/1 \
		-p $(OVMS_CPP_CONTAINER_PORT):$(OVMS_CPP_CONTAINER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name dummy --model_path /dummy --port $(OVMS_CPP_CONTAINER_PORT); sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py \
	  --grpc_port $(OVMS_CPP_CONTAINER_PORT) \
		--images_numpy_path tests/performance/dummy_input.npy \
		--labels_numpy_path tests/performance/dummy_lbs.npy \
		--iteration 10000 \
		--batchsize 1 \
		--report_every 1000 \
		--input_name b \
		--output_name a \
		--model_name dummy
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME)


test_throughput: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINER_NAME) \
		-v $(HOME)/resnet50-binary:/models/resnet50-binary \
		-p $(OVMS_CPP_CONTAINER_PORT):$(OVMS_CPP_CONTAINER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet-binary \
		--model_path /models/resnet50-binary \
		--port $(OVMS_CPP_CONTAINER_PORT); \
		sleep 10
	@echo "Running throughput test"
	@. $(ACTIVATE); cd tests/performance; ./grpc_throughput.sh 28 \
	  --grpc_port $(OVMS_CPP_CONTAINER_PORT) \
		--images_numpy_path imgs.npy \
		--labels_numpy_path labels.npy \
		--iteration 500 \
		--batchsize 1 \
		--input_name 0 \
		--output_name 1463 \
		--model_name resnet-binary
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME)

test_throughput_dummy_model: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME) || true
	@echo "Starting docker image"
	@docker run -d --name $(OVMS_CPP_CONTAINER_NAME) \
		-v $(PWD)/src/test/dummy/1:/dummy/1 \
		-p $(OVMS_CPP_CONTAINER_PORT):$(OVMS_CPP_CONTAINER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name dummy \
		--model_path /dummy \
		--port $(OVMS_CPP_CONTAINER_PORT); \
		sleep 10
	@echo "Running throughput test"
	@. $(ACTIVATE); cd tests/performance; ./grpc_throughput.sh 28 \
	  --grpc_port $(OVMS_CPP_CONTAINER_PORT) \
		--images_numpy_path dummy_input.npy \
		--labels_numpy_path dummy_lbs.npy \
		--iteration 10000 \
		--batchsize 1 \
		--input_name b \
		--output_name a \
		--model_name dummy
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINER_NAME)

test_functional: venv
	@. $(ACTIVATE); pytest --json=report.json -v -s $(TEST_PATH)

# Client library make style target, by default uses Python 3 env in .venv path
# This fact is used in test_client_lib, where make build runs in .venv Python 3 environment
test_client_lib:
	@cd client/python/ovmsclient/lib && \
		make style || exit 1 && \
		. .venv-ovmsclient/bin/activate; make build || exit 1 && \
		make test TEST_TYPE=FULL || exit 1 && \
		make clean

tools_get_deps:
	cd tools/deps/$(BASE_OS) && docker build --build-arg http_proxy="$(http_proxy)" --build-arg https_proxy="$(https_proxy)" -t  $(OVMS_CPP_DOCKER_IMAGE)-deps:$(OVMS_CPP_IMAGE_TAG) .
	-docker rm -f ovms-$(BASE_OS)-deps
	docker run -d --rm --name  ovms-$(BASE_OS)-deps  $(OVMS_CPP_DOCKER_IMAGE)-deps:$(OVMS_CPP_IMAGE_TAG)
	sleep 5
	docker cp ovms-$(BASE_OS)-deps:/root/rpms.tar.xz ./
	sleep 5
	-docker rm -f ovms-$(BASE_OS)-deps
	@echo "Success! Dependencies saved to rpms.tar.xz in this directory"

cpu_extension:
	cd src/example/SampleCpuExtension && \
	docker build -f Dockerfile.$(BASE_OS) -t sample_cpu_extension:latest \
		--build-arg http_proxy=${http_proxy} \
		--build-arg https_proxy=${https_proxy} \
		--build-arg no_proxy=${no_proxy} \
		--build-arg DLDT_PACKAGE_URL=${DLDT_PACKAGE_URL} \
		--build-arg APT_OV_PACKAGE=${APT_OV_PACKAGE} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} .
	mkdir -p ./lib/${BASE_OS}
	docker cp $$(docker create --rm sample_cpu_extension:latest):/workspace/libcustom_relu_cpu_extension.so ./lib/${BASE_OS}
