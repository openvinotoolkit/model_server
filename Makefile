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
BASE_OS_TAG_REDHAT ?= 8.6

INSTALL_RPMS_FROM_URL ?=

CHECK_COVERAGE ?=0
NVIDIA ?=0

# NOTE: when changing any value below, you'll need to adjust WORKSPACE file by hand:
#         - uncomment source build section, comment binary section
#         - adjust binary version path - version variable is not passed to WORKSPACE file!
OV_SOURCE_BRANCH ?= master

OV_USE_BINARY ?= 1
APT_OV_PACKAGE ?= openvino-2022.1.0
# opt, dbg:
BAZEL_BUILD_TYPE ?= opt
CMAKE_BUILD_TYPE ?= Release
MINITRACE ?= OFF

ifeq ($(BAZEL_BUILD_TYPE),dbg)
  BAZEL_DEBUG_FLAGS=" --strip=never --copt=-g -c dbg "
else
  BAZEL_DEBUG_FLAGS=" --strip=never "
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
  else
	BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG_UBUNTU)
  endif
  INSTALL_DRIVER_VERSION ?= "22.35.24055"
  DLDT_PACKAGE_URL ?= http://s3.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_ubuntu20_2022.3.0.8844.12f74f066a4_x86_64.tgz
endif
ifeq ($(BASE_OS),redhat)
  BASE_OS_TAG=$(BASE_OS_TAG_REDHAT)
  BASE_IMAGE ?= registry.access.redhat.com/ubi8/ubi:$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
  INSTALL_DRIVER_VERSION ?= "22.28.23726"
  DLDT_PACKAGE_URL ?= http://s3.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_rhel8_2022.3.0.8844.12f74f066a4_x86_64.tgz
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
PRODUCT_VERSION ?= "2022.3"

OVMS_CPP_CONTAINTER_NAME ?= server-test$(shell date +%Y-%m-%d-%H.%M.%S)
OVMS_CPP_CONTAINTER_PORT ?= 9178

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

style: venv clang-format
	@echo "Style-checking codebase..."
	@git diff --exit-code || (echo "clang-format changes not commited. Commit those changes first"; exit 1)
	@git diff --exit-code --staged || (echo "clang-format changes not commited. Commit those changes first"; exit 1)
	@. $(ACTIVATE); echo ${PWD}; cpplint ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}
	@echo "Checking cppclean..."
	@. $(ACTIVATE); bash -c "./cppclean.sh"

sdl-check: venv
	@echo "Checking SDL requirements..."
	@echo "Checking docker files..."
	@bash -c "if [ $$(find . -type f -name 'Dockerfile.*' -exec grep ADD {} \; | wc -l | xargs ) -eq 0 ]; then echo 'ok'; else echo 'replace ADD with COPY in dockerfiles'; exit 1 ; fi"

	@echo "Checking python files..."
	@. $(ACTIVATE); bash -c "bandit -x demos/benchmark/python -r demos/*/python > bandit.txt"
	@if ! grep -FRq "No issues identified." bandit.txt; then\
		error Run bandit on demos/*/python/*.py to fix issues.;\
	fi
	@rm bandit.txt
	@. $(ACTIVATE); bash -c "bandit -r client/python/ > bandit2.txt"
	@if ! grep -FRq "No issues identified." bandit2.txt; then\
		error Run bandit on  client/python/ to fix issues.;\
	fi
	@rm bandit2.txt
	@echo "Checking license headers in files..."
	@. $(ACTIVATE); bash -c "python3 lib_search.py . > missing_headers.txt"
	@if ! grep -FRq "All files have headers" missing_headers.txt; then\
        echo "Files with missing headers";\
        cat missing_headers.txt;\
		exit 1;\
	fi
	@rm missing_headers.txt

	@echo "Checking forbidden functions in files..."
	@. $(ACTIVATE); bash -c "python3 lib_search.py . functions > forbidden_functions.txt"
	@if ! grep -FRq "All files checked for forbidden functions" forbidden_functions.txt; then\
		error Run python3 lib_search.py . functions - to see forbidden functions file list.;\
	fi
	@rm forbidden_functions.txt

clang-format: venv
	@echo "Formatting files with clang-format.."
	@. $(ACTIVATE); find ${STYLE_CHECK_DIRS} -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec clang-format-6.0 -style=file -i {} \;

.PHONY: docker_build
docker_build:
ifeq ($(NVIDIA),1)
  ifeq ($(OV_USE_BINARY),1)
	@echo "Building NVIDIA plugin requires OV built from source. To build NVIDIA plugin and OV from source make command should look like this 'NVIDIA=1 OV_USE_BINARY=0 make docker_build'"; exit 1 ;
  endif
endif
ifeq ($(BUILD_CUSTOM_NODES),true)
	@echo "Building custom nodes"
	@cd src/custom_nodes && make BASE_OS=$(BASE_OS)
endif
	@echo "Building docker image $(BASE_OS)"
	# Provide metadata information into image if defined
	@mkdir -p .workspace
	@bash -c '$(eval PROJECT_VER_PATCH:=`git rev-parse --short HEAD`)'
	@bash -c '$(eval PROJECT_NAME:=${PRODUCT_NAME})'
	@bash -c '$(eval PROJECT_VERSION:=${PRODUCT_VERSION}.${PROJECT_VER_PATCH})'
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
	@docker pull $(BASE_IMAGE)
  ifeq ($(BASE_OS),redhat)
	@docker pull registry.access.redhat.com/ubi8/ubi-minimal:8.6
  endif
endif
ifneq ($(OVMS_METADATA_FILE),)
	@cp $(OVMS_METADATA_FILE) .workspace/metadata.json
else
	@touch .workspace/metadata.json
endif
	@cat .workspace/metadata.json
	docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTPS_PROXY) --build-arg no_proxy=$(NO_PROXY) \
		--build-arg ovms_metadata_file=.workspace/metadata.json --build-arg ov_source_branch="$(OV_SOURCE_BRANCH)" \
		--build-arg ov_use_binary=$(OV_USE_BINARY) --build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL) \
		--build-arg APT_OV_PACKAGE=$(APT_OV_PACKAGE) --build-arg CHECK_COVERAGE=$(CHECK_COVERAGE) \
		--build-arg build_type=$(BAZEL_BUILD_TYPE) --build-arg debug_bazel_flags=$(BAZEL_DEBUG_FLAGS) \
		--build-arg CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		--build-arg minitrace_flags=$(MINITRACE_FLAGS) \
		--build-arg PROJECT_NAME=${PROJECT_NAME} \
		--build-arg PROJECT_VERSION=${PROJECT_VERSION} \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NVIDIA=$(NVIDIA) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--build-arg JOBS=$(JOBS)
	docker build $(NO_CACHE_OPTION) -f DockerfileMakePackage . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg ov_use_binary=$(OV_USE_BINARY) --build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL) --build-arg BASE_OS=$(BASE_OS) \
		--build-arg NVIDIA=$(NVIDIA) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) \
		--build-arg BUILD_IMAGE=$(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)
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
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NVIDIA=$(NVIDIA) \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)
	cd dist/$(DIST_OS)/ && docker build $(NO_CACHE_OPTION) -f Dockerfile.$(BASE_OS) . \
    	--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
    	--build-arg no_proxy=$(NO_PROXY) \
    	--build-arg INSTALL_RPMS_FROM_URL="$(INSTALL_RPMS_FROM_URL)" \
		--build-arg INSTALL_DRIVER_VERSION="$(INSTALL_DRIVER_VERSION)" \
    	--build-arg GPU=1 \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NVIDIA=$(NVIDIA) \
    	-t $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-gpu$(IMAGE_TAG_SUFFIX)
	cd extras/nginx-mtls-auth && \
	    http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) no_proxy=$(NO_PROXY) ./build.sh "$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(BASE_OS)" && \
	    docker tag $(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-nginx-mtls$(IMAGE_TAG_SUFFIX)

# Ci build expects index.html in genhtml directory
get_coverage:
	@echo "Copying coverage report from build image to genhtml if exist..."
	@docker create -ti --name $(OVMS_CPP_CONTAINTER_NAME) $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG) bash
	@docker cp $(OVMS_CPP_CONTAINTER_NAME):/ovms/genhtml/ .  || true
	@docker rm -f $(OVMS_CPP_CONTAINTER_NAME)
	@if [ -d genhtml/src ]; then $(MAKE) check_coverage; \
	else echo "ERROR: genhtml/src was not generated during build"; \
	fi
check_coverage:
	@echo "Checking if coverage is above threshold..."
	@docker run $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG) ./check_coverage.bat | grep success
	
test_checksec:
	@echo "Running checksec on ovms binary..."
	@docker rm -f $(OVMS_CPP_CONTAINTER_NAME) || true
	@docker create -ti --name $(OVMS_CPP_CONTAINTER_NAME) $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash
	@docker cp $(OVMS_CPP_CONTAINTER_NAME):/ovms_release/bin/ovms /tmp
	@docker rm -f $(OVMS_CPP_CONTAINTER_NAME)
	@checksec --file=/tmp/ovms --format=csv > checksec.txt
	@if ! grep -FRq "Full RELRO,Canary found,NX enabled,PIE enabled,No RPATH,RUNPATH,Symbols,Yes" checksec.txt; then\
 		error Run checksec on ovms binary and fix issues.;\
	fi
	@rm -f checksec.txt
	@rm -f /tmp/ovms
	@echo "Checksec check success."

test_perf: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(HOME)/resnet50-binary:/models/resnet50-binary \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet-binary --model_path /models/resnet50-binary --port $(OVMS_CPP_CONTAINTER_PORT); sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py \
	  --grpc_port $(OVMS_CPP_CONTAINTER_PORT) \
		--images_numpy_path tests/performance/imgs.npy \
		--labels_numpy_path tests/performance/labels.npy \
		--iteration 1000 \
		--batchsize 1 \
		--report_every 100 \
		--input_name 0 \
		--output_name 1463 \
		--model_name resnet-binary
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)

test_perf_dummy_model: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(PWD)/src/test/dummy/1:/dummy/1 \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name dummy --model_path /dummy --port $(OVMS_CPP_CONTAINTER_PORT); sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py \
	  --grpc_port $(OVMS_CPP_CONTAINTER_PORT) \
		--images_numpy_path tests/performance/dummy_input.npy \
		--labels_numpy_path tests/performance/dummy_lbs.npy \
		--iteration 10000 \
		--batchsize 1 \
		--report_every 1000 \
		--input_name b \
		--output_name a \
		--model_name dummy
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)


test_throughput: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(HOME)/resnet50-binary:/models/resnet50-binary \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet-binary \
		--model_path /models/resnet50-binary \
		--port $(OVMS_CPP_CONTAINTER_PORT); \
		sleep 10
	@echo "Running throughput test"
	@. $(ACTIVATE); cd tests/performance; ./grpc_throughput.sh 28 \
	  --grpc_port $(OVMS_CPP_CONTAINTER_PORT) \
		--images_numpy_path imgs.npy \
		--labels_numpy_path labels.npy \
		--iteration 500 \
		--batchsize 1 \
		--input_name 0 \
		--output_name 1463 \
		--model_name resnet-binary
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)

test_throughput_dummy_model: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(PWD)/src/test/dummy/1:/dummy/1 \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name dummy \
		--model_path /dummy \
		--port $(OVMS_CPP_CONTAINTER_PORT); \
		sleep 10
	@echo "Running throughput test"
	@. $(ACTIVATE); cd tests/performance; ./grpc_throughput.sh 28 \
	  --grpc_port $(OVMS_CPP_CONTAINTER_PORT) \
		--images_numpy_path dummy_input.npy \
		--labels_numpy_path dummy_lbs.npy \
		--iteration 10000 \
		--batchsize 1 \
		--input_name b \
		--output_name a \
		--model_name dummy
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)

test_functional: venv
	@. $(ACTIVATE); pytest --json=report.json -v -s $(TEST_PATH)

# Client library make style target, by default uses Python 3 env in .venv path
# This fact is used in test_client_lib, where make build runs in .venv Python 3 environment
test_client_lib:
	@cd client/python/ovmsclient/lib && \
		make style || exit 1 && \
		. .venv/bin/activate; make build || exit 1 && \
		make test || exit 1 && \
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
		--build-arg APT_OV_PACKAGE=${APT_OV_PACKAGE} .
	mkdir -p ./lib/${BASE_OS}
	docker cp $$(docker create --rm sample_cpu_extension:latest):/workspace/libcustom_relu_cpu_extension.so ./lib/${BASE_OS}
