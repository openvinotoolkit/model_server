#
# Copyright (c) 2020 Intel Corporation
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
# limitations under the License.got

# workaround for docker clipping build step logs
BUILDKIT_STEP_LOG_MAX_SIZE=500000000
BUILDKIT_STEP_LOG_MAX_SPEED=10000000

VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
VIRTUALENV_STYLE_DIR := .venv-style
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"
ACTIVATE_STYLE="$(VIRTUALENV_STYLE_DIR)/bin/activate"
STYLE_CHECK_OPTS := --extensions=hpp,cc,cpp,h \
	--output=vs7 \
	--recursive \
	--linelength=120 \
	--filter=-build/c++11,-runtime/references,-whitespace/braces,-whitespace/indent,-build/include_order,-runtime/indentation_namespace,-build/namespaces,-whitespace/line_length,-runtime/string,-readability/casting,-runtime/explicit,-readability/todo
STYLE_CHECK_DIRS := src
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"
ifeq ($(shell uname),Darwin)
    # MacOS
    CORES_TOTAL := $(shell sysctl -n hw.physicalcpu)
else
    # Ubuntu & Redhat
    CORES_PER_SOCKET := $(shell lscpu | awk '/^Core\(s\) per socket:/ {print $$NF}')
    SOCKETS := $(shell lscpu | awk '/^Socket\(s\):/ {print $$NF}')
    CORES_TOTAL := $$(($(SOCKETS) * $(CORES_PER_SOCKET)))
endif
JOBS ?= $(CORES_TOTAL)


# Image on which OVMS is compiled. If DIST_OS is not set, it's also used for a release image.
# Currently supported BASE_OS values are: ubuntu24 ubuntu22 redhat
BASE_OS ?= ubuntu24

# do not change this; change versions per OS a few lines below (BASE_OS_TAG_*)!
BASE_OS_TAG ?= latest

BASE_OS_TAG_UBUNTU ?= 24.04
BASE_OS_TAG_REDHAT ?= 9.6

INSTALL_RPMS_FROM_URL ?=
BUILD_IMAGE ?= build
CHECK_COVERAGE ?=0
RUN_TESTS ?= 0
BUILD_TESTS ?= 0
RUN_GPU_TESTS ?=
GPU ?= 0
NPU ?= 0
BUILD_NGINX ?= 0
MEDIAPIPE_DISABLE ?= 0
PYTHON_DISABLE ?= 0
ifeq ($(MEDIAPIPE_DISABLE),1)
ifeq ($(PYTHON_DISABLE),0)
$(error PYTHON_DISABLE cannot be 0 when MEDIAPIPE_DISABLE is 1)
endif
endif
FUZZER_BUILD ?= 0

# NOTE: when changing any value below, you'll need to adjust WORKSPACE file by hand:
#         - uncomment source build section, comment binary section
#         - adjust binary version path - version variable is not passed to WORKSPACE file!

OV_SOURCE_BRANCH ?= 997b5c48447a1856728b5d6107bcac21fb1f6053 # master 2025/08/28
OV_CONTRIB_BRANCH ?= c39462ca8d7c550266dc70cdbfbe4fc8c5be0677  # master / 2024-10-31
OV_TOKENIZERS_BRANCH ?= 9a8ae92ce8197752c415de6200197c46fa1a8e0e # master 2025/08/22

OV_SOURCE_ORG ?= openvinotoolkit
OV_CONTRIB_ORG ?= openvinotoolkit

TEST_LLM_PATH ?= "src/test/llm_testing"
GPU_MODEL_PATH ?= "/tmp/face_detection_adas"

OV_USE_BINARY ?= 1
APT_OV_PACKAGE ?= openvino-2022.1.0
# opt, dbg:
BAZEL_BUILD_TYPE ?= opt
CMAKE_BUILD_TYPE ?= Release
MINITRACE ?= OFF
OV_TRACING_ENABLE ?= 0

ifeq ($(MEDIAPIPE_DISABLE),1)
	DISABLE_MEDIAPIPE_PARAMS = " --define MEDIAPIPE_DISABLE=1"
else
	DISABLE_MEDIAPIPE_PARAMS = " --define MEDIAPIPE_DISABLE=0"
endif

ifeq ($(MEDIAPIPE_DISABLE),1)
  DISABLE_PARAMS = " --config=mp_off_py_off"
else
  ifeq ($(PYTHON_DISABLE),1)
    DISABLE_PARAMS = " --config=mp_on_py_off"
  else
    DISABLE_PARAMS = " --config=mp_on_py_on"
  endif
endif

FUZZER_BUILD_PARAMS ?= ""
ifeq ($(FUZZER_BUILD),1)
	FUZZER_BUILD_PARAMS = " --define FUZZER_BUILD=1"
endif

STRIP = "always"
BAZEL_DEBUG_BUILD_FLAGS ?= ""
ifeq ($(BAZEL_BUILD_TYPE),dbg)
  BAZEL_DEBUG_BUILD_FLAGS = " --copt=-g -c dbg"
  STRIP = "never"
endif

ifeq ($(MINITRACE),ON)
  MINITRACE_FLAGS=" --copt=-DMTR_ENABLED"
else
  MINITRACE_FLAGS=""
endif

ifeq ($(OV_TRACING_ENABLE),1)
  OV_TRACING_PARAMS = " --define OV_TRACE=1"
else
  OV_TRACING_PARAMS = ""
endif

ifeq ($(findstring ubuntu,$(BASE_OS)),ubuntu)
  TARGET_DISTRO_PARAMS = " --//:distro=ubuntu"
else ifeq ($(findstring redhat,$(BASE_OS)),redhat)
  TARGET_DISTRO_PARAMS = " --//:distro=redhat"
else
  $(error BASE_OS must be either ubuntu or redhat)
endif
CAPI_FLAGS = "--strip=$(STRIP)"$(BAZEL_DEBUG_BUILD_FLAGS)"  --config=mp_off_py_off"$(OV_TRACING_PARAMS)$(TARGET_DISTRO_PARAMS)
BAZEL_DEBUG_FLAGS="--strip=$(STRIP)"$(BAZEL_DEBUG_BUILD_FLAGS)$(DISABLE_PARAMS)$(FUZZER_BUILD_PARAMS)$(OV_TRACING_PARAMS)$(TARGET_DISTRO_PARAMS)$(REPO_ENV)

# Option to Override release image.
# Release image OS *must have* glibc version >= glibc version on BASE_OS:
DIST_OS ?= $(BASE_OS)
DIST_OS_TAG ?= $(BASE_OS_TAG)

ifeq ($(findstring ubuntu,$(BASE_OS)),ubuntu)
  BASE_OS_TAG=$(BASE_OS_TAG_UBUNTU)
  DIST_OS=ubuntu
  ifeq ($(BASE_OS),ubuntu22)
	BASE_OS_TAG=22.04
  endif
  ifeq ($(BASE_OS),ubuntu24)
	BASE_OS_TAG=24.04
  endif
  BASE_IMAGE ?= ubuntu:$(BASE_OS_TAG)
  BASE_IMAGE_RELEASE=$(BASE_IMAGE)
  ifeq ($(BASE_OS_TAG),24.04)
        OS=ubuntu24
	INSTALL_DRIVER_VERSION ?= "25.31.34666"
	DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.4.0-19862-997b5c48447/openvino_toolkit_ubuntu24_2025.4.0.dev20250829_x86_64.tgz
  else ifeq  ($(BASE_OS_TAG),22.04)
        OS=ubuntu22
	INSTALL_DRIVER_VERSION ?= "24.39.31294"
	DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.4.0-19862-997b5c48447/openvino_toolkit_ubuntu22_2025.4.0.dev20250829_x86_64.tgz
  endif
endif
ifeq ($(BASE_OS),redhat)
  BASE_OS_TAG=$(BASE_OS_TAG_REDHAT)
  OS=redhat
  BASE_IMAGE ?= registry.access.redhat.com/ubi9/ubi:$(BASE_OS_TAG_REDHAT)
  BASE_IMAGE_RELEASE=registry.access.redhat.com/ubi9/ubi-minimal:$(BASE_OS_TAG_REDHAT)
  DIST_OS=redhat
  DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.4.0-19862-997b5c48447/openvino_toolkit_rhel8_2025.4.0.dev20250829_x86_64.tgz
  INSTALL_DRIVER_VERSION ?= "24.52.32224"
endif

OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
ifeq ($(BAZEL_BUILD_TYPE),dbg)
  OVMS_CPP_DOCKER_IMAGE:=$(OVMS_CPP_DOCKER_IMAGE)-dbg
endif

OVMS_CPP_IMAGE_TAG ?= latest

OVMS_PYTHON_IMAGE_TAG ?= py

PRODUCT_VERSION ?= "2025.3.0"
PROJECT_VER_PATCH =

$(eval PROJECT_VER_PATCH:=`git rev-parse --short HEAD`)
$(eval PROJECT_VERSION:=${PRODUCT_VERSION}.${PROJECT_VER_PATCH})

OVMS_CPP_CONTAINER_NAME ?= "server-test-${PROJECT_VER_PATCH}-$(shell date +%Y-%m-%d-%H.%M.%S)"
OVMS_CPP_CONTAINER_PORT ?= 9178

PYTHON_CLIENT_TEST_GRPC_PORT ?= 9279
PYTHON_CLIENT_TEST_REST_PORT ?= 9280
PYTHON_CLIENT_TEST_CONTAINER_NAME ?= python-client-test$(shell date +%Y-%m-%d-%H.%M.%S)

TEST_PATH ?= tests/functional/

BUILD_CUSTOM_NODES ?= false

VERBOSE_LOGS ?= OFF

BUILD_ARGS = --build-arg http_proxy=$(HTTP_PROXY)\
	--build-arg https_proxy=$(HTTPS_PROXY)\
	--build-arg no_proxy=$(NO_PROXY)\
	--build-arg ov_source_branch=$(OV_SOURCE_BRANCH)\
	--build-arg ov_source_org=$(OV_SOURCE_ORG)\
	--build-arg ov_contrib_org=$(OV_CONTRIB_ORG)\
	--build-arg ov_use_binary=$(OV_USE_BINARY)\
	--build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL)\
	--build-arg CHECK_COVERAGE=$(CHECK_COVERAGE)\
	--build-arg RUN_TESTS=$(RUN_TESTS)\
	--build-arg OPTIMIZE_BUILDING_TESTS=$(OPTIMIZE_BUILDING_TESTS)\
	--build-arg RUN_GPU_TESTS=$(RUN_GPU_TESTS)\
	--build-arg FUZZER_BUILD=$(FUZZER_BUILD)\
	--build-arg debug_bazel_flags=$(BAZEL_DEBUG_FLAGS)\
	--build-arg minitrace_flags=$(MINITRACE_FLAGS) \
	--build-arg CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)\
	--build-arg PROJECT_VERSION=$(PROJECT_VERSION)\
	--build-arg BASE_IMAGE=$(BASE_IMAGE)\
	--build-arg BASE_OS=$(BASE_OS)\
	--build-arg ov_contrib_branch=$(OV_CONTRIB_BRANCH)\
	--build-arg ov_tokenizers_branch=$(OV_TOKENIZERS_BRANCH)\
	--build-arg INSTALL_RPMS_FROM_URL=$(INSTALL_RPMS_FROM_URL)\
	--build-arg INSTALL_DRIVER_VERSION=$(INSTALL_DRIVER_VERSION)\
	--build-arg RELEASE_BASE_IMAGE=$(BASE_IMAGE_RELEASE)\
	--build-arg JOBS=$(JOBS)\
	--build-arg CAPI_FLAGS=$(CAPI_FLAGS)\
	--build-arg VERBOSE_LOGS=$(VERBOSE_LOGS)


.PHONY: default docker_build \

default: docker_build

venv:$(ACTIVATE)
	@echo $(BUILD_ARGS)
	@echo -n "Using venv "
	@. $(ACTIVATE); python3 --version

venv-style:$(ACTIVATE_STYLE)
	@echo $(BUILD_ARGS)
	@echo -n "Using venv "
	@python3 --version
spell: venv-style
	@{ git ls-files; git diff --name-only --cached; } | sort | uniq | xargs $(VIRTUALENV_STYLE_DIR)/bin/codespell --skip "spelling-whitelist.txt" 2>&1 | grep -vFf spelling-whitelist.txt; if [ $$? != 1 ]; then exit 1; fi
	@echo "Spelling check completed."

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@python3 -m pip install virtualenv
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip3 install --upgrade pip
	@. $(ACTIVATE); pip3 install -vUqq "setuptools<80"
	@. $(ACTIVATE); pip3 install -qq -r tests/requirements.txt
	@touch $(ACTIVATE)

$(ACTIVATE_STYLE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_STYLE_DIR)..."
	@python3 -m pip install virtualenv
	@test -d $(VIRTUALENV_STYLE_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_STYLE_DIR)
	@. $(ACTIVATE_STYLE); pip3 install --upgrade pip
	@. $(ACTIVATE_STYLE); pip3 install -vUqq "setuptools<80"
	@. $(ACTIVATE_STYLE); pip3 install -qq -r ci/style_requirements.txt
	@touch $(ACTIVATE_STYLE)

cppclean: venv-style
	@echo "Checking cppclean..."
	@bash -c "./ci/cppclean.sh"

style: venv-style spell clang-format-check cpplint cppclean

hadolint:
	@echo "Checking SDL requirements..."
	@echo "Checking docker files..."
	@./tests/hadolint.sh

bandit:
	@echo "Checking python files..."
	@. $(ACTIVATE_STYLE); bash -c "./ci/bandit.sh"

license-headers:
	@echo "Checking license headers in files..."
	@. $(ACTIVATE_STYLE); bash -c "python3 ./ci/lib_search.py . > missing_headers.txt"
	@if ! grep -FRq "All files have headers" missing_headers.txt; then\
        echo "Files with missing headers";\
        cat missing_headers.txt;\
		exit 1;\
	fi
	@rm missing_headers.txt

sdl-check: venv-style hadolint bandit license-headers

	@echo "Checking forbidden functions in files..."
	@. $(ACTIVATE_STYLE); bash -c "python3 ./ci/lib_search.py . functions > forbidden_functions.txt"
	@if ! grep -FRq "All files checked for forbidden functions" forbidden_functions.txt; then\
		error Run python3 ./ci/lib_search.py . functions - to see forbidden functions file list.;\
	fi
	@rm forbidden_functions.txt

cpplint: venv-style
	@echo "Style-checking codebase..."
	@. $(ACTIVATE_STYLE); echo ${PWD}; cpplint ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}

clang-format: venv-style
	@echo "Formatting files with clang-format.."
	@. $(ACTIVATE_STYLE); find ${STYLE_CHECK_DIRS} -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec clang-format -style=file -i {} \;

clang-format-check: clang-format
	@echo "Checking if clang-format changes were committed ..."
	@git diff --exit-code || (echo "clang-format changes not committed. Commit those changes first"; exit 1)
	@git diff --exit-code --staged || (echo "clang-format changes not committed. Commit those changes first"; exit 1)

.PHONY: docker_build
docker_build: ovms_builder_image targz_package ovms_release_images
ovms_builder_image:
ifeq ($(PYTHON_DISABLE),0)
  ifeq ($(MEDIAPIPE_DISABLE),1)
	@echo "Cannot build model server with Python support without building with Mediapipe enabled. Use 'MEDIAPIPE_DISABLE=0 PYTHON_DISABLE=0 make docker_build'"; exit 1 ;
  endif
endif
ifeq ($(CHECK_COVERAGE),1)
  ifeq ($(RUN_TESTS),0)
	@echo "Cannot test coverage without running tests. Use 'CHECK_COVERAGE=1 RUN_TESTS=1 make docker_build'"; exit 1 ;
  endif
endif
ifeq ($(FUZZER_BUILD),1)
  ifeq ($(RUN_TESTS),1)
	@echo "Cannot run tests for now with fuzzer build"; exit 1 ;
  endif
  ifeq ($(CHECK_COVERAGE),1)
	@echo "Cannot check coverage with fuzzer build"; exit 1 ;
  endif
  ifeq ($(BASE_OS),redhat)
	@echo "Cannot run fuzzer with redhat"; exit 1 ;
  endif
endif
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
	@docker pull $(BASE_IMAGE)
  ifeq ($(BASE_OS),redhat)
	@docker pull registry.access.redhat.com/ubi9/ubi-minimal:$(BASE_OS_TAG_REDHAT)
  endif
endif
ifeq ($(USE_BUILDX),true)
	$(eval BUILDX:=buildx)
endif

ifeq ($(BUILD_CUSTOM_NODES),true)
	@echo "Building custom nodes"
	@cd src/custom_nodes && make USE_BUILDX=$(USE_BUILDX) NO_DOCKER_CACHE=$(NO_DOCKER_CACHE) BASE_OS=$(OS) BASE_IMAGE=$(BASE_IMAGE) 
endif
	@echo "Building docker image $(BASE_OS)"
	# Provide metadata information into image if defined
	@mkdir -p .workspace
ifneq ($(OVMS_METADATA_FILE),)
	@cp $(OVMS_METADATA_FILE) .workspace/metadata.json
else
	@touch .workspace/metadata.json
endif
	@cat .workspace/metadata.json
	docker $(BUILDX) build $(NO_CACHE_OPTION) -f Dockerfile.$(DIST_OS) . \
		$(BUILD_ARGS) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--target=build

targz_package:
	docker $(BUILDX) build -f Dockerfile.$(DIST_OS) . \
		$(BUILD_ARGS) \
		--build-arg BUILD_IMAGE=$(BUILD_IMAGE) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) \
		--target=pkg && \
	rm -vrf dist/$(OS) && mkdir -p dist/$(OS) && \
	ID=$$(docker create $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG)) && \
	docker cp $$ID:/ovms_pkg/$(OS) dist/ && \
	docker rm $$ID
	cd dist/$(OS) && sha256sum --check ovms.tar.gz.sha256

ovms_release_images:
ifeq ($(USE_BUILDX),true)
	$(eval BUILDX:=buildx)
	$(eval NO_CACHE_OPTION:=--no-cache-filter release)
endif
ifeq ($(BASE_OS),redhat)
	$(eval NPU:=0)
else
	$(eval NPU:=1)
endif
	docker $(BUILDX) build $(NO_CACHE_OPTION) -f Dockerfile.$(DIST_OS) . \
		$(BUILD_ARGS) \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--target=release && \
	docker $(BUILDX) build $(NO_CACHE_OPTION) -f Dockerfile.$(DIST_OS) . \
		$(BUILD_ARGS) \
		--build-arg GPU=1 \
		--build-arg NPU=$(NPU) \
		-t $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--target=release && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-gpu:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-gpu$(IMAGE_TAG_SUFFIX)
ifeq ($(BUILD_NGINX), 1)
	cd extras/nginx-mtls-auth && \
	http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) no_proxy=$(NO_PROXY) ./build.sh "$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)" "$(DIST_OS)" && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-nginx-mtls$(IMAGE_TAG_SUFFIX)
endif

get_gpl_mpl_packages:
ifeq ($(findstring ubuntu,$(BASE_OS)),ubuntu)
	@docker run -u 0 --entrypoint bash $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) -c 'dpkg --get-selections | sed "s/\t//g" | sed "s/install//g" | cut -d":" -f1 | tr -d "\r"' > ubuntu.txt
	@-docker run -u 0 --entrypoint bash -v ${PWD}:/ovms $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) -c 'cd /ovms ; cat ubuntu.txt | tr -d "\r" | xargs -I % bash -c "grep -l -e GPL -e MPL /usr/share/doc/%/copyright" 2> /dev/null' > sources.txt
	@docker run -u 0 --entrypoint bash -v ${PWD}:/ovms $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) -c 'sed -Ei "s/^Types: deb$$/Types: deb deb-src/" /etc/apt/sources.list.d/ubuntu.sources ; apt update ; cd /ovms ; d="ovms_ubuntu_$(OVMS_CPP_IMAGE_TAG)" ;mkdir "$$d" ; cd "$$d" ; for I in `cat /ovms/sources.txt | cut -d"/" -f5`; do apt-get source -q --download-only $$I; done'
	@rm ubuntu.txt sources.txt
endif
ifeq ($(BASE_OS),redhat)
	touch base_packages.txt
	docker run registry.access.redhat.com/ubi9-minimal:9.6 rpm -qa  --qf "%{NAME}\n" | sort > base_packages.txt
	docker run --entrypoint rpm $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) -qa  --qf "%{NAME}\n" | sort > all_packages.txt
	rm -rf ovms_rhel_$(OVMS_CPP_IMAGE_TAG)
	mkdir ovms_rhel_$(OVMS_CPP_IMAGE_TAG)
	docker run -u 0 -v ${PWD}:/pkgs -v ${PWD}/ovms_rhel_$(OVMS_CPP_IMAGE_TAG):/srcs --entrypoint bash -it $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) -c ' \
	grep -v -f /pkgs/base_packages.txt /pkgs/all_packages.txt | while read line ;	do package=`echo $$line` ; \
	rpm -qa --qf "%{name}: %{license}\n" | grep -e GPL -e MPL ;\
	exit_status=$$? ; \
	if [ $$exit_status -eq 0 ]; then \
			cd /srcs ; \
			microdnf download -y $$package ; \
	fi ; done'
	@rm base_packages.txt all_packages.txt
endif

release_image:
ifeq ($(USE_BUILDX),true)
	$(eval BUILDX:=buildx)
	$(eval NO_CACHE_OPTION:=--no-cache-filter release)
endif
	docker $(BUILDX) build $(NO_CACHE_OPTION) -f Dockerfile.$(DIST_OS) . \
		$(BUILD_ARGS) \
		--build-arg BUILD_IMAGE=$(BUILD_IMAGE) \
		--build-arg GPU=$(GPU) \
		--build-arg NPU=$(NPU) \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--target=release
ifeq ($(BUILD_NGINX), 1)
	cd extras/nginx-mtls-auth && \
	http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) no_proxy=$(NO_PROXY) ./build.sh "$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)" "$(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG)" "$(DIST_OS)" && \
	docker tag $(OVMS_CPP_DOCKER_IMAGE)-nginx-mtls:$(OVMS_CPP_IMAGE_TAG) $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)-nginx-mtls
endif

python_image:
	@docker build --build-arg http_proxy="$(http_proxy)" --build-arg https_proxy="$(https_proxy)" --build-arg IMAGE_NAME=$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) -f demos/python_demos/Dockerfile.$(DIST_OS) demos/python_demos/ -t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_PYTHON_IMAGE_TAG)


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
	
test_checksec: venv
	@echo "Running checksec on libovms_shared library..."
	@docker rm -f $(OVMS_CPP_CONTAINER_NAME) || true
	@docker create -ti --name $(OVMS_CPP_CONTAINER_NAME) $(OVMS_CPP_DOCKER_IMAGE)-pkg:$(OVMS_CPP_IMAGE_TAG) bash
	@docker cp $(OVMS_CPP_CONTAINER_NAME):/libovms_shared.so /tmp
	@docker cp $(OVMS_CPP_CONTAINER_NAME):/ovms_release/bin/ovms /tmp
	@docker rm -f $(OVMS_CPP_CONTAINER_NAME) || true
	@. $(ACTIVATE); checksec -j /tmp/libovms_shared.so | jq '.[]| join(",")' > checksec.txt
	@if ! grep -FRq "Full,true,true,DSO,false,true,true,true" checksec.txt; then\
 		echo "ERROR: OVMS shared library security settings changed. Run checksec on ovms shared library and fix issues." ; \
		. $(ACTIVATE); checksec /tmp/libovms_shared.so ;\
		exit 1;\
	fi
	@echo "Running checksec on ovms binary..."
	@. $(ACTIVATE); checksec -j /tmp/ovms | jq '.[]| join(",")' > checksec.txt
	@if ! grep -FRq "Full,true,true,PIE,false,true,true,true" checksec.txt; then\
 		echo "ERROR: OVMS binary security settings changed. Run checksec on ovms binary and fix issues."; \
		. $(ACTIVATE); checksec /tmp/ovms ; \
		exit 1;\
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

test_python_clients:
	@echo "Prepare docker image"
	@docker build . -f tests/python/Dockerfile -t python_client_test
	@echo "Dropping test container if exist"
	@docker rm --force $(PYTHON_CLIENT_TEST_CONTAINER_NAME) || true
	@echo "Download models"
	@if [ ! -d "tests/python/models" ]; then cd tests/python && \
		mkdir models && \
		docker run -u $(id -u):$(id -g) -v ${PWD}/tests/python/models:/models openvino/ubuntu20_dev:2024.6.0 omz_downloader --name resnet-50-tf --output_dir /models && \
		docker run -u $(id -u):$(id -g) -v ${PWD}/tests/python/models:/models:rw openvino/ubuntu20_dev:2024.6.0 omz_converter --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32 && \
		docker run -u $(id -u):$(id -g) -v ${PWD}/tests/python/models:/models:rw openvino/ubuntu20_dev:2024.6.0 mv /models/public/resnet-50-tf/FP32 /models/public/resnet-50-tf/1; fi
	@echo "Start test container"
	@docker run -d --rm --name $(PYTHON_CLIENT_TEST_CONTAINER_NAME) -v ${PWD}/tests/python/models/public/resnet-50-tf:/models/public/resnet-50-tf -p $(PYTHON_CLIENT_TEST_REST_PORT):8000 -p $(PYTHON_CLIENT_TEST_GRPC_PORT):9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000 && \
		sleep 10
	@echo "Run tests"
	@exit_status=0 docker run --rm --network="host" python_client_test --grpc=$(PYTHON_CLIENT_TEST_GRPC_PORT) --rest=$(PYTHON_CLIENT_TEST_REST_PORT) --verbose --fastFail || exit_status=$?
	@echo "Removing test container"
	@docker rm --force $(PYTHON_CLIENT_TEST_CONTAINER_NAME)
	@exit $(exit_status)

tools_get_deps:
	cd tools/deps/$(OS) && docker build --build-arg http_proxy="$(http_proxy)" --build-arg https_proxy="$(https_proxy)" -t  $(OVMS_CPP_DOCKER_IMAGE)-deps:$(OVMS_CPP_IMAGE_TAG) .
	-docker rm -f ovms-$(BASE_OS)-deps
	docker run -d --rm --name  ovms-$(BASE_OS)-deps  $(OVMS_CPP_DOCKER_IMAGE)-deps:$(OVMS_CPP_IMAGE_TAG)
	sleep 5
	docker cp ovms-$(OS)-deps:/root/rpms.tar.xz ./
	sleep 5
	-docker rm -f ovms-$(OS)-deps
	@echo "Success! Dependencies saved to rpms.tar.xz in this directory"

cpu_extension:
	cd src/example/SampleCpuExtension && \
	docker build -f Dockerfile.$(DIST_OS) -t sample_cpu_extension:latest \
		--build-arg http_proxy=${http_proxy} \
		--build-arg https_proxy=${https_proxy} \
		--build-arg no_proxy=${no_proxy} \
		--build-arg DLDT_PACKAGE_URL=${DLDT_PACKAGE_URL} \
		--build-arg APT_OV_PACKAGE=${APT_OV_PACKAGE} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} .
	mkdir -p ./lib/${OS}
	docker cp $$(docker create --rm sample_cpu_extension:latest):/workspace/libcustom_relu_cpu_extension.so ./lib/${OS}

prepare_models:
	./prepare_llm_models.sh ${TEST_LLM_PATH}
ifeq ($(RUN_GPU_TESTS),1)
	./prepare_gpu_models.sh ${GPU_MODEL_PATH}
endif

run_unit_tests: prepare_models
	docker rm -f $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX)
ifeq ($(RUN_GPU_TESTS),1)
	docker run \
		--name $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		--device=/dev/dri \
		--group-add=$(shell stat -c "%g" /dev/dri/render* | head -n 1) \
		-u 0 \
		-v $(shell realpath ./run_unit_tests.sh):/ovms/./run_unit_tests.sh \
		-v $(shell realpath ${GPU_MODEL_PATH}):/ovms/src/test/face_detection_adas/1:ro \
		-v $(shell realpath ${TEST_LLM_PATH}):/ovms/src/test/llm_testing:ro \
		-e https_proxy=${https_proxy} \
		-e RUN_TESTS=1 \
		-e RUN_GPU_TESTS=$(RUN_GPU_TESTS) \
		-e JOBS=$(JOBS) \
		-e debug_bazel_flags=${BAZEL_DEBUG_FLAGS} \
		$(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		./run_unit_tests.sh ;\
		exit_code=$$? ;\
		docker container cp $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX):/ovms/test_logs.tar.gz . ;\
		docker container cp $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX):/ovms/linux_tests_summary.log . ;\
		docker rm -f $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) ;\
		exit $$exit_code
else
	docker run \
		--name $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		-v $(shell realpath ./run_unit_tests.sh):/ovms/./run_unit_tests.sh \
		-v $(shell realpath ${TEST_LLM_PATH}):/ovms/src/test/llm_testing:ro \
		-e https_proxy=${https_proxy} \
		-e RUN_TESTS=1 \
		-e JOBS=$(JOBS) \
		-e debug_bazel_flags=${BAZEL_DEBUG_FLAGS} \
		$(OVMS_CPP_DOCKER_IMAGE)-build:$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) \
		./run_unit_tests.sh ;\
		exit_code=$$? ; \
		docker container cp $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX):/ovms/test_logs.tar.gz . ;\
		docker container cp $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX):/ovms/linux_tests_summary.log . ;\
		docker rm -f $(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) ;\
		exit $$exit_code
endif


run_lib_files_test:
	docker run --entrypoint bash -v $(realpath tests/file_lists):/test $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)$(IMAGE_TAG_SUFFIX) ./test/test_release_files.sh ${BAZEL_DEBUG_FLAGS} > file_test.log 2>&1 ; exit_status=$$? ; tail -200 file_test.log ; exit $$exit_status
