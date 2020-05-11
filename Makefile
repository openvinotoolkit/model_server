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
# limitations under the License.
#

VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"
STYLE_CHECK_OPTS := --extensions=hpp,cc,cpp,h \
	--output=vs7 \
	--recursive \
	--linelength=120 \
	--filter=-build/c++11,-runtime/references,-whitespace/indent,-build/include_order,-runtime/indentation_namespace,-build/namespaces,-whitespace/line_length,-runtime/string,-readability/casting,-runtime/explicit,-readability/todo
STYLE_CHECK_DIRS := src
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"

OVMS_CPP_DOCKER_IMAGE ?= ovms
OVMS_CPP_IMAGE_TAG ?= latest
OVMS_CPP_CONTAINTER_NAME ?= server-test
OVMS_CPP_CONTAINTER_PORT ?= 9178

TEST_PATH ?= tests/functional/

.PHONY: default docker_build \

default: docker_build

venv: $(ACTIVATE)
	@echo -n "Using venv "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip$(PY_VERSION) install --upgrade pip
	@. $(ACTIVATE); pip$(PY_VERSION) install -vUqq setuptools
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r tests/requirements.txt
	@touch $(ACTIVATE)

style: venv
	@echo "Style-checking codebase..."
	@. $(ACTIVATE); echo ${PWD}; cpplint ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}

docker_build:
	@echo "Building docker image"
	@echo docker build . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)
	@docker build . \
		--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" \
		-t $(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG)

test_perf: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(HOME)/resnet50:/models/resnet50 \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet --model_path /models/resnet50 --port $(OVMS_CPP_CONTAINTER_PORT); sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py \
		--images_numpy_path tests/performance/imgs.npy \
		--labels_numpy_path tests/performance/labels.npy \
		--iteration 1000 \
		--batchsize 1 \
		--report_every 100 \
		--input_name data
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)

test_throughput: venv
	@echo "Dropping test container if exist"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME) || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name $(OVMS_CPP_CONTAINTER_NAME) \
		-v $(HOME)/resnet50:/models/resnet50 \
		-p $(OVMS_CPP_CONTAINTER_PORT):$(OVMS_CPP_CONTAINTER_PORT) \
		$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		--model_name resnet \
		--model_path /models/resnet50 \
		--port $(OVMS_CPP_CONTAINTER_PORT); \
		sleep 5
	@echo "Running throughput test"
	@. $(ACTIVATE); cd tests/performance; ./grpc_throughput.sh 28 \
		--images_numpy_path imgs.npy \
		--labels_numpy_path labels.npy \
		--iteration 500 \
		--batchsize 1 \
		--input_name data
	@echo "Removing test container"
	@docker rm --force $(OVMS_CPP_CONTAINTER_NAME)

test_functional: venv
	@. $(ACTIVATE); pytest --json=report.json -v \
		--image=$(OVMS_CPP_DOCKER_IMAGE):$(OVMS_CPP_IMAGE_TAG) \
		$(TEST_PATH)
