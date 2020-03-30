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
STYLE_CHECK_OPTS := --extensions=hpp,cc,cpp,h --recursive
STYLE_CHECK_DIRS := src
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
NO_PROXY := "$(no_proxy)"

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
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r tests/performance/requirements.txt
	@touch $(ACTIVATE)

style:
	@echo "Style-checking codebase..."
	@. $(ACTIVATE); echo ${PWD}; cpplint ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}

docker_build:
	@echo "Building docker image"
	@echo docker build . --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t cpp-experiments
	@docker build . --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t cpp-experiments

test_perf: venv
	@echo "Dropping test container if exist"
	@docker rm --force server-test || true
	@echo "Starting docker image"
	@./tests/performance/download_model.sh
	@docker run -d --name server-test -v /tmp/resnet50:/models/resnet50 -p 9178:9178 cpp-experiments:latest ; sleep 5
	@echo "Running latency test"
	@. $(ACTIVATE); python3 tests/performance/grpc_latency.py --images_numpy_path tests/performance/imgs.npy --iteration 1000 --batchsize 1 --report_every 100
	@echo "Removing test container"
	@docker rm --force server-test

