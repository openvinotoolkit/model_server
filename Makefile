#
# Copyright (c) 2018 Intel Corporation
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

PY_VERSION := 3
VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"
STYLEVIRTUALENV_DIR=".styleenv$(PY_VERSION)"
STYLE_CHECK_OPTS := --exclude=ie_serving/tensorflow_serving_api
STYLE_CHECK_DIRS := tests ie_serving setup.py
TEST_OPTS :=
TEST_DIRS ?= tests/
CONFIG := "$(CONFIG)"
ML_DIR := "$(MK_DIR)"
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
OVMS_VERSION := "2020_R1"
DLDT_PACKAGE_URL := "$(dldt_package_url)"
TEST_MODELS_DIR = /tmp/ovms_models

.PHONY: default install uninstall requirements \
	venv test unit_test coverage style dist clean \

default: install

venv: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE): requirements.txt requirements-dev.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip$(PY_VERSION) install --upgrade pip
	@. $(ACTIVATE); pip$(PY_VERSION) install -vUqq setuptools
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r requirements.txt
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r requirements-dev.txt
	@touch $(ACTIVATE)

install: venv
	@. $(ACTIVATE); pip$(PY_VERSION) install .

run: venv install
	@. $(ACTIVATE); python ie_serving/main.py --config "$CONFIG"

unit: venv
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/unit/

coverage: venv
	@echo "Computing unit test coverage..."
	@. $(ACTIVATE); coverage run --source=ie_serving -m pytest $(TEST_DIRS)/unit/ && coverage report --fail-under=70

test: venv
	@echo "Executing functional tests..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/ --test_dir $(TEST_MODELS_DIR)

test_local_only: venv
	@echo "Executing functional tests with only local models..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_batching.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_mapping.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_single_model.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_version_policy.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_versions_handling.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_versions_handling.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_update.py

style: venv
	@echo "Style-checking codebase..."
	@. $(ACTIVATE); flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)

clean_pyc:
	@echo "Removing .pyc files..."
	@find . -name '*.pyc' -exec rm -f {} \;

clean: clean_pyc
	@echo "Removing virtual env files..."
	@rm -rf $(VIRTUALENV_DIR)

docker_build_apt_ubuntu:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: 2020_R1 apt >> version
	@echo docker build -f Dockerfile --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .
	@docker build -f Dockerfile --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .

docker_build_ov_base:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: 2020_R1 apt >> version
	@echo docker build -f Dockerfile_openvino_base --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .
	@docker build -f Dockerfile_openvino_base --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .

docker_build_bin:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: `ls -1 l_openvino_toolkit*` >> version
	@echo docker build -f Dockerfile_binary_openvino --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg DLDT_PACKAGE_URL="$(DLDT_PACKAGE_URL)" -t ie-serving-py:latest .
	@docker build -f Dockerfile_binary_openvino --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg DLDT_PACKAGE_URL="$(DLDT_PACKAGE_URL)" -t ie-serving-py:latest .

docker_build_clearlinux:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: 2019_R3 clearlinux >> version
	@echo docker build -f Dockerfile_clearlinux --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .
	@docker build -f Dockerfile_clearlinux --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t ie-serving-py:latest .

docker_run:
	@echo "Starting the docker container with serving model"
	@docker run --rm -d --name ie-serving-py-test-multi -v /tmp/test_models/saved_models/:/opt/ml:ro -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
