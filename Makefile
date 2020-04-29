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
AMS_VIRTUALENV_DIR := .venv_ams
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate
AMS_ACTIVATE := $(AMS_VIRTUALENV_DIR)/bin/activate
STYLEVIRTUALENV_DIR=".styleenv$(PY_VERSION)"
STYLE_CHECK_OPTS := --exclude=ie_serving/tensorflow_serving_api
STYLE_CHECK_DIRS := tests ie_serving setup.py
TEST_OPTS :=
TEST_DIRS ?= tests/
AMS_EXAMPLE ?= extras/ams_wrapper/
CONFIG := "$(CONFIG)"
ML_DIR := "$(MK_DIR)"
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
OVMS_VERSION := "2020_R1"
DLDT_PACKAGE_URL := "$(dldt_package_url)"
TEST_MODELS_DIR = /tmp/ovms_models
DOCKER_OVMS_TAG ?= ie-serving-py:latest

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

install: $(ACTIVATE)
	@. $(ACTIVATE); pip$(PY_VERSION) install .

run: $(ACTIVATE) install
	@. $(ACTIVATE); python ie_serving/main.py --config "$CONFIG"

unit: $(ACTIVATE)
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/unit/

coverage: $(ACTIVATE)
	@echo "Computing unit test coverage..."
	@. $(ACTIVATE); coverage run --source=ie_serving -m pytest $(TEST_DIRS)/unit/ && coverage report --fail-under=70

ams_test: $(AMS_EXAMPLE)requirements.txt $(AMS_EXAMPLE)requirements-dev.txt
	echo "Running ams image preprocessing tests" 
	test -d $(AMS_VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(AMS_VIRTUALENV_DIR)
	@. $(AMS_ACTIVATE); pip install -qq -r $(AMS_EXAMPLE)requirements.txt
	@. $(AMS_ACTIVATE); pip install -qq -r $(AMS_EXAMPLE)requirements-dev.txt
	@. $(AMS_ACTIVATE); test -d $(AMS_EXAMPLE)tests/test_images || (sh $(AMS_EXAMPLE)tests/get_test_images.sh && mv test_images $(AMS_EXAMPLE)tests/)
	@. $(AMS_ACTIVATE); pytest  $(AMS_EXAMPLE)tests/

ams_clean: 
	@echo "Removing ams virtual env files and test images ..."
	@rm -rf $(AMS_VIRTUALENV_DIR)	
	@rm -rf $(AMS_EXAMPLE)tests/test_images

test: $(ACTIVATE)
	@echo "Executing functional tests..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/ --test_dir $(TEST_MODELS_DIR)

test_local_only: $(ACTIVATE)
	@echo "Executing functional tests with only local models..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_batching.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_mapping.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_single_model.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_version_policy.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_versions_handling.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_model_versions_handling.py
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/test_update.py

style: $(ACTIVATE)
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
	@echo OpenVINO version: $(OVMS_VERSION) apt >> version
	@echo docker build -f Dockerfile --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .
	@docker build -f Dockerfile --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .

docker_build_ov_base:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: 2020_R1 apt >> version
	@echo docker build -f Dockerfile_openvino_base --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .
	@docker build -f Dockerfile_openvino_base --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .

docker_build_bin:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: `ls -1 l_openvino_toolkit*` >> version
	@echo docker build -f Dockerfile_binary_openvino --build-arg no_proxy=$(no_proxy) --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg DLDT_PACKAGE_URL="$(DLDT_PACKAGE_URL)" -t $(DOCKER_OVMS_TAG) .
	@docker build -f Dockerfile_binary_openvino --build-arg no_proxy=$(no_proxy) --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg DLDT_PACKAGE_URL="$(DLDT_PACKAGE_URL)" -t $(DOCKER_OVMS_TAG) .

docker_build_clearlinux:
	@echo "Building docker image"
	@echo OpenVINO Model Server version: $(OVMS_VERSION) > version
	@echo Git commit: `git rev-parse HEAD` >> version
	@echo OpenVINO version: 2019_R3 clearlinux >> version
	@echo docker build -f Dockerfile_clearlinux --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .
	@docker build -f Dockerfile_clearlinux --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy="$(HTTPS_PROXY)" -t $(DOCKER_OVMS_TAG) .

docker_run:
	@echo "Starting the docker container with serving model"
	@docker run --rm -d --name ie-serving-py-test-multi -v /tmp/test_models/saved_models/:/opt/ml:ro -p 9001:9001 -t $(DOCKER_OVMS_TAG) /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
