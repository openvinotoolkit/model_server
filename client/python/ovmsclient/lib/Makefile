#
# Copyright (c) 2021 Intel Corporation
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
VIRTUALENV_DIR := .venv-ovmsclient
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"

PACKAGE_PATH := dist/ovmsclient-2023.0-py3-none-any.whl
TEST_TYPE := BASIC

.PHONY: build-deps build-package build test clean style

venv:$(ACTIVATE)
	@echo -n "Using virtual environment: $(VIRTUALENV_DIR)\n"
	@. $(ACTIVATE); python3 --version

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip3 install --upgrade pip
	@. $(ACTIVATE); pip3 install -r tests/requirements.txt
	@touch $(ACTIVATE)

update-packages:
	@echo "Updating crucial packages"
	@pip3 install --upgrade pip
	@pip3 install "setuptools>=40.1.0"
	@pip3 install "wheel>=0.23.0"

build-deps: update-packages
	@echo "Building dependencies"
	@python3 setup.py build_apis

build-package: update-packages
	@echo "Building ovmsclient package"
	@python3 setup.py bdist_wheel

build: build-deps build-package
	@echo "Build completed successfully. The wheel should be present in dist/ directory"

test: venv
	@echo "Installing ovmsclient package from: $(PACKAGE_PATH)"
	@. $(ACTIVATE); pip3 install $(PACKAGE_PATH) --force-reinstall
	@echo "Running ovmsclient tests"
	@. $(ACTIVATE); pytest -v tests/
ifeq ($(TEST_TYPE), FULL)
	@echo "Installing NumPy version: 1.23"
	@. $(ACTIVATE); pip3 install numpy==1.23
	@echo "Rerunning ovmsclient tests with NumPy 1.23"
	@. $(ACTIVATE); pytest -v tests/
endif

clean:
	@rm -rf tf/ tfs/ compiled_protos/ build/ dist/ ovmsclient/tfs_compat/protos

style: venv
	@. $(ACTIVATE); pip3 install flake8==5.0.1
	@echo "Running ovmsclient style checks",
	@. $(ACTIVATE); flake8 ovmsclient tests/ --max-line-length=100 --count
